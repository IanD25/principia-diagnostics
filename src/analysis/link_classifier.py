"""
link_classifier.py — LLM-powered link type classifier for the DS Wiki knowledge graph.

SCOPE: Currently DS Wiki–specific (reads WIC/MF fields from ds_wiki.db schema, writes
       back to the links table). Generalisation path: accept any SQLite graph db with
       entries + links tables and a configurable text field, so it can improve link
       quality in RRP bundles during intake (Pass 1 post-processing).
       Track in: Phase 3 / intake enrichment milestone.

Pipeline
--------
  BGE cosine filter  →  content extraction  →  LLM classification  →  DB insertion
  (sim_threshold)       (WIC + MF per pair)    (link_type + conf)     (≥ min_confidence)

Two modes:
  1. Interactive (no API key needed):
       classifier.get_candidates()          → List[CandidatePair]
       classifier.format_triage_prompt()    → str  (paste to Claude / MCP)

  2. Batch (requires ANTHROPIC_API_KEY env var):
       classifier.batch_classify(pairs)     → List[ClassificationResult]

Usage
-----
    from link_classifier import LinkClassifier
    lc = LinkClassifier()
    candidates = lc.get_candidates(sim_threshold=0.78, max_pairs=60)
    prompt     = lc.format_triage_prompt(candidates)   # interactive
    results    = lc.batch_classify(candidates)          # automated
    inserted   = lc.insert_results(results, min_confidence=0.80)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import SOURCE_DB, HISTORY_DB, score_to_tier  # noqa: E402


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class EntryContent:
    entry_id:   str
    title:      str
    domain:     str
    entry_type: str
    wic:        str   # What It Claims
    math_form:  str   # Mathematical Form


@dataclass
class CandidatePair:
    entry_a:    EntryContent
    entry_b:    EntryContent
    similarity: float


@dataclass
class FewShotExample:
    source_id:    str
    source_title: str
    source_domain: str
    source_wic:   str
    source_mf:    str
    target_id:    str
    target_title: str
    target_domain: str
    target_wic:   str
    target_mf:    str
    link_type:    str
    description:  str


@dataclass
class ClassificationResult:
    source_id:    str
    source_label: str
    target_id:    str
    target_label: str
    has_link:     bool
    link_type:    Optional[str]    = None
    confidence:   float            = 0.0
    description:  Optional[str]    = None
    reasoning:    Optional[str]    = None
    similarity:   float            = 0.0


# ── Link type definitions (mirrors link_type_definitions table) ────────────────

LINK_TYPE_DEFINITIONS: Dict[str, str] = {
    "generalizes":  "Target is a special case of Source",
    "derives from": "Source is logically derived from Target",
    "constrains":   "Source sets a boundary condition on Target",
    "couples to":   "Source and Target share a state variable",
    "predicts for": "Source generates a testable claim about Target",
    "tensions with":"Source and Target offer competing explanations",
    "analogous to": "Same mathematical structure, different domain",
    "tests":        "Source provides evidence for/against Target",
    "implements":   "Source is the mathematical method used by Target",
}

VALID_LINK_TYPES = set(LINK_TYPE_DEFINITIONS.keys())


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a structural-relationship classifier for a scientific knowledge graph.
Your task: decide whether two scientific laws/principles have a structural \
relationship, and if so, classify its type.

LINK TYPES (pick exactly one, or output has_link=false):
{link_type_block}

RULES:
1. Only classify STRUCTURAL relationships (same mathematical form, derivation, \
generalization, shared state variables, competing explanations). Do NOT classify \
mere topic overlap, historical association, or same-domain proximity.
2. "analogous to" = same mathematical skeleton in different domains \
(e.g., Fick↔Fourier, Fermat↔Least Action). Requires identical equation structure.
3. Directionality matters for "derives from", "generalizes", "implements", \
"constrains" — check the definitions carefully.
4. If genuinely uncertain, prefer has_link=false over a low-confidence guess.
5. Description must be 40–80 words: state the specific structural relationship, \
name the shared variable or equation form, and explain why the link type is correct.

Respond ONLY with valid JSON — no markdown fences, no extra text.\
"""

_LINK_TYPE_BLOCK = "\n".join(
    f'  "{lt}": {defn}' for lt, defn in LINK_TYPE_DEFINITIONS.items()
)

_PAIR_TEMPLATE = """\
ENTRY A — {a_id} ({a_domain})
  Title: {a_title}
  What It Claims: {a_wic}
  Mathematical Form: {a_mf}

ENTRY B — {b_id} ({b_domain})
  Title: {b_title}
  What It Claims: {b_wic}
  Mathematical Form: {b_mf}

Cosine similarity: {sim:.4f}
"""

_EXAMPLE_TEMPLATE = """\
--- Example ({link_type}) ---
{pair_block}
Classification:
{{"has_link": true, "link_type": "{link_type}", "confidence": 0.95,
 "description": "{description}", "reasoning": "structural match confirmed"}}\
"""

_BATCH_PAIR_TEMPLATE = """\
=== PAIR {n} ===
{pair_block}
"""


# ── Core class ─────────────────────────────────────────────────────────────────

class LinkClassifier:
    """
    Generates candidate pairs, formats prompts, classifies via LLM, inserts results.

    Parameters
    ----------
    source_db  : path to ds_wiki.db
    history_db : path to wiki_history.db
    """

    N_EXAMPLES_PER_TYPE = 2    # few-shot examples per link type in triage prompts

    def __init__(
        self,
        source_db:  Path | str = SOURCE_DB,
        history_db: Path | str = HISTORY_DB,
    ) -> None:
        self.source_db  = Path(source_db)
        self.history_db = Path(history_db)
        if not self.source_db.exists():
            raise FileNotFoundError(f"source_db not found: {self.source_db}")
        if not self.history_db.exists():
            raise FileNotFoundError(f"history_db not found: {self.history_db}")

        self._centroids: Optional[Dict[str, np.ndarray]] = None
        self._content:   Optional[Dict[str, EntryContent]] = None
        self._existing:  Optional[set[tuple[str, str]]] = None
        self._examples:  Optional[Dict[str, List[FewShotExample]]] = None

    # ── Private: data loading ──────────────────────────────────────────────────

    def _load_centroids(self) -> Dict[str, np.ndarray]:
        conn = sqlite3.connect(self.history_db)
        snap = conn.execute(
            "SELECT snapshot_id FROM wiki_snapshots ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not snap:
            conn.close()
            raise RuntimeError("No snapshots found — run sync.py first.")
        snap_id = snap[0]

        rows = conn.execute(
            "SELECT entry_id, embedding FROM chunk_embedding_history WHERE snapshot_id=?",
            (snap_id,),
        ).fetchall()
        conn.close()

        bucket: Dict[str, list] = defaultdict(list)
        for entry_id, blob in rows:
            bucket[entry_id].append(np.frombuffer(blob, dtype=np.float32).copy())

        centroids: Dict[str, np.ndarray] = {}
        for eid, embs in bucket.items():
            c = np.mean(embs, axis=0)
            n = np.linalg.norm(c)
            centroids[eid] = c / n if n > 0 else c
        return centroids

    def _load_content(self) -> Dict[str, EntryContent]:
        """Load entry metadata + WIC + Mathematical Form sections."""
        conn = sqlite3.connect(f"file:{self.source_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        entries = conn.execute(
            "SELECT id, title, domain, entry_type FROM entries"
        ).fetchall()

        sections: Dict[str, Dict[str, str]] = defaultdict(dict)
        for row in conn.execute(
            "SELECT entry_id, section_name, content FROM sections "
            "WHERE section_name IN ('What It Claims', 'Mathematical Form')"
        ).fetchall():
            sections[row["entry_id"]][row["section_name"]] = row["content"] or ""

        conn.close()

        content: Dict[str, EntryContent] = {}
        for e in entries:
            eid = e["id"]
            content[eid] = EntryContent(
                entry_id=eid,
                title=e["title"] or eid,
                domain=e["domain"] or "",
                entry_type=e["entry_type"] or "unknown",
                wic=sections[eid].get("What It Claims", ""),
                math_form=sections[eid].get("Mathematical Form", ""),
            )
        return content

    def _load_existing_links(self) -> set[tuple[str, str]]:
        conn = sqlite3.connect(f"file:{self.source_db}?mode=ro", uri=True)
        rows = conn.execute("SELECT source_id, target_id FROM links").fetchall()
        conn.close()
        pairs: set[tuple[str, str]] = set()
        for src, tgt in rows:
            pairs.add((src, tgt))
            pairs.add((tgt, src))
        return pairs

    def _load_few_shot_examples(self) -> Dict[str, List[FewShotExample]]:
        """
        Select up to N_EXAMPLES_PER_TYPE few-shot examples per link type.
        Prefers reference_law ↔ reference_law, cross-domain pairs.
        """
        conn = sqlite3.connect(f"file:{self.source_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT l.link_type, l.source_id, l.target_id, l.description,
                   e_s.entry_type as src_type, e_t.entry_type as tgt_type,
                   e_s.domain as src_domain, e_t.domain as tgt_domain,
                   e_s.title as src_title, e_t.title as tgt_title,
                   wic_s.content as src_wic, wic_t.content as tgt_wic,
                   mf_s.content  as src_mf,  mf_t.content  as tgt_mf
            FROM links l
            JOIN entries e_s ON e_s.id = l.source_id
            JOIN entries e_t ON e_t.id = l.target_id
            LEFT JOIN sections wic_s ON wic_s.entry_id = l.source_id
                                     AND wic_s.section_name = 'What It Claims'
            LEFT JOIN sections wic_t ON wic_t.entry_id = l.target_id
                                     AND wic_t.section_name = 'What It Claims'
            LEFT JOIN sections mf_s  ON mf_s.entry_id  = l.source_id
                                     AND mf_s.section_name  = 'Mathematical Form'
            LEFT JOIN sections mf_t  ON mf_t.entry_id  = l.target_id
                                     AND mf_t.section_name  = 'Mathematical Form'
            WHERE wic_s.content IS NOT NULL AND wic_t.content IS NOT NULL
              AND l.description IS NOT NULL AND LENGTH(l.description) > 30
            ORDER BY l.link_type,
                     -- Prefer RL↔RL, then cross-domain, then shorter descriptions
                     (e_s.entry_type = 'reference_law' AND e_t.entry_type = 'reference_law') DESC,
                     (e_s.domain != e_t.domain) DESC,
                     LENGTH(l.description) ASC
        """).fetchall()
        conn.close()

        by_type: Dict[str, List[FewShotExample]] = defaultdict(list)
        for r in rows:
            if len(by_type[r["link_type"]]) >= self.N_EXAMPLES_PER_TYPE:
                continue
            by_type[r["link_type"]].append(FewShotExample(
                source_id=r["source_id"],
                source_title=r["src_title"],
                source_domain=r["src_domain"] or "",
                source_wic=(r["src_wic"] or "")[:300],
                source_mf=r["src_mf"] or "",
                target_id=r["target_id"],
                target_title=r["tgt_title"],
                target_domain=r["tgt_domain"] or "",
                target_wic=(r["tgt_wic"] or "")[:300],
                target_mf=r["tgt_mf"] or "",
                link_type=r["link_type"],
                description=r["description"],
            ))
        return dict(by_type)

    def _ensure_loaded(self) -> None:
        if self._centroids is None:
            self._centroids = self._load_centroids()
        if self._content is None:
            self._content = self._load_content()
        if self._existing is None:
            self._existing = self._load_existing_links()
        if self._examples is None:
            self._examples = self._load_few_shot_examples()

    # ── Public: candidate generation ──────────────────────────────────────────

    def get_candidates(
        self,
        sim_threshold:  float = 0.78,
        max_pairs:      int   = 100,
        entry_types:    Optional[List[str]] = None,
        exclude_linked: bool  = True,
    ) -> List[CandidatePair]:
        """
        Return all unlinked entry pairs with cosine similarity ≥ sim_threshold,
        sorted descending by similarity.

        Parameters
        ----------
        sim_threshold  : cosine similarity floor (default 0.78)
        max_pairs      : cap on returned pairs (default 100)
        entry_types    : restrict to these entry_type values (None = all)
        exclude_linked : skip pairs that already have an explicit link
        """
        self._ensure_loaded()

        # Filter entries by type if requested
        valid_ids = [
            eid for eid, meta in self._content.items()
            if eid in self._centroids
            and (entry_types is None or meta.entry_type in entry_types)
        ]
        valid_ids.sort()
        N = len(valid_ids)

        matrix = np.stack([self._centroids[eid] for eid in valid_ids])
        sim_mat = matrix @ matrix.T
        np.fill_diagonal(sim_mat, 0.0)

        pairs: List[CandidatePair] = []
        for i in range(N):
            for j in range(i + 1, N):
                s = float(sim_mat[i, j])
                if s < sim_threshold:
                    continue
                a, b = valid_ids[i], valid_ids[j]
                if exclude_linked and (a, b) in self._existing:
                    continue
                pairs.append(CandidatePair(
                    entry_a=self._content[a],
                    entry_b=self._content[b],
                    similarity=round(s, 4),
                ))

        pairs.sort(key=lambda p: -p.similarity)
        return pairs[:max_pairs]

    # ── Public: prompt formatting ──────────────────────────────────────────────

    def _format_pair_block(self, pair: CandidatePair) -> str:
        return _PAIR_TEMPLATE.format(
            a_id=pair.entry_a.entry_id,
            a_domain=pair.entry_a.domain,
            a_title=pair.entry_a.title,
            a_wic=pair.entry_a.wic[:300],
            a_mf=pair.entry_a.math_form,
            b_id=pair.entry_b.entry_id,
            b_domain=pair.entry_b.domain,
            b_title=pair.entry_b.title,
            b_wic=pair.entry_b.wic[:300],
            b_mf=pair.entry_b.math_form,
            sim=pair.similarity,
        )

    def _format_examples_block(self) -> str:
        """Format all few-shot examples as a reference block."""
        self._ensure_loaded()
        lines = ["## CLASSIFICATION EXAMPLES (from existing knowledge graph)\n"]
        for lt, examples in sorted(self._examples.items()):
            for ex in examples:
                pair_block = _PAIR_TEMPLATE.format(
                    a_id=ex.source_id, a_domain=ex.source_domain,
                    a_title=ex.source_title, a_wic=ex.source_wic[:200],
                    a_mf=ex.source_mf,
                    b_id=ex.target_id,   b_domain=ex.target_domain,
                    b_title=ex.target_title, b_wic=ex.target_wic[:200],
                    b_mf=ex.target_mf,
                    sim=0.0,  # not shown in examples
                ).replace("Cosine similarity: 0.0000\n", "")
                lines.append(_EXAMPLE_TEMPLATE.format(
                    link_type=lt,
                    pair_block=pair_block.strip(),
                    description=ex.description[:120],
                ))
        return "\n".join(lines)

    def format_triage_prompt(
        self,
        pairs: List[CandidatePair],
        include_examples: bool = True,
    ) -> str:
        """
        Format a single prompt for interactive classification by Claude.
        Returns a string you can paste into a Claude conversation or MCP tool.

        The expected response format is a JSON array:
        [
          {"pair": 1, "has_link": true, "link_type": "...",
           "confidence": 0.9, "description": "...", "reasoning": "..."},
          ...
        ]
        """
        self._ensure_loaded()
        sections = [
            _SYSTEM_PROMPT.format(link_type_block=_LINK_TYPE_BLOCK),
            "",
        ]
        if include_examples:
            sections.append(self._format_examples_block())
            sections.append("")

        sections.append(
            f"## PAIRS TO CLASSIFY ({len(pairs)} total)\n\n"
            "For each pair below, output one JSON object. "
            "Collect all objects into a JSON array as your final response.\n"
        )
        for n, pair in enumerate(pairs, 1):
            sections.append(_BATCH_PAIR_TEMPLATE.format(
                n=n,
                pair_block=self._format_pair_block(pair).strip(),
            ))

        sections.append(
            "\n## RESPONSE FORMAT\n"
            "Return a JSON array of objects, one per pair:\n"
            '[\n'
            '  {\n'
            '    "pair": 1,\n'
            '    "has_link": true,\n'
            '    "link_type": "analogous to",\n'
            '    "confidence": 0.92,\n'
            '    "description": "60-80 word description for the DB...",\n'
            '    "reasoning": "brief justification"\n'
            '  },\n'
            '  ...\n'
            ']'
        )
        return "\n".join(sections)

    def format_batch_prompt(self, pairs: List[CandidatePair]) -> str:
        """Alias for format_triage_prompt — same output, used in batch mode."""
        return self.format_triage_prompt(pairs)

    # ── Public: response parsing ───────────────────────────────────────────────

    def parse_response(
        self,
        response_text: str,
        pairs:         List[CandidatePair],
    ) -> List[ClassificationResult]:
        """
        Parse a JSON array response from the LLM into ClassificationResult objects.
        Pairs are matched by the 1-indexed "pair" field.

        Robustly handles:
        - JSON wrapped in markdown code fences
        - Missing "pair" key (falls back to order)
        - Unknown link types (sets has_link=False)
        """
        # Strip markdown fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        try:
            raw_list = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {e}\n---\n{text[:500]}")

        if not isinstance(raw_list, list):
            raw_list = [raw_list]

        results: List[ClassificationResult] = []
        pair_index = {i + 1: pair for i, pair in enumerate(pairs)}

        for item in raw_list:
            n   = item.get("pair", len(results) + 1)
            pair = pair_index.get(n)
            if pair is None:
                continue

            has_link  = bool(item.get("has_link", False))
            link_type = item.get("link_type")
            if has_link and link_type not in VALID_LINK_TYPES:
                has_link  = False
                link_type = None

            results.append(ClassificationResult(
                source_id=pair.entry_a.entry_id,
                source_label=pair.entry_a.title,
                target_id=pair.entry_b.entry_id,
                target_label=pair.entry_b.title,
                has_link=has_link,
                link_type=link_type,
                confidence=float(item.get("confidence", 0.0)),
                description=item.get("description"),
                reasoning=item.get("reasoning"),
                similarity=pair.similarity,
            ))
        return results

    # ── Public: batch classification (requires anthropic SDK) ─────────────────

    def batch_classify(
        self,
        pairs:        List[CandidatePair],
        model:        str   = "claude-haiku-4-5-20251001",
        api_key:      Optional[str] = None,
        chunk_size:   int   = 20,
    ) -> List[ClassificationResult]:
        """
        Classify pairs using the Anthropic API. Processes in chunks to stay
        within context limits.

        Requires: pip install anthropic
        Requires: ANTHROPIC_API_KEY environment variable (or api_key parameter)

        Parameters
        ----------
        pairs      : from get_candidates()
        model      : Anthropic model ID (default: haiku for cost efficiency)
        api_key    : override ANTHROPIC_API_KEY env var
        chunk_size : pairs per API call (default 20)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            )

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY env var or pass api_key="
            )

        client = anthropic.Anthropic(api_key=key)
        all_results: List[ClassificationResult] = []

        for start in range(0, len(pairs), chunk_size):
            chunk = pairs[start: start + chunk_size]
            prompt = self.format_triage_prompt(chunk)

            print(f"  Classifying pairs {start+1}–{start+len(chunk)} / {len(pairs)}...")
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            chunk_results = self.parse_response(text, chunk)
            all_results.extend(chunk_results)

        return all_results

    # ── Public: DB insertion ───────────────────────────────────────────────────

    def insert_results(
        self,
        results:        List[ClassificationResult],
        min_confidence: float = 0.80,
        dry_run:        bool  = False,
    ) -> Dict[str, int]:
        """
        Insert classified links into ds_wiki.db.
        Skips: has_link=False, confidence < min_confidence, existing links.

        Returns summary dict: {"inserted": n, "skipped_confidence": n,
                               "skipped_no_link": n, "skipped_exists": n}
        """
        self._ensure_loaded()

        counts = {"inserted": 0, "skipped_confidence": 0,
                  "skipped_no_link": 0, "skipped_exists": 0}

        to_insert = []
        for r in results:
            if not r.has_link or r.link_type is None:
                counts["skipped_no_link"] += 1
                continue
            if r.confidence < min_confidence:
                counts["skipped_confidence"] += 1
                continue
            if (r.source_id, r.target_id) in self._existing:
                counts["skipped_exists"] += 1
                continue

            tier = score_to_tier(r.similarity)
            # Clamp to tier 2 minimum — LLM classification doesn't elevate tier
            if tier not in ("1", "1.5", "2"):
                tier = "2"

            to_insert.append((
                r.link_type,
                r.source_id, r.source_label,
                r.target_id, r.target_label,
                r.description,
                tier,
            ))

        if not dry_run and to_insert:
            conn = sqlite3.connect(self.source_db)
            for row in to_insert:
                conn.execute("""
                    INSERT OR IGNORE INTO links
                      (link_type, source_id, source_label, target_id, target_label,
                       description, link_order, confidence_tier)
                    VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """, row)
                print(f"  [{row[6]}] {row[1]} --{row[0]}--> {row[3]}")
                counts["inserted"] += 1
            conn.commit()
            conn.close()
            # Invalidate cache so future calls reflect new links
            self._existing = None
        elif dry_run:
            for row in to_insert:
                print(f"  DRY [{row[6]}] {row[1]} --{row[0]}--> {row[3]}")
                counts["inserted"] += 1

        return counts

    # ── Convenience: full report for human review queue ───────────────────────

    def pending_review_report(
        self,
        results:           List[ClassificationResult],
        confidence_cutoff: float = 0.80,
    ) -> str:
        """
        Format a human-readable review queue for results that need manual review
        (has_link=True but confidence < confidence_cutoff).
        """
        flagged = [
            r for r in results
            if r.has_link and r.confidence < confidence_cutoff
        ]
        flagged.sort(key=lambda r: -r.confidence)

        if not flagged:
            return "✓ No pairs require manual review at this threshold."

        lines = [
            f"# Human Review Queue  ({len(flagged)} pairs below confidence {confidence_cutoff})\n",
            "Format: PAIR | sim | conf | link_type | description\n",
            "---",
        ]
        for r in flagged:
            lines += [
                f"\n## {r.source_id} → {r.target_id}",
                f"- sim={r.similarity:.4f}  conf={r.confidence:.2f}  type=**{r.link_type}**",
                f"- {r.source_label} → {r.target_label}",
                f"- Description: {r.description}",
                f"- Reasoning: {r.reasoning}",
            ]
        return "\n".join(lines)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DS Wiki link classifier — generate candidates and/or classify them"
    )
    parser.add_argument("--sim-threshold", type=float, default=0.78,
                        help="Cosine similarity floor (default 0.78)")
    parser.add_argument("--max-pairs",     type=int,   default=60,
                        help="Max candidate pairs to return (default 60)")
    parser.add_argument("--entry-types",   nargs="*",
                        help="Restrict to these entry types (default: all)")
    parser.add_argument("--output",
                        choices=["prompt", "candidates_json", "batch", "review"],
                        default="prompt",
                        help="Output mode:\n"
                             "  prompt          — print triage prompt for interactive use\n"
                             "  candidates_json — print candidates as JSON\n"
                             "  batch           — run Anthropic API batch classification\n"
                             "  review          — run batch then print review queue")
    parser.add_argument("--auto-insert",   action="store_true",
                        help="Auto-insert high-confidence results (batch mode only)")
    parser.add_argument("--min-confidence",type=float, default=0.80,
                        help="Min confidence for insertion (default 0.80)")
    parser.add_argument("--model",
                        default="claude-haiku-4-5-20251001",
                        help="Anthropic model for batch mode")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print insertions without writing (batch + auto-insert)")
    args = parser.parse_args()

    lc = LinkClassifier()
    candidates = lc.get_candidates(
        sim_threshold=args.sim_threshold,
        max_pairs=args.max_pairs,
        entry_types=args.entry_types or None,
    )
    print(f"Found {len(candidates)} candidate pairs at sim ≥ {args.sim_threshold}\n",
          file=sys.stderr)

    if args.output == "candidates_json":
        out = [
            {
                "pair":       i + 1,
                "source_id":  c.entry_a.entry_id,
                "source":     c.entry_a.title,
                "target_id":  c.entry_b.entry_id,
                "target":     c.entry_b.title,
                "similarity": c.similarity,
            }
            for i, c in enumerate(candidates)
        ]
        print(json.dumps(out, indent=2))

    elif args.output == "prompt":
        print(lc.format_triage_prompt(candidates))

    elif args.output in ("batch", "review"):
        results = lc.batch_classify(candidates, model=args.model)
        if args.auto_insert or args.output == "batch":
            counts = lc.insert_results(
                results,
                min_confidence=args.min_confidence,
                dry_run=args.dry_run,
            )
            print(f"\nInsertion summary: {counts}", file=sys.stderr)
        if args.output == "review":
            print(lc.pending_review_report(results, args.min_confidence))
