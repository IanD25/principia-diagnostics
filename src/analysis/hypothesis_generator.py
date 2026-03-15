"""
hypothesis_generator.py — Find surprising high-similarity entity pairs and
generate natural-language research prompts for each.

SCOPE: Currently DS Wiki–specific (reads from wiki_history.db embedding snapshots).
       Potential generalisation: run on any RRP's cross_universe_bridges to surface
       surprising high-similarity pairs between RRP entries and DS Wiki anchors,
       adding hypothesis commentary to Tier-2 reports.
       Track in: Phase 3 / report enrichment milestone.

Algorithm
---------
1. Load embeddings from the latest snapshot in wiki_history.db.
2. Group chunks by entry_id → compute per-entry centroid (L2-normalised mean).
3. Load entity metadata (title, type, domain) from ds_wiki.db.
4. Compute the N×N pairwise cosine similarity matrix across entry centroids.
5. For each (entity_type_A, entity_type_B) combination compute a baseline
   mean similarity (falling back to the global mean for sparse pairs).
6. Find pairs where:
     similarity  ≥ sim_threshold  (default 0.80)
     surprise_factor = similarity / baseline  ≥ surprise_threshold (default 1.15)
7. For each surprising pair, generate 5-7 research prompts using typed templates.
8. Mark whether the pair already has an existing link in ds_wiki.db.

Entry points
------------
    gen = HypothesisGenerator(source_db, history_db)
    pairs = gen.find_surprising_pairs()          # List[SurprisingPair]
    report = gen.generate_markdown_report(pairs)  # str

Both source_db and history_db accept pathlib.Path or str.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path bootstrap so this module can be run standalone ────────────────────────
# Insert src/ (parent of this file's parent) onto sys.path if not already there.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import SOURCE_DB, HISTORY_DB  # noqa: E402  (after path bootstrap)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class EntityInfo:
    entity_id:   str
    title:       str
    entity_type: str
    domain:      str


@dataclass
class SurprisingPair:
    entity_a:          EntityInfo
    entity_b:          EntityInfo
    similarity:        float
    baseline:          float          # mean sim for this (type_a, type_b) pair
    surprise_factor:   float          # similarity / baseline
    has_existing_link: bool
    research_prompts:  List[str] = field(default_factory=list)


# ── Research-prompt templates ──────────────────────────────────────────────────
# Keys are tuples of (entity_type_a, entity_type_b) — always alphabetically sorted.
# The fallback BASE_TEMPLATES are used when no typed pair matches.

BASE_TEMPLATES: List[str] = [
    "Could {a} and {b} share a common underlying mathematical structure?",
    "Is {b} a generalisation of {a} in a higher-dimensional or extended framework?",
    "What would a unified principle encompassing both {a} and {b} look like?",
    "Does the mathematical form of {a} predict behaviour described by {b}?",
    "Could extremal (variational) principles governing {a} also constrain {b}?",
    "Under what boundary conditions do {a} and {b} become formally equivalent?",
    "What experimental signature would definitively distinguish {a} from {b}?",
]

_T = Dict[Tuple[str, str], List[str]]

TYPED_TEMPLATES: _T = {
    ("reference_law", "reference_law"): [
        "Is there a unified law from which both {a} and {b} emerge as special cases?",
        "Do {a} and {b} arise from the same underlying symmetry principle?",
        "What dimensional regime separates the regime of {a} from that of {b}?",
        "Could {a} be derived as a limiting case of {b} (or vice versa)?",
        "Is the mathematical form of {a} isomorphic to that of {b}?",
        "Do both {a} and {b} follow from the same variational action?",
        "Could a single scaling argument reconcile {a} and {b}?",
    ],
    ("law", "reference_law"): [
        "Is {a} a dimensionally-modified form of {b}?",
        "Does {a} represent the DS-framework generalisation of {b}?",
        "Can {b} be recovered from {a} in the D_eff = 4 limit?",
        "What physical analogy connects {b} to {a} under dimensional scaling?",
        "Does {b} predict the dimensional behaviour described by {a}?",
        "Is {a} novel, or is it {b} restated in dimensional language?",
        "What experimental test would distinguish the D_eff ≠ 4 prediction of {a} from {b}?",
    ],
    ("law", "law"): [
        "Are {a} and {b} manifestations of a single DS principle?",
        "Could the Ω_D operators in {a} and {b} be unified into one?",
        "Does {a} reduce to {b} in a specific limit of D_eff?",
        "Is there a DS conservation law linking {a} and {b}?",
        "What symmetry breaking separates the regime where {a} dominates from {b}?",
    ],
    ("constraint", "reference_law"): [
        "Does {b} impose the fundamental bounds formalised by {a}?",
        "Is {a} derived from {b} under dimensional restriction?",
        "Could {a} be violated if {b} breaks in non-integer dimensions?",
        "What scenario satisfies {b} while approaching the limit set by {a}?",
    ],
    ("constraint", "law"): [
        "Is {a} necessary for {b} to hold across dimensional regimes?",
        "Does {b} generate the constraint expressed in {a}?",
        "Under what D_eff value does {a} become the binding constraint for {b}?",
        "Is {a} a consequence of {b}, or does it arise from independent physics?",
    ],
    ("method", "reference_law"): [
        "Is {a} the computational implementation of {b}?",
        "Does {b} provide the theoretical basis for {a}?",
        "Could {a} be extended by applying {b} in non-integer dimensions?",
        "What assumptions in {a} break down when {b} is violated?",
    ],
    ("law", "method"): [
        "Is {a} the computational realisation of {b}?",
        "Does {a} correctly capture all implications of {b}?",
        "What limitations of {a} are revealed by the formal structure of {b}?",
        "Could {a} be made more efficient by exploiting the symmetries in {b}?",
    ],
    ("method", "method"): [
        "Could {a} and {b} be unified into a single computational framework?",
        "Under what conditions does {a} outperform {b} (and vice versa)?",
        "Do {a} and {b} share the same convergence assumptions?",
        "Is one of {a} or {b} a special case of the other?",
    ],
    ("open question", "reference_law"): [
        "Does {b} contain the key to resolving {a}?",
        "Could dimensional scaling of {b} directly answer the question posed by {a}?",
        "Is {a} open precisely because {b} breaks in non-D = 4 regimes?",
        "What experiment suggested by {b} would resolve {a}?",
    ],
    ("law", "open question"): [
        "Is {b} blocking the full formulation of {a}?",
        "If {b} were resolved, what new predictions would {a} make?",
        "Does {a} suggest a specific direction for resolving {b}?",
        "Is {b} an open question because the dimensional behaviour of {a} is unknown?",
    ],
    ("axiom", "reference_law"): [
        "Is {a} a formal axiomatisation of {b}?",
        "Does {b} follow as a theorem from {a}?",
        "Is {a} more fundamental than {b}, or are they logically equivalent?",
        "What is lost when {b} is stated as a law rather than derived from {a}?",
    ],
    ("axiom", "law"): [
        "Is {b} provable from {a} within the DS framework?",
        "Does {a} constrain the allowed forms of {b}?",
        "Is {a} the dimensional axiom from which {b} can be derived from first principles?",
    ],
    ("instantiation", "reference_law"): [
        "Is {a} a concrete realisation of {b} in non-standard dimensions?",
        "Does the instantiation in {a} recover {b} in the D_eff = 4 limit?",
        "What predictions does {b} make in the regime described by {a}?",
        "Is {a} the most natural embedding of {b} in the DS framework?",
    ],
    ("instantiation", "law"): [
        "Is {a} an instance of the general principle expressed by {b}?",
        "Does {b} predict {a}, or is {a} an independent result?",
        "Under what conditions does {a} cease to be a valid instantiation of {b}?",
    ],
    ("theorem", "reference_law"): [
        "Does {a} prove or fundamentally constrain {b}?",
        "Is {b} a consequence of the mathematical result in {a}?",
        "Could {a} be derived from dimensional analysis applied to {b}?",
        "What would it mean physically if {b} violated {a}?",
    ],
    ("theorem", "law"): [
        "Does {a} establish the formal validity of {b}?",
        "Is {b} provably consistent with {a}?",
        "What generalisations of {a} might yield a stronger form of {b}?",
    ],
    ("mechanism", "reference_law"): [
        "Is {a} the physical mechanism underlying {b}?",
        "Does {b} emerge from the operation of {a} at macroscopic scale?",
        "What dimensional dependence does the mechanism in {a} impart to {b}?",
    ],
    ("mechanism", "law"): [
        "Is {a} the microscopic origin of {b}?",
        "Does {b} correctly describe the large-scale limit of {a}?",
        "Could {a} produce deviations from {b} in non-integer dimensional regimes?",
    ],
    ("parameter", "reference_law"): [
        "Is {a} the critical parameter governing the form of {b}?",
        "How does dimensional scaling of {a} alter {b}?",
        "Does {b} define the physical range within which {a} is meaningful?",
    ],
    ("parameter", "law"): [
        "Is {a} the DS-specific parameter that modifies the standard form in {b}?",
        "Does {b} predict how {a} should scale with D_eff?",
        "Is {a} well-defined across all regimes implied by {b}?",
    ],
}


def _get_typed_templates(type_a: str, type_b: str) -> List[str]:
    """Return templates for (type_a, type_b), trying both orderings, else BASE."""
    key1 = (type_a, type_b)
    key2 = (type_b, type_a)
    if key1 in TYPED_TEMPLATES:
        return TYPED_TEMPLATES[key1]
    if key2 in TYPED_TEMPLATES:
        # Swap {a} / {b} labels in the reversed templates
        return [t.replace("{a}", "__X__").replace("{b}", "{a}").replace("__X__", "{b}")
                for t in TYPED_TEMPLATES[key2]]
    return BASE_TEMPLATES


# ── Core class ─────────────────────────────────────────────────────────────────

class HypothesisGenerator:
    """
    Finds surprisingly high-similarity entity pairs in the DS knowledge base
    and generates natural-language research prompts for each.

    Parameters
    ----------
    source_db  : path to ds_wiki.db     (read-only)
    history_db : path to wiki_history.db (read-only)
    """

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

        # Lazily computed; populated on first call to find_surprising_pairs()
        self._entity_meta: Optional[Dict[str, EntityInfo]] = None
        self._existing_links: Optional[set[Tuple[str, str]]] = None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_entity_meta(self) -> Dict[str, EntityInfo]:
        """Load entity id/title/type/domain from ds_wiki.db."""
        conn = sqlite3.connect(f"file:{self.source_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, title, entry_type, domain FROM entries"
        ).fetchall()
        conn.close()
        return {
            r["id"]: EntityInfo(
                entity_id=r["id"],
                title=r["title"] or r["id"],
                entity_type=r["entry_type"] or "unknown",
                domain=r["domain"] or "",
            )
            for r in rows
        }

    def _load_existing_links(self) -> set[Tuple[str, str]]:
        """Return set of (source_id, target_id) pairs that already have links."""
        conn = sqlite3.connect(f"file:{self.source_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT source_id, target_id FROM links").fetchall()
        conn.close()
        pairs: set[Tuple[str, str]] = set()
        for r in rows:
            a, b = r["source_id"], r["target_id"]
            pairs.add((a, b))
            pairs.add((b, a))   # treat as undirected for "has_existing_link"
        return pairs

    def _latest_snapshot_id(self, conn: sqlite3.Connection) -> Optional[str]:
        row = conn.execute(
            "SELECT snapshot_id FROM wiki_snapshots ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def _load_entry_centroids(
        self,
        entry_ids: set[str],
    ) -> Dict[str, np.ndarray]:
        """
        Load chunk embeddings from wiki_history.db (latest snapshot).
        For each entry_id, average all its chunk embeddings → L2-normalised centroid.
        Only entities in `entry_ids` are considered (skips conj_*, gate_*, bridge).
        """
        conn = sqlite3.connect(self.history_db)
        snap_id = self._latest_snapshot_id(conn)
        if not snap_id:
            conn.close()
            raise RuntimeError("No snapshots found in wiki_history.db. Run sync first.")

        rows = conn.execute(
            "SELECT entry_id, embedding FROM chunk_embedding_history WHERE snapshot_id = ?",
            (snap_id,),
        ).fetchall()
        conn.close()

        # Accumulate embeddings per entry
        bucket: Dict[str, List[np.ndarray]] = {}
        for entry_id, blob in rows:
            if entry_id not in entry_ids:
                continue
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            bucket.setdefault(entry_id, []).append(emb)

        centroids: Dict[str, np.ndarray] = {}
        for eid, embs in bucket.items():
            mat = np.stack(embs)                     # (k, 384)
            c   = mat.mean(axis=0)                   # (384,)
            norm = np.linalg.norm(c)
            if norm > 0:
                c /= norm
            centroids[eid] = c
        return centroids

    def _compute_pairwise(
        self,
        entry_ids: List[str],
        centroids:  Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute N×N cosine similarity matrix.
        Embeddings are already L2-normalised so cosine sim = dot product.
        """
        matrix = np.stack([centroids[eid] for eid in entry_ids])  # (N, 384)
        sim = matrix @ matrix.T                                    # (N, N)
        np.fill_diagonal(sim, 0.0)                                 # exclude self
        return sim

    def _compute_baselines(
        self,
        entry_ids:    List[str],
        entity_meta:  Dict[str, EntityInfo],
        sim_matrix:   np.ndarray,
        min_pairs:    int = 5,
    ) -> Tuple[Dict[Tuple[str, str], float], float]:
        """
        For each (entity_type_a, entity_type_b) combination (alphabetically sorted),
        compute the mean pairwise similarity as the expected baseline.

        Falls back to the global mean when a type-pair has fewer than `min_pairs`
        observations (sparse type combinations).

        Returns (baselines_dict, global_mean).
        """
        N = len(entry_ids)
        type_pair_sims: Dict[Tuple[str, str], List[float]] = {}

        for i in range(N):
            type_a = entity_meta[entry_ids[i]].entity_type
            for j in range(i + 1, N):
                type_b = entity_meta[entry_ids[j]].entity_type
                key = tuple(sorted([type_a, type_b]))
                type_pair_sims.setdefault(key, []).append(float(sim_matrix[i, j]))

        all_sims = [s for sims in type_pair_sims.values() for s in sims]
        global_mean = float(np.mean(all_sims)) if all_sims else 0.5

        baselines = {
            key: float(np.mean(sims)) if len(sims) >= min_pairs else global_mean
            for key, sims in type_pair_sims.items()
        }
        return baselines, global_mean

    def _generate_prompts(
        self,
        pair: SurprisingPair,
        n: int = 6,
    ) -> List[str]:
        """Generate up to `n` research prompts for a surprising pair."""
        templates = _get_typed_templates(
            pair.entity_a.entity_type,
            pair.entity_b.entity_type,
        )
        a_title = pair.entity_a.title
        b_title = pair.entity_b.title

        prompts = []
        for tmpl in templates[:n]:
            prompts.append(tmpl.format(a=a_title, b=b_title))
        # Pad with base templates if typed bank is short
        for tmpl in BASE_TEMPLATES:
            if len(prompts) >= n:
                break
            candidate = tmpl.format(a=a_title, b=b_title)
            if candidate not in prompts:
                prompts.append(candidate)
        return prompts[:n]

    # ── Public API ─────────────────────────────────────────────────────────────

    def find_surprising_pairs(
        self,
        sim_threshold:      float = 0.80,
        surprise_threshold: float = 1.15,
        max_pairs:          int   = 200,
        include_linked:     bool  = True,
    ) -> List[SurprisingPair]:
        """
        Find entity pairs that are significantly more similar than expected
        for their entity-type combination.

        Parameters
        ----------
        sim_threshold      : minimum cosine similarity to consider (default 0.80)
        surprise_threshold : minimum surprise_factor = sim/baseline (default 1.15)
        max_pairs          : cap on returned pairs (default 200)
        include_linked     : if False, skip pairs that already have an explicit link

        Returns
        -------
        List[SurprisingPair] sorted by surprise_factor descending.
        """
        if self._entity_meta is None:
            self._entity_meta = self._load_entity_meta()
        if self._existing_links is None:
            self._existing_links = self._load_existing_links()

        entry_ids_set = set(self._entity_meta.keys())
        centroids = self._load_entry_centroids(entry_ids_set)

        # Only keep entities that have embedding data
        valid_ids = [eid for eid in self._entity_meta if eid in centroids]
        valid_ids.sort()   # deterministic ordering

        sim_matrix = self._compute_pairwise(valid_ids, centroids)
        baselines, global_mean = self._compute_baselines(
            valid_ids, self._entity_meta, sim_matrix
        )

        N = len(valid_ids)
        pairs: List[SurprisingPair] = []

        for i in range(N):
            meta_a = self._entity_meta[valid_ids[i]]
            type_a = meta_a.entity_type
            for j in range(i + 1, N):
                meta_b   = self._entity_meta[valid_ids[j]]
                sim_val  = float(sim_matrix[i, j])
                if sim_val < sim_threshold:
                    continue

                type_b  = meta_b.entity_type
                key     = tuple(sorted([type_a, type_b]))
                baseline = baselines.get(key, global_mean)
                sf       = sim_val / baseline if baseline > 0 else 0.0

                if sf < surprise_threshold:
                    continue

                has_link = (
                    (valid_ids[i], valid_ids[j]) in self._existing_links
                    or (valid_ids[j], valid_ids[i]) in self._existing_links
                )
                if not include_linked and has_link:
                    continue

                sp = SurprisingPair(
                    entity_a=meta_a,
                    entity_b=meta_b,
                    similarity=round(sim_val, 4),
                    baseline=round(baseline, 4),
                    surprise_factor=round(sf, 4),
                    has_existing_link=has_link,
                )
                pairs.append(sp)

        # Sort by surprise_factor descending, then similarity descending
        pairs.sort(key=lambda p: (-p.surprise_factor, -p.similarity))
        pairs = pairs[:max_pairs]

        # Generate prompts for every retained pair
        for sp in pairs:
            sp.research_prompts = self._generate_prompts(sp)

        return pairs

    def generate_markdown_report(
        self,
        pairs:      Optional[List[SurprisingPair]] = None,
        sim_threshold:      float = 0.80,
        surprise_threshold: float = 1.15,
        max_pairs:          int   = 50,
    ) -> str:
        """
        Build a human-readable markdown report of the most surprising pairs.

        If `pairs` is None, calls find_surprising_pairs() first.
        """
        if pairs is None:
            pairs = self.find_surprising_pairs(
                sim_threshold=sim_threshold,
                surprise_threshold=surprise_threshold,
                max_pairs=max_pairs,
            )

        # Summary counts
        cross_type = sum(
            1 for p in pairs
            if p.entity_a.entity_type != p.entity_b.entity_type
        )
        unlinked   = sum(1 for p in pairs if not p.has_existing_link)
        linked     = len(pairs) - unlinked

        lines = [
            "# Hypothesis Generator Report",
            "",
            "## Summary",
            "",
            f"- **Total surprising pairs**: {len(pairs)}",
            f"  - Cross-type pairs: {cross_type}",
            f"  - Same-type pairs: {len(pairs) - cross_type}",
            f"- **No existing link**: {unlinked}",
            f"- **Already linked**: {linked}",
            f"- Similarity threshold: ≥ {sim_threshold}",
            f"- Surprise threshold: ≥ {surprise_threshold}× type baseline",
            "",
            "---",
            "",
            "## Top Surprising Pairs",
            "",
        ]

        for rank, p in enumerate(pairs, 1):
            link_tag = "🔗 linked" if p.has_existing_link else "💡 no link"
            lines += [
                f"### {rank}. {p.entity_a.entity_id} ↔ {p.entity_b.entity_id} "
                f"({link_tag})",
                "",
                f"| | Entity A | Entity B |",
                f"|---|---|---|",
                f"| **ID** | `{p.entity_a.entity_id}` | `{p.entity_b.entity_id}` |",
                f"| **Title** | {p.entity_a.title} | {p.entity_b.title} |",
                f"| **Type** | {p.entity_a.entity_type} | {p.entity_b.entity_type} |",
                f"| **Domain** | {p.entity_a.domain or '—'} | {p.entity_b.domain or '—'} |",
                "",
                f"- **Similarity**: {p.similarity:.4f}",
                f"- **Type-pair baseline**: {p.baseline:.4f}",
                f"- **Surprise factor**: {p.surprise_factor:.4f}×",
                "",
                "**Research prompts**:",
                "",
            ]
            for q in p.research_prompts:
                lines.append(f"- _{q}_")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def get_stats(
        self,
        sim_threshold:      float = 0.80,
        surprise_threshold: float = 1.15,
    ) -> dict:
        """
        Return a compact statistics dictionary (suitable for MCP tool output).
        Does not generate prompts to keep this fast.
        """
        pairs = self.find_surprising_pairs(
            sim_threshold=sim_threshold,
            surprise_threshold=surprise_threshold,
            max_pairs=500,
        )
        type_pair_counts: Dict[str, int] = {}
        for p in pairs:
            key = f"{p.entity_a.entity_type} ↔ {p.entity_b.entity_type}"
            type_pair_counts[key] = type_pair_counts.get(key, 0) + 1

        return {
            "total_surprising_pairs": len(pairs),
            "unlinked_pairs": sum(1 for p in pairs if not p.has_existing_link),
            "linked_pairs": sum(1 for p in pairs if p.has_existing_link),
            "cross_type_pairs": sum(
                1 for p in pairs
                if p.entity_a.entity_type != p.entity_b.entity_type
            ),
            "max_surprise_factor": round(max((p.surprise_factor for p in pairs), default=0.0), 4),
            "mean_similarity": round(
                float(np.mean([p.similarity for p in pairs])) if pairs else 0.0, 4
            ),
            "type_pair_distribution": dict(
                sorted(type_pair_counts.items(), key=lambda x: -x[1])
            ),
            "sim_threshold":      sim_threshold,
            "surprise_threshold": surprise_threshold,
        }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run Hypothesis Generator on DS wiki")
    parser.add_argument("--sim-threshold",      type=float, default=0.80)
    parser.add_argument("--surprise-threshold", type=float, default=1.15)
    parser.add_argument("--max-pairs",          type=int,   default=50)
    parser.add_argument("--output",             choices=["markdown", "json", "stats"],
                                                default="markdown")
    parser.add_argument("--no-linked",          action="store_true",
                        help="Exclude pairs that already have explicit links")
    args = parser.parse_args()

    gen = HypothesisGenerator()

    if args.output == "stats":
        stats = gen.get_stats(args.sim_threshold, args.surprise_threshold)
        print(json.dumps(stats, indent=2))
    else:
        pairs = gen.find_surprising_pairs(
            sim_threshold=args.sim_threshold,
            surprise_threshold=args.surprise_threshold,
            max_pairs=args.max_pairs,
            include_linked=not args.no_linked,
        )
        if args.output == "json":
            out = []
            for p in pairs:
                out.append({
                    "entity_a": p.entity_a.entity_id,
                    "entity_b": p.entity_b.entity_id,
                    "similarity": p.similarity,
                    "baseline": p.baseline,
                    "surprise_factor": p.surprise_factor,
                    "has_existing_link": p.has_existing_link,
                    "prompts": p.research_prompts,
                })
            print(json.dumps(out, indent=2))
        else:
            report = gen.generate_markdown_report(pairs)
            print(report)
