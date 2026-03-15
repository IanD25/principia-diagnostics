"""
result_validator.py — Validate a research claim against the knowledge base.

SCOPE: DS Wiki–specific (queries against the DS Wiki ChromaDB index).
       Phase 3A enhancement: resolve_claim() provides dynamic channel resolution
       for free-text claims with polarity hints (SCF methodology).

Algorithm (validate_claim — Phase 1)
---------
1. Embed the claim text using the BGE model (same model as sync.py).
2. Query ChromaDB for the top-k most similar chunks (by cosine similarity).
3. Deduplicate chunks → entries (take max sim per entry).
4. Retrieve entry metadata (title, type, domain) from ds_wiki.db.
5. For each pair of top-k entries, look up existing links in ds_wiki.db.
6. Classify evidence:
     supporting   : entry sim ≥ high_threshold AND no tensions-with link to
                    another high-sim entry (i.e. the KB is consistent here)
     contradicting: two high-sim entries have a "tensions with" link between them
                    → both are flagged, the link is reported as a contradiction
     related      : entry sim ≥ low_threshold but < high_threshold (KB is
                    relevant but doesn't directly speak to the claim)
7. Compute consistency_score = (S - 0.5·C) / max(1, S + C + R)
   where S = supporting count, C = contradicting count, R = related count.

Algorithm (resolve_claim — Phase 3A, SCF dynamic channel resolution)
---------
1. Accept a Claim object (from claim_extractor.py) or raw text.
2. Embed and query ChromaDB for top-K DS Wiki channels.
3. For each channel match, infer polarity from:
   a. Claim's polarity_hint (linguistic cues from claim_extractor)
   b. DS Wiki link graph structure (if matched entry has tensions_with links)
4. Return ClaimResolution with channel matches, polarity, confidence.
5. GATE: Claim must have human_approved=True before resolution is trusted.

Entry points
------------
    validator = ResultValidator(source_db, chroma_dir)
    result    = validator.validate_claim("Entropy increases with dimension")
    print(result.summary)
    print(result.as_markdown())

    # Phase 3A — dynamic channel resolution:
    from claim_extractor import Claim
    resolution = validator.resolve_claim(claim_or_text)
    print(resolution.as_markdown())

Both source_db and chroma_dir accept pathlib.Path or str.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import SOURCE_DB, CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL  # noqa: E402


# ── Link semantics ─────────────────────────────────────────────────────────────
# These link types mean the two entries are mutually consistent / supporting.
SUPPORTING_LINK_TYPES: frozenset[str] = frozenset({
    "derives from",
    "analogous to",
    "generalizes",
    "implements",
    "predicts for",
    "couples to",
    "constrains",
})

# These link types indicate structural tension (the closest we have to "contradicts").
CONTRADICTING_LINK_TYPES: frozenset[str] = frozenset({
    "tensions with",
})

# Default similarity thresholds
HIGH_SIM_THRESHOLD = 0.72   # entry is directly relevant to the claim
LOW_SIM_THRESHOLD  = 0.55   # entry is topically related but not direct evidence


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    entry_id:    str
    title:       str
    entry_type:  str
    domain:      str
    similarity:  float                        # cosine sim to claim embedding
    link_type:   Optional[str]  = None        # link type if paired via link
    linked_to:   Optional[str]  = None        # other entry_id in the link pair
    linked_title: Optional[str] = None        # other entry's title
    excerpt:     str            = ""          # leading content from WIC section


@dataclass
class ValidationResult:
    claim:                str
    consistency_score:    float                              # 0.0 – 1.0 (clamped)
    supporting_evidence:  List[EvidenceItem] = field(default_factory=list)
    contradictions:       List[EvidenceItem] = field(default_factory=list)
    related_entities:     List[EvidenceItem] = field(default_factory=list)
    notes:                List[str]          = field(default_factory=list)

    @property
    def summary(self) -> str:
        s = len(self.supporting_evidence)
        c = len(self.contradictions)
        r = len(self.related_entities)
        return (
            f"Consistency score: {self.consistency_score:.2f} | "
            f"Supporting: {s} | Contradictions: {c} | Related: {r}"
        )

    def as_markdown(self) -> str:
        lines = [
            f"## Claim Validation Report",
            f"",
            f"**Claim**: {self.claim}",
            f"",
            f"**Consistency score**: {self.consistency_score:.2f}  ",
            f"**Supporting evidence**: {len(self.supporting_evidence)}  ",
            f"**Contradictions**: {len(self.contradictions)}  ",
            f"**Related (lower sim)**: {len(self.related_entities)}",
            f"",
        ]
        if self.notes:
            lines += ["**Notes**:"] + [f"- {n}" for n in self.notes] + [""]

        if self.supporting_evidence:
            lines.append("### Supporting Evidence")
            for e in self.supporting_evidence:
                lines.append(f"- **{e.entry_id}** — {e.title} *(sim={e.similarity:.3f})*")
                if e.excerpt:
                    lines.append(f"  > {e.excerpt[:120]}")
                if e.link_type and e.linked_title:
                    lines.append(f"  ↔ [{e.link_type}] {e.linked_to}: {e.linked_title}")
            lines.append("")

        if self.contradictions:
            lines.append("### Contradictions")
            for e in self.contradictions:
                lines.append(f"- **{e.entry_id}** — {e.title} *(sim={e.similarity:.3f})*")
                if e.link_type and e.linked_title:
                    lines.append(f"  ⚡ [{e.link_type}] {e.linked_to}: {e.linked_title}")
            lines.append("")

        if self.related_entities:
            lines.append("### Related Entities")
            for e in self.related_entities:
                lines.append(f"- **{e.entry_id}** — {e.title} *(sim={e.similarity:.3f})*")
            lines.append("")

        return "\n".join(lines)


# ── ResultValidator ────────────────────────────────────────────────────────────

# ── Phase 3A: Claim Resolution ────────────────────────────────────────────────

@dataclass
class ChannelMatch:
    """A DS Wiki entry matched to a claim (SCF channel resolution)."""
    entry_id: str
    title: str
    entry_type: str
    domain: str
    similarity: float
    polarity_hint: str = "neutral"     # positive/negative/neutral
    polarity_markers: List[str] = field(default_factory=list)
    excerpt: str = ""
    tier: str = ""                     # tier-1/tier-1.5/tier-2 based on similarity


@dataclass
class ClaimResolution:
    """Result of resolving a claim against DS Wiki channels (Phase 3A)."""
    claim_text: str
    source_section: str = ""
    claim_approved: bool = False       # GATE: must be True for trusted output
    channel_matches: List[ChannelMatch] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def top_channel(self) -> Optional[ChannelMatch]:
        return self.channel_matches[0] if self.channel_matches else None

    @property
    def summary(self) -> str:
        n = len(self.channel_matches)
        gate = "APPROVED" if self.claim_approved else "PENDING APPROVAL"
        top = (
            f" | top: {self.top_channel.entry_id} "
            f"(sim={self.top_channel.similarity:.3f})"
            if self.top_channel else ""
        )
        return f"[{gate}] {n} DS Wiki channels resolved{top}"

    def as_markdown(self) -> str:
        gate = "APPROVED" if self.claim_approved else "PENDING APPROVAL"
        lines = [
            "## Claim Channel Resolution",
            "",
            f"**Claim:** {self.claim_text}",
            f"**Section:** {self.source_section or '(unknown)'}",
            f"**Gate status:** {gate}",
            "",
        ]
        if not self.claim_approved:
            lines.append(
                "> **WARNING:** This claim has NOT been human-approved. "
                "Resolution results are provisional."
            )
            lines.append("")

        if self.notes:
            lines += ["**Notes:**"] + [f"- {n}" for n in self.notes] + [""]

        if self.channel_matches:
            lines.append("### DS Wiki Channel Matches")
            lines.append("")
            for i, ch in enumerate(self.channel_matches, 1):
                pol_icon = {"positive": "+", "negative": "!", "neutral": "~"}.get(
                    ch.polarity_hint, "~"
                )
                lines.append(
                    f"{i}. **{ch.entry_id}** — {ch.title} | "
                    f"sim={ch.similarity:.3f} | {ch.tier} | "
                    f"{pol_icon} {ch.polarity_hint}"
                )
                if ch.excerpt:
                    lines.append(f"   > {ch.excerpt}")
                if ch.polarity_markers:
                    lines.append(
                        f"   > *Polarity markers: {', '.join(ch.polarity_markers)}*"
                    )
            lines.append("")
        else:
            lines.append("*No DS Wiki channels matched above threshold.*")
            lines.append("")

        return "\n".join(lines)


# ── ResultValidator ────────────────────────────────────────────────────────────

class ResultValidator:
    """
    Validate research claims against the knowledge base.

    Parameters
    ----------
    source_db  : path to ds_wiki.db (read-only)
    chroma_dir : path to ChromaDB persistent directory
    collection : ChromaDB collection name (default from config)
    model_name : BGE model name (default from config)
    """

    def __init__(
        self,
        source_db:  Path | str = SOURCE_DB,
        chroma_dir: Path | str = CHROMA_DIR,
        collection: str        = CHROMA_COLLECTION,
        model_name: str        = EMBED_MODEL,
    ) -> None:
        self._source_db  = Path(source_db)
        self._chroma_dir = Path(chroma_dir)
        self._collection = collection
        self._model_name = model_name

        # Lazy-loaded — only imported/initialised when first needed.
        self._model      = None   # SentenceTransformer
        self._chroma_col = None   # chromadb Collection

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _get_collection(self):
        if self._chroma_col is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(self._chroma_dir))
            self._chroma_col = client.get_collection(self._collection)
        return self._chroma_col

    def _embed(self, text: str) -> np.ndarray:
        """Return L2-normalised embedding for text."""
        model = self._get_model()
        # BGE instruction prefix for retrieval queries
        query = f"Represent this sentence for searching relevant passages: {text}"
        vec = model.encode(query, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def _query_chroma(self, embedding: np.ndarray, top_k: int) -> List[Tuple[str, str, float]]:
        """
        Return list of (chunk_id, entry_id, similarity) sorted by descending similarity.
        Similarity = 1 - cosine_distance (ChromaDB returns distances by default).
        """
        col = self._get_collection()
        results = col.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, col.count()),
            include=["metadatas", "distances"],
        )
        out = []
        for chunk_id, meta, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sim = max(0.0, 1.0 - dist)   # convert cosine distance → similarity
            out.append((chunk_id, meta.get("entry_id", ""), sim))
        return out

    def _best_sim_per_entry(
        self, chunk_results: List[Tuple[str, str, float]]
    ) -> Dict[str, float]:
        """Deduplicate chunks → entries, keeping max sim per entry."""
        best: Dict[str, float] = {}
        for _, entry_id, sim in chunk_results:
            if entry_id and sim > best.get(entry_id, -1.0):
                best[entry_id] = sim
        return best

    def _fetch_metadata(self, entry_ids: List[str]) -> Dict[str, dict]:
        """Return {entry_id: {title, entry_type, domain}} for each requested id."""
        if not entry_ids:
            return {}
        conn = sqlite3.connect(self._source_db)
        ph   = ",".join("?" * len(entry_ids))
        rows = conn.execute(
            f"SELECT id, title, entry_type, domain FROM entries WHERE id IN ({ph})",
            entry_ids,
        ).fetchall()
        conn.close()
        return {r[0]: {"title": r[1], "entry_type": r[2], "domain": r[3] or ""} for r in rows}

    def _fetch_excerpt(self, entry_id: str) -> str:
        """Return first ~120 chars of 'What It Claims' (or first section)."""
        conn = sqlite3.connect(self._source_db)
        row  = conn.execute(
            """SELECT content FROM sections
               WHERE entry_id = ?
               ORDER BY CASE WHEN section_name LIKE '%Claims%' THEN 0
                             WHEN section_name LIKE '%Says%'   THEN 1
                             ELSE section_order END
               LIMIT 1""",
            (entry_id,),
        ).fetchone()
        conn.close()
        if row and row[0]:
            text = row[0].strip().replace("\n", " ")
            return text[:120] + ("…" if len(text) > 120 else "")
        return ""

    def _fetch_links_between(
        self, entry_ids: List[str]
    ) -> List[Tuple[str, str, str, str, str]]:
        """
        Return all links where BOTH source_id AND target_id are in entry_ids.
        Columns: (link_type, source_id, target_id, source_label, target_label)
        """
        if len(entry_ids) < 2:
            return []
        conn = sqlite3.connect(self._source_db)
        ph   = ",".join("?" * len(entry_ids))
        rows = conn.execute(
            f"""SELECT link_type, source_id, target_id, source_label, target_label
                FROM links
                WHERE source_id IN ({ph}) AND target_id IN ({ph})""",
            entry_ids + entry_ids,
        ).fetchall()
        conn.close()
        return rows

    # ── Public API ─────────────────────────────────────────────────────────────

    def validate_claim(
        self,
        claim:          str,
        top_k:          int   = 15,
        high_threshold: float = HIGH_SIM_THRESHOLD,
        low_threshold:  float = LOW_SIM_THRESHOLD,
    ) -> ValidationResult:
        """
        Validate a free-text research claim against the knowledge base.

        Parameters
        ----------
        claim          : The claim to validate (natural language).
        top_k          : Number of chunks to retrieve from ChromaDB.
        high_threshold : Cosine sim above which an entry is "directly relevant".
        low_threshold  : Cosine sim above which an entry is "related".

        Returns
        -------
        ValidationResult with consistency_score, supporting_evidence,
        contradictions, and related_entities.
        """
        notes: List[str] = []

        # 1. Embed claim
        embedding = self._embed(claim)

        # 2. Query ChromaDB
        chunk_results = self._query_chroma(embedding, top_k)
        if not chunk_results:
            return ValidationResult(
                claim=claim,
                consistency_score=0.0,
                notes=["ChromaDB returned no results — is the index populated?"],
            )

        # 3. Deduplicate → best sim per entry
        entry_sims = self._best_sim_per_entry(chunk_results)

        # 4. Split into high-sim and related buckets
        high_entries    = {eid: s for eid, s in entry_sims.items() if s >= high_threshold}
        related_entries = {
            eid: s for eid, s in entry_sims.items()
            if low_threshold <= s < high_threshold
        }

        if not high_entries and not related_entries:
            notes.append(
                f"No entries found above low_threshold={low_threshold:.2f}. "
                "Claim may be outside KB coverage."
            )
            return ValidationResult(claim=claim, consistency_score=0.0, notes=notes)

        # 5. Fetch metadata
        all_ids  = list(high_entries) + list(related_entries)
        metadata = self._fetch_metadata(all_ids)

        # 6. Find links between high-sim entries
        high_ids = list(high_entries)
        links    = self._fetch_links_between(high_ids)

        # Build a set of (src, tgt) pairs for each link type
        tension_pairs: set[frozenset[str]] = set()
        supporting_pairs: set[frozenset[str]] = set()
        for link_type, src, tgt, *_ in links:
            pair = frozenset({src, tgt})
            if link_type in CONTRADICTING_LINK_TYPES:
                tension_pairs.add(pair)
            elif link_type in SUPPORTING_LINK_TYPES:
                supporting_pairs.add(pair)

        # 7. Classify high-sim entries
        contradicted_ids: set[str] = set()
        for pair in tension_pairs:
            contradicted_ids.update(pair)

        # Build link index: entry_id → best link info (for reporting)
        entry_link_info: Dict[str, Tuple[str, str, str]] = {}  # {eid: (link_type, other_id, other_title)}
        for link_type, src, tgt, src_lbl, tgt_lbl in links:
            if link_type in CONTRADICTING_LINK_TYPES:
                if src in high_entries and src not in entry_link_info:
                    entry_link_info[src] = (link_type, tgt, tgt_lbl)
                if tgt in high_entries and tgt not in entry_link_info:
                    entry_link_info[tgt] = (link_type, src, src_lbl)
            elif link_type in SUPPORTING_LINK_TYPES:
                if src in high_entries and src not in entry_link_info:
                    entry_link_info[src] = (link_type, tgt, tgt_lbl)
                if tgt in high_entries and tgt not in entry_link_info:
                    entry_link_info[tgt] = (link_type, src, src_lbl)

        supporting_evidence: List[EvidenceItem] = []
        contradictions:      List[EvidenceItem] = []

        for eid, sim in sorted(high_entries.items(), key=lambda x: -x[1]):
            meta    = metadata.get(eid, {})
            excerpt = self._fetch_excerpt(eid)
            ltype, lother, lother_title = entry_link_info.get(eid, (None, None, None))
            item = EvidenceItem(
                entry_id     = eid,
                title        = meta.get("title", eid),
                entry_type   = meta.get("entry_type", ""),
                domain       = meta.get("domain", ""),
                similarity   = sim,
                link_type    = ltype,
                linked_to    = lother,
                linked_title = lother_title,
                excerpt      = excerpt,
            )
            if eid in contradicted_ids:
                contradictions.append(item)
            else:
                supporting_evidence.append(item)

        # 8. Related entities
        related_items: List[EvidenceItem] = []
        for eid, sim in sorted(related_entries.items(), key=lambda x: -x[1]):
            meta = metadata.get(eid, {})
            related_items.append(EvidenceItem(
                entry_id   = eid,
                title      = meta.get("title", eid),
                entry_type = meta.get("entry_type", ""),
                domain     = meta.get("domain", ""),
                similarity = sim,
            ))

        # 9. Consistency score
        S = len(supporting_evidence)
        C = len(contradictions)
        R = len(related_items)
        raw_score     = (S - 0.5 * C) / max(1, S + C + R)
        consistency   = max(0.0, min(1.0, raw_score))

        if S == 0 and C == 0:
            notes.append("No high-sim entries found — consider lowering high_threshold.")
        if C > 0:
            notes.append(
                f"{C} entry pair(s) have 'tensions with' links — inspect contradictions."
            )

        return ValidationResult(
            claim               = claim,
            consistency_score   = consistency,
            supporting_evidence = supporting_evidence,
            contradictions      = contradictions,
            related_entities    = related_items,
            notes               = notes,
        )

    # ── Phase 3A: Dynamic Channel Resolution ──────────────────────────────────

    def resolve_claim(
        self,
        claim,
        top_k: int = 10,
        min_similarity: float = 0.65,
    ) -> ClaimResolution:
        """
        Resolve a claim to DS Wiki channels with polarity hints (Phase 3A SCF).

        Parameters
        ----------
        claim : str or Claim object (from claim_extractor.py)
            If a Claim object, uses its text and polarity_hint.
            If a string, treats it as raw claim text with neutral polarity.
        top_k : int
            Number of DS Wiki channels to return.
        min_similarity : float
            Minimum cosine similarity for a channel match.

        Returns
        -------
        ClaimResolution with channel matches, polarity, gate status.
        """
        # Handle both Claim objects and raw strings
        claim_text = ""
        source_section = ""
        claim_approved = False
        polarity_hint = "neutral"
        polarity_markers: List[str] = []

        # Duck-type check for Claim object
        if hasattr(claim, "text"):
            claim_text = claim.text
            source_section = getattr(claim, "source_section", "")
            claim_approved = getattr(claim, "human_approved", False)
            hint = getattr(claim, "polarity_hint", None)
            if hint is not None:
                polarity_hint = hint.value if hasattr(hint, "value") else str(hint)
            polarity_markers = getattr(claim, "polarity_markers", [])
        else:
            claim_text = str(claim)
            # Detect polarity from raw text
            try:
                from analysis.claim_extractor import detect_polarity
                hint_enum, markers = detect_polarity(claim_text)
                polarity_hint = hint_enum.value
                polarity_markers = markers
            except ImportError:
                pass

        notes: List[str] = []

        if not claim_approved:
            notes.append(
                "GATE: Claim not yet human-approved. "
                "Results are provisional per Foundational Plan §3.1."
            )

        # 1. Embed claim
        embedding = self._embed(claim_text)

        # 2. Query ChromaDB
        chunk_results = self._query_chroma(embedding, top_k * 3)  # oversample for dedup
        if not chunk_results:
            return ClaimResolution(
                claim_text=claim_text,
                source_section=source_section,
                claim_approved=claim_approved,
                notes=["ChromaDB returned no results — is the index populated?"],
            )

        # 3. Deduplicate → best sim per entry
        entry_sims = self._best_sim_per_entry(chunk_results)

        # 4. Filter by min_similarity and take top_k
        filtered = {
            eid: sim for eid, sim in entry_sims.items()
            if sim >= min_similarity
        }
        sorted_entries = sorted(filtered.items(), key=lambda x: -x[1])[:top_k]

        if not sorted_entries:
            notes.append(
                f"No entries found above min_similarity={min_similarity:.2f}. "
                "Claim may be outside DS Wiki coverage."
            )
            return ClaimResolution(
                claim_text=claim_text,
                source_section=source_section,
                claim_approved=claim_approved,
                notes=notes,
            )

        # 5. Fetch metadata and excerpts
        entry_ids = [eid for eid, _ in sorted_entries]
        metadata = self._fetch_metadata(entry_ids)

        # 6. Build channel matches with polarity and tier
        channels: List[ChannelMatch] = []
        for eid, sim in sorted_entries:
            meta = metadata.get(eid, {})

            # Determine tier
            if sim >= 0.85:
                tier = "tier-1"
            elif sim >= 0.75:
                tier = "tier-1.5"
            else:
                tier = "tier-2"

            excerpt = self._fetch_excerpt(eid)

            channels.append(ChannelMatch(
                entry_id=eid,
                title=meta.get("title", eid),
                entry_type=meta.get("entry_type", ""),
                domain=meta.get("domain", ""),
                similarity=sim,
                polarity_hint=polarity_hint,
                polarity_markers=polarity_markers,
                excerpt=excerpt,
                tier=tier,
            ))

        return ClaimResolution(
            claim_text=claim_text,
            source_section=source_section,
            claim_approved=claim_approved,
            channel_matches=channels,
            notes=notes,
        )


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a claim against the DS Wiki KB")
    parser.add_argument("claim", help="Claim text to validate")
    parser.add_argument("--top-k",    type=int,   default=15,               help="Chunks to retrieve")
    parser.add_argument("--high-sim", type=float, default=HIGH_SIM_THRESHOLD, help="High-sim threshold")
    parser.add_argument("--low-sim",  type=float, default=LOW_SIM_THRESHOLD,  help="Low-sim threshold")
    args = parser.parse_args()

    v = ResultValidator()
    result = v.validate_claim(args.claim, top_k=args.top_k,
                              high_threshold=args.high_sim,
                              low_threshold=args.low_sim)
    print(result.as_markdown())
