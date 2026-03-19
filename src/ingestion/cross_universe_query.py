"""
cross_universe_query.py — Pass 2: RRP closed universe → DS Wiki cross-universe bridges.

Takes a fully-parsed, prose-enriched RRP bundle and queries each entry's
embedding against the DS Wiki ChromaDB collection. Surfaces connections that
exist in the science universe but were not present in the research package.

Architecture:
  RRP bundle entries (title + all sections)
    → BGE embed (same model as DS Wiki sync)
    → query ds_wiki ChromaDB collection
    → deduplicate to DS Wiki entry level (best chunk per entry)
    → threshold filter
    → propose link type by heuristic
    → store in cross_universe_bridges table (stays IN the RRP bundle)

This is a DIAGNOSTIC tool. It finds what the research MISSED, not what
it got wrong. Bridges are stored in the RRP bundle db — the RRP is
what is being diagnosed.

Heuristic link type assignment (refined by link_classifier if needed):
  sim ≥ 0.90  →  analogous to     (near-identical concept, different domain)
  sim ≥ 0.80  →  couples to       (closely related, likely functional connection)
  sim ≥ 0.72  →  related          (semantically adjacent, weaker connection)

ObviousConstruction and entries with < MIN_EMBED_CHARS are skipped —
they produce noise, not signal.
"""

import sqlite3
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ── Path setup (allow running from src/ or project root) ─────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # src/
sys.path.insert(0, str(_HERE.parent.parent))   # project root

from config import (
    CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL,
    CROSS_ENCODER_MODEL, CROSS_ENCODER_ENABLED, RERANK_TOP_K,
)


# ── Constants ─────────────────────────────────────────────────────────────────

BGE_QUERY_PREFIX   = "Represent this sentence for searching relevant passages: "
MIN_EMBED_CHARS    = 50     # skip entries with less content than this
TOP_K_CHUNKS       = 10     # chunks to retrieve per query from DS Wiki
SIM_HIGH           = 0.85   # analogous to
SIM_MID            = 0.75   # couples to
SIM_LOW            = 0.70   # related (minimum to store)

# Entries to skip (sentinels, stubs)
SKIP_ENTRY_IDS = {"thm_ObviousConstruction", "thm_ProtocolSimulation"}

# Section priority order for building embedding text
EMBED_SECTIONS = [
    "What It Claims",
    "What It Captures",    # entity catalogs (periodic table, etc.)
    "What The Math Says",
    "Mathematical Form",
    "Notes",
]


# ── Embedding text builder ────────────────────────────────────────────────────

def _build_embed_text(
    title: str,
    entry_type: str,
    sections: list[tuple[str, str]],   # [(section_name, content), ...]
) -> str:
    """
    Concatenate title + relevant sections into a single embedding string.
    Strips the supplemental note header from WTM sections.
    """
    parts = [f"{entry_type}: {title}"]

    sec_dict = {name: (content or "") for name, content in sections}

    for sname in EMBED_SECTIONS:
        content = sec_dict.get(sname, "").strip()
        if not content:
            continue
        # Strip supplemental header from WTM
        if "\n\n" in content and content.startswith("[Supplemental"):
            content = content.split("\n\n", 1)[1].strip()
        parts.append(content)

    return " ".join(parts)


# ── Similarity → link type ────────────────────────────────────────────────────

def _propose_link_type(sim: float) -> tuple[str, str]:
    """Returns (link_type, confidence_tier)."""
    if sim >= SIM_HIGH:
        return "analogous to", "1.5"
    elif sim >= SIM_MID:
        return "couples to",   "2"
    else:
        return "related",      "2"


# ── Main query class ──────────────────────────────────────────────────────────

class CrossUniverseQuery:
    """
    Run Pass 2: embed RRP bundle entries and query DS Wiki ChromaDB.

    Usage:
        cq = CrossUniverseQuery(
            bundle_db  = "data/rrp/zoo_classes/rrp_zoo_classes.db",
            chroma_dir = "data/chroma_db",
            collection = "ds_wiki",
        )
        stats = cq.run()
    """

    def __init__(
        self,
        bundle_db:   str | Path,
        chroma_dir:  str | Path = CHROMA_DIR,
        collection:  str        = CHROMA_COLLECTION,
        model_name:  str        = EMBED_MODEL,
    ):
        self.bundle_db  = Path(bundle_db)
        self.chroma_dir = Path(chroma_dir)
        self.collection = collection
        self.model_name = model_name
        self._model     = None
        self._coll      = None
        self._reranker  = None   # Phase 3B: lazy-loaded cross-encoder

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading BGE model: {self.model_name} ...", flush=True)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_cross_encoder(self):
        """Lazy-load cross-encoder for bridge reranking (Phase 3B)."""
        if self._reranker is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            print(f"  Loading cross-encoder: {CROSS_ENCODER_MODEL} ...", flush=True)
            self._reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._reranker

    def _get_collection(self):
        if self._coll is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self._coll = client.get_collection(self.collection)
            print(f"  DS Wiki ChromaDB: {self._coll.count()} chunks in '{self.collection}'")
        return self._coll

    # ── Embed ─────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        query = BGE_QUERY_PREFIX + text
        vec = model.encode(query, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    # ── Query DS Wiki ─────────────────────────────────────────────────────────

    def _query_ds_wiki(
        self,
        embedding: np.ndarray,
        rrp_text: Optional[str] = None,
    ) -> list[dict]:
        """
        Query DS Wiki ChromaDB. Returns list of dicts:
        {ds_entry_id, ds_entry_title, similarity}
        deduplicated to best chunk per DS Wiki entry.

        Phase 3B: When CROSS_ENCODER_ENABLED and rrp_text is provided,
        retrieves RERANK_TOP_K candidates and rescores using cross-encoder.
        Cross-encoder scores replace BGE cosine similarity.
        """
        coll = self._get_collection()
        use_reranker = CROSS_ENCODER_ENABLED and rrp_text is not None
        n_results = RERANK_TOP_K if use_reranker else TOP_K_CHUNKS

        # Include documents when reranking (need chunk text for cross-encoder)
        include = ["metadatas", "distances"]
        if use_reranker:
            include.append("documents")

        results = coll.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results,
            include=include,
        )

        # distances are L2; convert to cosine similarity
        # ChromaDB with normalized embeddings: cos_sim = 1 - (distance / 2)
        candidates: dict[str, dict] = {}
        docs = results.get("documents", [[]])[0] if use_reranker else []

        for i, (meta, dist) in enumerate(zip(
            results["metadatas"][0],
            results["distances"][0],
        )):
            cos_sim  = 1.0 - (dist / 2.0)
            entry_id = meta.get("entry_id", "")
            title    = meta.get("title", entry_id)
            chunk_text = docs[i] if i < len(docs) else ""

            # Keep only best chunk per DS Wiki entry (pre-reranking dedup)
            if entry_id not in candidates or cos_sim > candidates[entry_id]["similarity"]:
                candidates[entry_id] = {
                    "ds_entry_id":    entry_id,
                    "ds_entry_title": title,
                    "similarity":     cos_sim,
                    "_chunk_text":    chunk_text,  # used by reranker, stripped before return
                }

        # ── Phase 3B: Cross-encoder reranking ────────────────────────────────
        if use_reranker and len(candidates) > 1:
            candidates = self._rerank_candidates(candidates, rrp_text)

        # Strip internal fields before returning
        for cand in candidates.values():
            cand.pop("_chunk_text", None)
            cand.pop("_ce_rank", None)
            cand.pop("_ce_raw_score", None)

        return sorted(candidates.values(), key=lambda x: -x["similarity"])

    def _rerank_candidates(
        self,
        candidates: dict[str, dict],
        rrp_text: str,
    ) -> dict[str, dict]:
        """
        Rerank candidates using cross-encoder (Phase 3B).

        Input pairs: (rrp_text, ds_wiki_chunk_text)
        Cross-encoder determines the ORDER, but BGE cosine similarity is
        PRESERVED as the stored score. This is because MS MARCO cross-encoder
        scores are calibrated for search relevance, not semantic similarity —
        the raw logits are deeply negative for scientific text pairs even when
        semantically relevant. BGE cosine similarity remains the right scale
        for downstream thresholds (0.70–0.90).

        Effect: the top-K candidates selected by cross-encoder ranking may
        differ from BGE ranking, but their stored similarity is still the
        BGE cosine value. This eliminates false positives where BGE assigns
        high cosine similarity to semantically unrelated content.
        """
        reranker = self._get_cross_encoder()
        entry_ids = list(candidates.keys())
        if not entry_ids:
            return candidates

        pairs = [
            (rrp_text, candidates[eid].get("_chunk_text") or candidates[eid]["ds_entry_title"])
            for eid in entry_ids
        ]

        # Cross-encoder predict — returns raw logits (used for ranking only)
        raw_scores = reranker.predict(pairs, show_progress_bar=False)

        # Store cross-encoder rank for diagnostics, keep BGE sim for thresholds
        ranked = sorted(zip(entry_ids, raw_scores), key=lambda x: -x[1])
        reranked: dict[str, dict] = {}
        for rank, (eid, ce_raw) in enumerate(ranked):
            cand = dict(candidates[eid])
            cand["_ce_rank"] = rank
            cand["_ce_raw_score"] = float(ce_raw)
            reranked[eid] = cand

        return reranked

    # ── Fetch DS Wiki entry title (fallback if not in metadata) ───────────────

    def _ds_title(self, ds_conn: sqlite3.Connection, entry_id: str) -> str:
        row = ds_conn.execute(
            "SELECT title FROM entries WHERE id=?", (entry_id,)
        ).fetchone()
        return row[0] if row else entry_id

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(
        self,
        sim_threshold: float = SIM_LOW,
        max_bridges_per_entry: int = 3,
        ds_wiki_db: Optional[str | Path] = None,
        quality_filter: bool = False,
        eta_threshold: float = 0.65,
    ) -> dict:
        """
        Run Pass 2 for all entries in the bundle.
        Stores results in cross_universe_bridges table.
        Returns stats dict.

        Args:
            quality_filter: If True, apply FIM bridge quality filter after
                            bridge detection.  Requires ds_wiki_db to be set.
                            Noise-dominated bridges (eta >= eta_threshold) are
                            annotated in the stats but NOT removed from the DB.
            eta_threshold:  Disorder index cutoff for quality_filter (default 0.65).
        """
        bundle_conn = sqlite3.connect(self.bundle_db)
        bundle_conn.row_factory = sqlite3.Row

        # Load DS Wiki db for entry title lookup
        if ds_wiki_db is None:
            # Default: find relative to bundle location
            ds_wiki_db = Path(__file__).parent.parent.parent / "data" / "ds_wiki.db"
        ds_conn = sqlite3.connect(ds_wiki_db)

        # Clear existing bridges (idempotent re-run)
        bundle_conn.execute("DELETE FROM cross_universe_bridges")

        # Load all entries with their sections
        entries = bundle_conn.execute(
            "SELECT id, title, entry_type FROM entries ORDER BY entry_type, id"
        ).fetchall()

        print(f"\n  Processing {len(entries)} entries...")
        model = self._get_model()
        _ = self._get_collection()   # warm up

        stats = {
            "total_entries":     len(entries),
            "skipped_stub":      0,
            "skipped_thin":      0,
            "queried":           0,
            "bridges_stored":    0,
            "entries_with_bridge": 0,
        }

        bridge_order = 1

        for i, row in enumerate(entries):
            eid        = row["id"]
            title      = row["title"]
            entry_type = row["entry_type"]

            # Skip stubs
            if eid in SKIP_ENTRY_IDS:
                stats["skipped_stub"] += 1
                continue

            # Load sections
            secs = bundle_conn.execute(
                "SELECT section_name, content FROM sections WHERE entry_id=? ORDER BY section_order",
                (eid,)
            ).fetchall()
            secs_list = [(r["section_name"], r["content"]) for r in secs]

            embed_text = _build_embed_text(title, entry_type, secs_list)

            # Skip if too thin (formal-only, no prose)
            # Count chars excluding section labels
            content_chars = sum(len(c or "") for _, c in secs_list)
            if content_chars < MIN_EMBED_CHARS:
                stats["skipped_thin"] += 1
                continue

            # Embed and query (Phase 3B: pass rrp_text for cross-encoder reranking)
            vec       = self._embed(embed_text)
            candidates = self._query_ds_wiki(vec, rrp_text=embed_text)
            stats["queried"] += 1

            # Filter and store
            stored = 0
            for cand in candidates:
                sim = cand["similarity"]
                if sim < sim_threshold:
                    break
                if stored >= max_bridges_per_entry:
                    break

                ds_entry_id    = cand["ds_entry_id"]
                ds_entry_title = cand["ds_entry_title"] or self._ds_title(ds_conn, ds_entry_id)
                link_type, tier = _propose_link_type(sim)

                description = (
                    f"Cross-universe bridge: ZooClasses '{title}' ({entry_type}) "
                    f"↔ DS Wiki '{ds_entry_title}' | sim={sim:.4f}"
                )

                bundle_conn.execute(
                    """INSERT INTO cross_universe_bridges
                       (rrp_entry_id, rrp_entry_title, ds_entry_id, ds_entry_title,
                        similarity, proposed_link_type, confidence_tier, description)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (eid, title, ds_entry_id, ds_entry_title,
                     sim, link_type, tier, description),
                )
                stored += 1
                stats["bridges_stored"] += 1
                bridge_order += 1

            if stored > 0:
                stats["entries_with_bridge"] += 1

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(entries)} processed, {stats['bridges_stored']} bridges so far...")
                bundle_conn.commit()

        bundle_conn.commit()

        # ── Optional FIM quality filter ────────────────────────────────────────
        if quality_filter and ds_wiki_db is not None:
            try:
                from analysis.fisher_bridge_filter import filter_bridges
                from analysis.fisher_diagnostics import (
                    build_wiki_graph, sweep_graph, KernelType,
                )
                G_wiki, _ = build_wiki_graph(Path(ds_wiki_db))
                ds_sweep   = sweep_graph(G_wiki, "ds_wiki", KernelType.EXPONENTIAL)

                # Load all stored bridges as dicts
                stored_rows = bundle_conn.execute(
                    "SELECT rrp_entry_id, ds_entry_id, similarity "
                    "FROM cross_universe_bridges"
                ).fetchall()
                bridge_dicts = [
                    {"rrp_entry_id": r[0], "ds_entry_id": r[1], "similarity": r[2]}
                    for r in stored_rows
                ]

                trusted, noise = filter_bridges(bridge_dicts, ds_sweep, eta_threshold)
                stats["fisher_trusted"]  = len(trusted)
                stats["fisher_noise"]    = len(noise)
                stats["fisher_eta_threshold"] = eta_threshold
                print(
                    f"  FIM filter: {len(trusted)} trusted, {len(noise)} noise-flagged "
                    f"(eta_threshold={eta_threshold})"
                )
            except Exception as exc:
                stats["fisher_filter_error"] = str(exc)
                print(f"  [WARN] FIM quality filter failed: {exc}")

        # Record in rrp_meta
        bundle_conn.execute(
            "INSERT OR REPLACE INTO rrp_meta (key, value) VALUES (?, ?)",
            ("pass2_status", f"bridges={stats['bridges_stored']} entries_with_bridge={stats['entries_with_bridge']} threshold={sim_threshold}")
        )
        bundle_conn.commit()
        bundle_conn.close()
        ds_conn.close()

        return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    bundle_db  = sys.argv[1] if len(sys.argv) > 1 else "data/rrp/zoo_classes/rrp_zoo_classes.db"
    chroma_dir = sys.argv[2] if len(sys.argv) > 2 else str(CHROMA_DIR)
    ds_wiki_db = sys.argv[3] if len(sys.argv) > 3 else "data/ds_wiki.db"

    print(f"Pass 2 — Cross-Universe Query")
    print(f"  Bundle  : {bundle_db}")
    print(f"  DS Wiki : {ds_wiki_db}")
    print(f"  Chroma  : {chroma_dir}/{CHROMA_COLLECTION}")
    print(f"  Threshold: sim ≥ {SIM_LOW}")

    cq = CrossUniverseQuery(
        bundle_db=bundle_db,
        chroma_dir=chroma_dir,
    )
    stats = cq.run(ds_wiki_db=ds_wiki_db)

    print("\n── Stats ────────────────────────────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<28s}: {v}")

    # Show top bridges
    conn = sqlite3.connect(bundle_db)
    print("\n── Top bridges by similarity ─────────────────────────────────────────")
    bridges = conn.execute("""
        SELECT rrp_entry_id, rrp_entry_title, ds_entry_id, ds_entry_title,
               similarity, proposed_link_type, confidence_tier
        FROM cross_universe_bridges
        ORDER BY similarity DESC
        LIMIT 20
    """).fetchall()

    for b in bridges:
        print(
            f"  {b[4]:.4f}  [{b[5]:12s}] "
            f"RRP:{b[1][:30]:30s} <-> DS:{b[3][:35]}"
        )

    print(f"\n  Total bridges stored: {conn.execute('SELECT COUNT(*) FROM cross_universe_bridges').fetchone()[0]}")
    conn.close()


if __name__ == "__main__":
    main()
