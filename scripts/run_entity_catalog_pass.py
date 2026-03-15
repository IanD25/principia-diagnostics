"""
run_entity_catalog_pass.py — Pass 1.5 + Pass 2b for entity catalog bundles.

Usage:
    python scripts/run_entity_catalog_pass.py <bundle_db> [chroma_dir] [ds_wiki_db]

Defaults:
    chroma_dir  → data/chroma_db
    ds_wiki_db  → data/ds_wiki.db

Steps:
  1. Classify the bundle using classify_dataset_type (warns if not entity_catalog)
  2. Run EntityCatalogPass (Pass 1.5) — inserts synthetic derived_pattern entries
  3. Re-run CrossUniverseQuery (Pass 2b) — full bridge re-computation
  4. Print final bridge stats and top 10 bridges
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import CHROMA_DIR, SOURCE_DB
from ingestion.detector import classify_dataset_type
from ingestion.passes.entity_catalog_pass import EntityCatalogPass
from ingestion.cross_universe_query import CrossUniverseQuery
import sqlite3


def main():
    bundle_db  = sys.argv[1] if len(sys.argv) > 1 else "data/rrp/periodic_table/rrp_periodic_table.db"
    chroma_dir = sys.argv[2] if len(sys.argv) > 2 else str(CHROMA_DIR)
    ds_wiki_db = sys.argv[3] if len(sys.argv) > 3 else str(SOURCE_DB)

    print(f"Entity Catalog Pass — Pass 1.5 + 2b")
    print(f"  Bundle  : {bundle_db}")
    print(f"  Chroma  : {chroma_dir}")
    print(f"  DS Wiki : {ds_wiki_db}")
    print()

    # ── Step 1: Classify ──────────────────────────────────────────────────────
    print("Step 1 — Classify bundle")
    classification = classify_dataset_type(bundle_db)
    dtype      = classification["dataset_type"]
    confidence = classification["confidence"]
    signals    = classification["signals"]

    print(f"  Type       : {dtype} (confidence {confidence:.0%})")
    print(f"  tier_1_5   : {signals['tier_1_5_ratio']:.1%}")
    print(f"  mean_sim   : {signals['mean_sim']:.4f}")
    print(f"  source_types: {signals['source_type_count']}")
    print(f"  hub_frac   : {signals['max_hub_frac']:.1%}")
    print(f"  bridges    : {signals['total_bridges']}")

    if dtype != "entity_catalog":
        print(f"\n  [WARN] Dataset classified as '{dtype}', not 'entity_catalog'.")
        print(f"    Pass 1.5 is designed for entity catalogs. Proceeding anyway.")
    print()

    # ── Step 2: Pass 1.5 — Extract patterns ──────────────────────────────────
    print("Step 2 — Pass 1.5: Extract entity patterns")
    stats = EntityCatalogPass(bundle_db).run()
    print(f"  group_trends    : {stats['group_trends']}")
    print(f"  period_trends   : {stats['period_trends']}")
    print(f"  block_entries   : {stats['block_entries']}")
    print(f"  category_entries: {stats['category_entries']}")
    print(f"  anomaly_entries : {stats['anomaly_entries']}")
    print(f"  notable_anomalies: {stats['notable_anomaly_entries']}")
    print(f"  -----------------------------")
    print(f"  total_synthetic : {stats['total_synthetic']}")
    print()

    # ── Step 3: Pass 2b — Full cross-universe re-run ──────────────────────────
    print("Step 3 — Pass 2b: Cross-universe query (full re-run)")
    cq = CrossUniverseQuery(bundle_db=bundle_db, chroma_dir=chroma_dir)
    bridge_stats = cq.run(ds_wiki_db=ds_wiki_db)
    print()
    print("  Bridge stats:")
    for k, v in bridge_stats.items():
        print(f"    {k:<28s}: {v}")

    # ── Step 4: Top bridges ───────────────────────────────────────────────────
    conn = sqlite3.connect(bundle_db)
    print()
    print("-- Top 15 bridges by similarity --------------------------------------")
    rows = conn.execute("""
        SELECT b.rrp_entry_id, e.source_type, b.ds_entry_id, b.similarity,
               b.proposed_link_type, b.confidence_tier
        FROM cross_universe_bridges b
        JOIN entries e ON e.id = b.rrp_entry_id
        ORDER BY b.similarity DESC LIMIT 15
    """).fetchall()

    for r in rows:
        src_label = f"[{r[1][:8]}]" if r[1] else ""
        print(
            f"  {r[3]:.4f} {src_label:12s} "
            f"{'[' + r[4] + ']':15s} "
            f"{r[0]:40s} <->  {r[2]}"
        )

    print()
    print("-- Derived pattern top bridges ---------------------------------------")
    rows_p = conn.execute("""
        SELECT b.rrp_entry_id, b.ds_entry_id, b.similarity, b.proposed_link_type
        FROM cross_universe_bridges b
        JOIN entries e ON e.id = b.rrp_entry_id
        WHERE e.source_type = 'derived_pattern'
        ORDER BY b.similarity DESC LIMIT 20
    """).fetchall()

    for r in rows_p:
        print(
            f"  {r[2]:.4f}  [{r[3]:12s}]  "
            f"{r[0]:45s} <->  {r[1]}"
        )

    tier_1_5 = conn.execute(
        "SELECT COUNT(*) FROM cross_universe_bridges WHERE confidence_tier = '1.5'"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM cross_universe_bridges").fetchone()[0]
    print()
    print(f"  Total bridges: {total}  |  Tier-1.5 ('analogous to'): {tier_1_5}")
    conn.close()


if __name__ == "__main__":
    main()
