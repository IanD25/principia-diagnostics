"""
ccbh_cluster_parser.py — Parse CCBH paper cluster into an RRP bundle.

Source: data/rrp/ccbh/ccbh_cluster_raw.json
Output: data/rrp/ccbh/rrp_ccbh_cluster.db

Three-paper cluster on Cosmologically Coupled Black Holes (CCBH):
  - Farrah et al. 2023 (ApJL 944 L31): observational k = 3.11 measurement
  - Cadoni et al. 2025: GR framework for regular-horizon CE solutions
  - DESI Collaboration 2025 (PRL 135 081003): CCBH + DESI DR2 cosmology

Entry types used:
  reference_law  — external physics principle cited as context/constraint
  mechanism      — physical effect or coupling mechanism
  method         — experimental/theoretical methodology
  measurement    — a quantitative result
  claim          — an assertion made by the paper cluster

Link types encode the argument chain:
  motivates, provides_framework_for, produces, supports, implies,
  predicts, is_formalized_by, uses, enables, satisfies,
  provides_physical_basis_for, is_quantified_by, is_constrained_by,
  yields, explains, is_consistent_with, provides_coupling_constant_for,
  provides_singularity_free_model_for, constrains
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ingestion.rrp_bundle import create_rrp_bundle, bundle_stats

# ── Default paths ──────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent.parent
DEFAULT_SOURCE = str(_PROJECT / "data" / "rrp" / "ccbh" / "ccbh_cluster_raw.json")
DEFAULT_OUTPUT = str(_PROJECT / "data" / "rrp" / "ccbh" / "rrp_ccbh_cluster.db")


# ── Domain mapping ─────────────────────────────────────────────────────────────

_DOMAIN_MAP = {
    "method":        "cosmology",
    "measurement":   "cosmology",
    "mechanism":     "general_relativity",
    "claim":         "cosmology",
    "reference_law": "general_relativity",
}

_STATUS_MAP = {
    "method":        "established",
    "measurement":   "established",
    "mechanism":     "established",
    "claim":         "proposed",
    "reference_law": "established",
}


# ── Main parse function ────────────────────────────────────────────────────────

def parse_ccbh_cluster(
    source_path: str = DEFAULT_SOURCE,
    output_path: str = DEFAULT_OUTPUT,
) -> dict:
    """
    Parse ccbh_cluster_raw.json into an RRP bundle at output_path.
    Returns bundle_stats dict.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("CCBH Cluster Parser")
    print(f"  Source : {source_path}")
    print(f"  Output : {output_path}")
    print()

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    meta    = data["metadata"]
    entries = data["entries"]
    links   = data["links"]

    print(f"  Loaded: {len(entries)} entries, {len(links)} links")
    print(f"  Cluster: {meta['title']}")
    print(f"  Papers : {len(meta['papers'])}")
    for p in meta["papers"]:
        print(f"    - {p['id']}: {p['title'][:70]}...")
    print()

    conn = create_rrp_bundle(
        db_path=output_path,
        name=meta["title"],
        source="multi-paper cluster",
        fmt="paper_json",
    )

    # Store cluster-level metadata
    conn.executemany(
        "INSERT OR REPLACE INTO rrp_meta (key, value) VALUES (?, ?)",
        [
            ("cluster_title", meta["title"]),
            ("cluster_note",  meta["note"]),
            ("paper_count",   str(len(meta["papers"]))),
        ]
    )
    # Store per-paper metadata
    for p in meta["papers"]:
        prefix = f"paper_{p['id']}"
        conn.executemany(
            "INSERT OR REPLACE INTO rrp_meta (key, value) VALUES (?, ?)",
            [
                (f"{prefix}_title",   p["title"]),
                (f"{prefix}_authors", p["authors"]),
                (f"{prefix}_journal", p.get("journal", "")),
                (f"{prefix}_year",    str(p.get("year", ""))),
            ]
        )
    conn.commit()

    # ── Pass 1: Insert entries ─────────────────────────────────────────────────
    print("  Pass 1 — Inserting entries...")
    entry_count = 0

    for entry in entries:
        eid         = entry["id"]
        title       = entry["title"]
        etype       = entry["entry_type"]
        paper       = entry.get("paper", "")
        section     = entry.get("section", "")
        description = entry.get("description", "")

        status = _STATUS_MAP.get(etype, "established")
        domain = _DOMAIN_MAP.get(etype, "cosmology")

        conn.execute(
            """INSERT OR IGNORE INTO entries
               (id, title, entry_type, source_type, domain, status, confidence)
               VALUES (?, ?, ?, ?, ?, ?, 'Tier 1')""",
            (eid, title, etype, "paper_section", domain, status)
        )

        # Description → sections table ("Paper Content" section)
        if description:
            conn.execute(
                """INSERT OR IGNORE INTO sections
                   (entry_id, section_name, content, section_order)
                   VALUES (?, 'Paper Content', ?, 1)""",
                (eid, description)
            )

        # Paper section reference → sections table
        if section:
            conn.execute(
                """INSERT OR IGNORE INTO sections
                   (entry_id, section_name, content, section_order)
                   VALUES (?, 'Paper Section', ?, 2)""",
                (eid, section)
            )

        # Properties
        props = [
            ("source_paper",  paper),
            ("paper_section", section),
            ("entry_type",    etype),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO entry_properties (entry_id, property_name, property_value) VALUES (?, ?, ?)",
            [(eid, k, v) for k, v in props if v]
        )

        entry_count += 1

    # ── Pass 2: Insert links ───────────────────────────────────────────────────
    print("  Pass 2 — Inserting links...")
    link_count = 0
    skipped    = 0

    known_ids = {e["id"] for e in entries}

    for link in links:
        src   = link["source"]
        tgt   = link["target"]
        ltype = link["type"]

        if src not in known_ids:
            print(f"    [WARN] Unknown source id '{src}' — skipping link")
            skipped += 1
            continue
        if tgt not in known_ids:
            print(f"    [WARN] Unknown target id '{tgt}' — skipping link")
            skipped += 1
            continue

        conn.execute(
            """INSERT OR IGNORE INTO links
               (link_type, source_id, source_label, target_id, target_label,
                description, confidence_tier)
               VALUES (?, ?, ?, ?, ?, ?, '1')""",
            (
                ltype,
                src, src,
                tgt, tgt,
                f"{src} --[{ltype}]--> {tgt}",
            )
        )
        link_count += 1

    conn.commit()

    stats = bundle_stats(conn)
    conn.close()

    print()
    print(f"  Entries inserted  : {entry_count}")
    print(f"  Links inserted    : {link_count}")
    if skipped:
        print(f"  Links skipped     : {skipped} (unknown IDs)")
    print(f"  Isolated entries  : {stats['isolated_entries']}")
    print(f"  Sections          : {stats['total_sections']}")
    print(f"  Entry properties  : {stats['total_entry_properties']}")
    print()
    print(f"  Entry types:")
    for etype, count in stats["by_type"].items():
        print(f"    {etype:20s} : {count}")
    print()
    print(f"[OK] Bundle written: {output_path}")

    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse CCBH cluster JSON into RRP bundle"
    )
    parser.add_argument(
        "source", nargs="?", default=DEFAULT_SOURCE,
        help="Path to ccbh_cluster_raw.json"
    )
    parser.add_argument(
        "output", nargs="?", default=DEFAULT_OUTPUT,
        help="Path for output .db"
    )
    args = parser.parse_args()

    parse_ccbh_cluster(args.source, args.output)
