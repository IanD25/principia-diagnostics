"""
opera_paper_parser.py — Parse OPERA paper (arXiv:1109.4897) into an RRP bundle.

Source: data/rrp/opera/opera_paper_raw.json
Output: data/rrp/opera/rrp_opera_paper.db

This is the first Phase 3 paper-based parser. Unlike prior parsers (ecoli, zoo,
periodic_table, ieee) which parsed structured data files with empty description
fields, this parser reads a hand-curated JSON where each entry carries a full
prose description extracted verbatim or near-verbatim from the paper text.

Result: description fields average ~300 words, enabling meaningful Tier-2 bridge
detection via the BGE semantic embedding (vs ~20 chars for title-only datasets).

Entry types used:
  method         — experimental apparatus, technique, or analysis procedure
  measurement    — a quantitative result or calibration datum
  mechanism      — physical effect or systematic error source
  claim          — an assertion made by the paper (may be superseded)
  reference_law  — external physics principle cited as context/constraint

Link types encode the argument chain:
  uses, measures, produces, validates, supports, supersedes, contributes_to,
  bounds_uncertainty_of, explains_anomaly_in, would_have_violated,
  is_consistent_with, contradicted, independently_validates, etc.
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
DEFAULT_SOURCE = str(_PROJECT / "data" / "rrp" / "opera" / "opera_paper_raw.json")
DEFAULT_OUTPUT = str(_PROJECT / "data" / "rrp" / "opera" / "rrp_opera_paper.db")


# ── Domain mapping ─────────────────────────────────────────────────────────────

_DOMAIN_MAP = {
    "method":        "experimental_physics",
    "measurement":   "experimental_physics",
    "mechanism":     "experimental_physics",
    "claim":         "experimental_physics",
    "reference_law": "theoretical_physics",
}

_STATUS_MAP = {
    "method":        "established",
    "measurement":   "established",
    "mechanism":     "established",
    "claim":         "superseded",   # default; corrected entries override below
    "reference_law": "established",
}

# Entries that are NOT superseded (override the claim default)
_ACTIVE_CLAIMS = {"corrected_result_2012", "conclusions"}


# ── Main parse function ────────────────────────────────────────────────────────

def parse_opera_paper(
    source_path: str = DEFAULT_SOURCE,
    output_path: str = DEFAULT_OUTPUT,
) -> dict:
    """
    Parse opera_paper_raw.json into an RRP bundle at output_path.
    Returns bundle_stats dict.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("OPERA Paper Parser (arXiv:1109.4897)")
    print(f"  Source : {source_path}")
    print(f"  Output : {output_path}")
    print()

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    meta    = data["metadata"]
    entries = data["entries"]
    links   = data["links"]

    print(f"  Loaded: {len(entries)} entries, {len(links)} links")
    print(f"  Paper : {meta['title']}")
    print(f"  arXiv : {meta['arxiv_id']} ({meta['version']})")
    print()

    conn = create_rrp_bundle(
        db_path=output_path,
        name=f"OPERA Paper — {meta['title']}",
        source=f"arXiv:{meta['arxiv_id']}",
        fmt="paper_json",
    )

    # Store paper-level metadata as extra rrp_meta rows
    conn.executemany(
        "INSERT OR REPLACE INTO rrp_meta (key, value) VALUES (?, ?)",
        [
            ("paper_title",   meta["title"]),
            ("paper_authors", meta["authors"]),
            ("arxiv_id",      meta["arxiv_id"]),
            ("journal",       meta["journal"]),
            ("paper_version", meta["version"]),
            ("paper_note",    meta["note"]),
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
        section     = entry.get("section", "")
        description = entry.get("description", "")

        # Determine status
        if etype == "claim" and eid not in _ACTIVE_CLAIMS:
            status = "superseded"
        else:
            status = _STATUS_MAP.get(etype, "established")

        domain = _DOMAIN_MAP.get(etype, "experimental_physics")

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
            ("arxiv_id",      meta["arxiv_id"]),
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

    # Index known entry IDs for validation
    known_ids = {e["id"] for e in entries}

    for link in links:
        src  = link["source"]
        tgt  = link["target"]
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
        description="Parse OPERA paper JSON (arXiv:1109.4897) into RRP bundle"
    )
    parser.add_argument(
        "source", nargs="?", default=DEFAULT_SOURCE,
        help="Path to opera_paper_raw.json"
    )
    parser.add_argument(
        "output", nargs="?", default=DEFAULT_OUTPUT,
        help="Path for output .db"
    )
    args = parser.parse_args()

    parse_opera_paper(args.source, args.output)
