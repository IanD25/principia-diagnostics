"""
opera_parser.py — Parse OPERA FTL neutrino experiment JSON into an RRP bundle.

Source: opera_ftl_neutrino.json (hand-encoded from Adam et al. 2011 + 2012 erratum)
Domain: particle physics / metrology
Status: Contested → Refuted (systematic timing errors identified Jan 2012)

This RRP is a validation dataset for PFD:
  - Tier-1 should show reasonable internal coherence (methodology chain is intact)
  - Tier-2 should show the FTL claim entries bridging POORLY to DS Wiki formal physics
    (SR, Lorentz invariance, causality all contradict the claim)
  - Established physics entries (special_relativity, lorentz_invariance) should bridge WELL

Entry types:
  instantiation   — experimental apparatus
  measurement     — measured quantities
  claim           — the FTL assertion and its implications
  reference_law   — Special Relativity, Lorentz invariance, causality
  systematic_error — hardware faults
  verification    — ICARUS independent confirmation
  method          — audit methodology
"""

import json
import sqlite3
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ingestion.rrp_bundle import create_rrp_bundle, bundle_stats


def parse_opera(
    raw_json: str | Path,
    output_db: str | Path,
) -> sqlite3.Connection:
    """Parse OPERA FTL experiment JSON into an RRP bundle SQLite database.

    Args:
        raw_json:  Path to opera_ftl_neutrino.json
        output_db: Output SQLite path

    Returns:
        Open sqlite3.Connection to the created database
    """
    raw_json  = Path(raw_json)
    output_db = Path(output_db)

    data = json.loads(raw_json.read_text(encoding="utf-8"))
    meta = data["meta"]

    conn = create_rrp_bundle(
        db_path = output_db,
        name    = meta.get("package_name", meta.get("name", "OPERA")),
        source  = str(raw_json),
        fmt     = "opera_json",
    )

    # ── Pass 1: Entries ───────────────────────────────────────────────────────
    for entry in data["entries"]:
        conn.execute(
            """
            INSERT OR IGNORE INTO entries
                (id, title, entry_type, source_type, domain, status, confidence, authoring_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'rrp_ingested')
            """,
            (
                entry["id"],
                entry["title"],
                entry["entry_type"],
                entry.get("source_type", "unknown"),
                entry.get("domain", "physics"),
                entry.get("status", "established"),
                entry.get("confidence", "Tier 2"),
            ),
        )

        # Sections — each section becomes an embeddable chunk
        for order, (section_name, content) in enumerate(entry.get("sections", {}).items()):
            conn.execute(
                """
                INSERT OR IGNORE INTO sections
                    (entry_id, section_name, content, section_order)
                VALUES (?, ?, ?, ?)
                """,
                (entry["id"], section_name, content, order),
            )

    conn.commit()

    # ── Pass 1: Links ─────────────────────────────────────────────────────────
    for order, link in enumerate(data["links"]):
        # Assign confidence tier based on link type
        fwd_types = {"consistent_with", "requires", "enforces", "implies", "agrees_with"}
        contra_types = {"contradicts", "refutes"}
        ltype = link.get("link_type", link.get("type", "related_to"))
        tier = "1.5" if ltype in fwd_types | contra_types else "2"

        conn.execute(
            """
            INSERT OR IGNORE INTO links
                (link_type, source_id, target_id, description, link_order, confidence_tier)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                link.get("link_type", link.get("type", "related_to")),
                link["source"],
                link["target"],
                link.get("description", ""),
                order,
                tier,
            ),
        )

    conn.commit()

    stats = bundle_stats(conn)
    print(f"  OPERA RRP: {stats['total_entries']} entries, {stats['total_links']} links")
    return conn


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse OPERA FTL neutrino JSON into an RRP bundle."
    )
    parser.add_argument(
        "--raw",
        default="data/rrp/opera/opera_raw.json",
        help="Path to opera_ftl_neutrino.json",
    )
    parser.add_argument(
        "--out",
        default="data/rrp/opera/rrp_opera.db",
        help="Output SQLite path",
    )
    args = parser.parse_args()

    output_db = Path(args.out)
    output_db.parent.mkdir(parents=True, exist_ok=True)

    conn = parse_opera(args.raw, output_db)
    stats = bundle_stats(conn)
    conn.close()

    print(f"\nRRP bundle created: {output_db}")
    print(f"  Entries : {stats['total_entries']}")
    print(f"  Links   : {stats['total_links']}")
    print(f"  Sections: {stats['total_sections']}")


if __name__ == "__main__":
    main()
