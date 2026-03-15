"""
periodic_table_parser.py — Parse MIT Periodic Table JSON into an RRP bundle.

Source: https://github.com/Bowserinator/Periodic-Table-JSON
File:   PeriodicTableJSON.json  (119 elements, 33 fields each)
Format: flat_json — one flat dict per element, no nested structure

Pass 1 — Deterministic field mapping
=====================================
Element → DS Wiki entry_type:
  all elements → reference_law  (each element is an established scientific fact)

Source type:
  all entries → source_type = "element"

Section mapping:
  summary                → "What It Captures"   (WIC — already good Wikipedia prose)
  identity fields        → "Identity"            (number, symbol, category, period, group, block, phase)
  physical fields        → "Physical Properties" (mass, density, melt, boil, molar_heat, appearance)
  electronic fields      → "Electronic Structure"(shells, configuration, electronegativity, affinity, ionization)
  discovery fields       → "Discovery"           (discovered_by, named_by)

Link type mapping:
  same group, consecutive periods → analogous_to  (tier 1.5 — chemical analogues)
  same period, adjacent Z         → correlates_with  (tier 2 — periodic trends)
  same category, small groups     → analogous_to  (tier 1.5 — for groups ≤6 elements)

ID scheme:
  elem_{symbol}   e.g. elem_H, elem_Fe, elem_Au

No prose enricher needed — summary field is already full prose.
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ingestion.rrp_bundle import create_rrp_bundle, bundle_stats


# ── Categories small enough to fully mesh ────────────────────────────────────
# (only categories with ≤ 8 known/established members get extra same-category links)
SMALL_CATEGORIES = {
    "alkali metal",
    "alkaline earth metal",
    "noble gas",
    "diatomic nonmetal",
    "metalloid",
}


def _elem_id(symbol: str) -> str:
    return f"elem_{symbol}"


def _fmt_optional(value, unit: str = "", precision: int = 3) -> Optional[str]:
    """Return formatted string or None if value is missing."""
    if value is None:
        return None
    if isinstance(value, float):
        return f"{value:.{precision}f}{(' ' + unit) if unit else ''}"
    return f"{value}{(' ' + unit) if unit else ''}"


def _build_identity_section(el: dict) -> str:
    lines = [
        f"Atomic number: {el['number']}",
        f"Symbol: {el['symbol']}",
        f"Category: {el['category']}",
        f"Period: {el['period']}   Group: {el['group']}   Block: {el['block']}",
        f"Standard phase: {el['phase']}",
    ]
    if el.get("electron_configuration_semantic"):
        lines.append(f"Electron configuration: {el['electron_configuration_semantic']}")
    return "\n".join(lines)


def _build_physical_section(el: dict) -> Optional[str]:
    lines = []
    am = _fmt_optional(el.get("atomic_mass"), "u", 4)
    if am:
        lines.append(f"Atomic mass: {am}")
    density = _fmt_optional(el.get("density"), "g/cm³", 4)
    if density:
        lines.append(f"Density: {density}")
    melt = _fmt_optional(el.get("melt"), "K", 3)
    if melt:
        lines.append(f"Melting point: {melt}")
    boil = _fmt_optional(el.get("boil"), "K", 3)
    if boil:
        lines.append(f"Boiling point: {boil}")
    molar_heat = _fmt_optional(el.get("molar_heat"), "J/(mol·K)", 3)
    if molar_heat:
        lines.append(f"Molar heat capacity: {molar_heat}")
    appearance = el.get("appearance")
    if appearance:
        lines.append(f"Appearance: {appearance}")
    return "\n".join(lines) if lines else None


def _build_electronic_section(el: dict) -> Optional[str]:
    lines = []
    shells = el.get("shells")
    if shells:
        lines.append(f"Shell configuration: {shells}")
    cfg = el.get("electron_configuration")
    if cfg:
        lines.append(f"Electron configuration: {cfg}")
    en = _fmt_optional(el.get("electronegativity_pauling"), "(Pauling scale)", 2)
    if en:
        lines.append(f"Electronegativity: {en}")
    ea = _fmt_optional(el.get("electron_affinity"), "kJ/mol", 3)
    if ea:
        lines.append(f"Electron affinity: {ea}")
    ie = el.get("ionization_energies")
    if ie:
        ie_str = ", ".join(f"{v:.1f}" for v in ie[:4])
        suffix = " ..." if len(ie) > 4 else ""
        lines.append(f"Ionization energies (kJ/mol): {ie_str}{suffix}")
    return "\n".join(lines) if lines else None


def _build_discovery_section(el: dict) -> Optional[str]:
    lines = []
    disc = el.get("discovered_by")
    if disc:
        lines.append(f"Discovered by: {disc}")
    named = el.get("named_by")
    if named:
        lines.append(f"Named by: {named}")
    return "\n".join(lines) if lines else None


# ── Main parser class ─────────────────────────────────────────────────────────

class PeriodicTableParser:
    """
    Parses PeriodicTableJSON.json into an RRP bundle SQLite database.

    Call parse(output_db) to produce the bundle.
    """

    BUNDLE_META = {
        "package_name": "PeriodicTable",
        "source": "https://github.com/Bowserinator/Periodic-Table-JSON",
        "format": "flat_json",
    }

    def __init__(self, raw_json_path: str | Path):
        self.raw_path = Path(raw_json_path)

    def parse(self, output_db: str | Path) -> sqlite3.Connection:
        """Run Pass 1. Returns an open connection to the bundle DB."""
        with open(self.raw_path) as f:
            data = json.load(f)
        elements: list[dict] = data["elements"]

        conn = create_rrp_bundle(
            str(output_db),
            self.BUNDLE_META["package_name"],
            self.BUNDLE_META["source"],
            self.BUNDLE_META["format"],
        )

        # Pass 1a — insert all entries
        print(f"  Inserting {len(elements)} elements...")
        for el in elements:
            self._insert_entry(conn, el)

        # Pass 1b — insert links (IDs all exist now)
        print("  Building link graph...")
        n_links = self._insert_all_links(conn, elements)
        print(f"  Inserted {n_links} links")

        conn.commit()
        stats = bundle_stats(conn)
        print(f"  Bundle: {stats['total_entries']} entries, {stats['total_links']} links")
        return conn

    # ── Entry insertion ───────────────────────────────────────────────────────

    def _insert_entry(self, conn: sqlite3.Connection, el: dict) -> None:
        eid = _elem_id(el["symbol"])
        title = el["name"]

        # Domain: chemistry for all; chemistry · physics for radioactive/quantum-notable
        z = el["number"]
        category = el.get("category", "")
        if z > 83 or category == "noble gas":
            domain = "chemistry · physics"
        else:
            domain = "chemistry"

        conn.execute("""
            INSERT OR IGNORE INTO entries
              (id, title, entry_type, source_type, domain, status, confidence)
            VALUES (?, ?, 'reference_law', 'element', ?, 'established', 'Tier 1')
        """, (eid, title, domain))

        # ── sections ─────────────────────────────────────────────────────────
        sections = []

        # WIC — Wikipedia summary (already good prose)
        summary = el.get("summary", "").strip()
        if summary:
            sections.append(("What It Captures", summary, 0))

        # Identity
        sections.append(("Identity", _build_identity_section(el), 1))

        # Physical properties
        phys = _build_physical_section(el)
        if phys:
            sections.append(("Physical Properties", phys, 2))

        # Electronic structure
        elec = _build_electronic_section(el)
        if elec:
            sections.append(("Electronic Structure", elec, 3))

        # Discovery
        disc = _build_discovery_section(el)
        if disc:
            sections.append(("Discovery", disc, 4))

        for sname, content, order in sections:
            conn.execute("""
                INSERT OR IGNORE INTO sections (entry_id, section_name, content, section_order)
                VALUES (?, ?, ?, ?)
            """, (eid, sname, content, order))

        # ── entry_properties ─────────────────────────────────────────────────
        props = [
            ("atomic_number",  str(el["number"])),
            ("symbol",         el["symbol"]),
            ("category",       category),
            ("block",          el.get("block", "")),
            ("period",         str(el["period"])),
            ("group",          str(el["group"])),
            ("phase_at_stp",   el.get("phase", "")),
        ]
        if el.get("atomic_mass") is not None:
            props.append(("atomic_mass_u", f"{el['atomic_mass']:.4f}"))
        if el.get("electronegativity_pauling") is not None:
            props.append(("electronegativity_pauling", f"{el['electronegativity_pauling']:.2f}"))
        if el.get("density") is not None:
            props.append(("density_g_cm3", f"{el['density']:.4f}"))
        if el.get("melt") is not None:
            props.append(("melting_point_K", f"{el['melt']:.3f}"))
        if el.get("boil") is not None:
            props.append(("boiling_point_K", f"{el['boil']:.3f}"))
        ie = el.get("ionization_energies")
        if ie and len(ie) > 0 and ie[0] is not None:
            props.append(("first_ionization_energy_kJ_mol", f"{ie[0]:.1f}"))
        if el.get("electron_affinity") is not None:
            props.append(("electron_affinity_kJ_mol", f"{el['electron_affinity']:.3f}"))
        if el.get("molar_heat") is not None:
            props.append(("molar_heat_J_mol_K", f"{el['molar_heat']:.3f}"))

        for pname, pval in props:
            if pval:
                conn.execute("""
                    INSERT OR IGNORE INTO entry_properties (entry_id, property_name, property_value)
                    VALUES (?, ?, ?)
                """, (eid, pname, pval))

    # ── Link graph ────────────────────────────────────────────────────────────

    def _insert_all_links(self, conn: sqlite3.Connection, elements: list[dict]) -> int:
        """
        Build three tiers of links:
          1. Same-group chains (consecutive periods) → analogous_to  (tier 1.5)
          2. Same-period neighbors (adjacent Z)      → correlates_with (tier 2)
          3. Same small-category mesh                → analogous_to  (tier 1.5)
        Returns total link count inserted.
        """
        inserted = 0
        seen = set()

        def add_link(src_id, tgt_id, link_type, tier, rationale):
            nonlocal inserted
            key = (min(src_id, tgt_id), max(src_id, tgt_id), link_type)
            if key in seen:
                return
            seen.add(key)
            order = conn.execute(
                "SELECT COALESCE(MAX(link_order),0)+1 FROM links"
            ).fetchone()[0]
            conn.execute("""
                INSERT OR IGNORE INTO links
                  (source_id, target_id, link_type,
                   description, confidence_tier, link_order)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (src_id, tgt_id, link_type, rationale, tier, order))
            inserted += 1

        # Index by group and period for fast lookup
        by_group = defaultdict(list)   # group → [el sorted by period, Z]
        by_period = defaultdict(list)  # period → [el sorted by Z]
        by_category = defaultdict(list)

        for el in elements:
            by_group[el["group"]].append(el)
            by_period[el["period"]].append(el)
            by_category[el["category"]].append(el)

        # Sort each bucket by atomic number for deterministic ordering
        for bucket in (by_group, by_period, by_category):
            for k in bucket:
                bucket[k].sort(key=lambda e: e["number"])

        # 1. Same-group chains (consecutive members)
        for group, members in by_group.items():
            for i in range(len(members) - 1):
                a, b = members[i], members[i + 1]
                add_link(
                    _elem_id(a["symbol"]),
                    _elem_id(b["symbol"]),
                    "analogous_to",
                    "1.5",
                    f"Both in Group {group} — same valence electron pattern, "
                    f"analogous chemical behaviour across periods {a['period']} and {b['period']}.",
                )

        # 2. Same-period neighbors (adjacent Z within the same period)
        for period, members in by_period.items():
            for i in range(len(members) - 1):
                a, b = members[i], members[i + 1]
                # Only link if truly adjacent (Z differs by 1) — avoids spurious
                # jumps at lanthanide/actinide insertion points
                if b["number"] - a["number"] == 1:
                    add_link(
                        _elem_id(a["symbol"]),
                        _elem_id(b["symbol"]),
                        "correlates_with",
                        "2",
                        f"Adjacent elements in Period {period} (Z={a['number']} and Z={b['number']}) "
                        f"— linked by periodic trends (electronegativity, ionisation energy, atomic radius).",
                    )

        # 3. Same small-category mesh (fully connected within small categories)
        for category, members in by_category.items():
            cat_clean = category.lower().strip()
            # Skip 'unknown...' categories and large ones (transition metals=35, etc.)
            if cat_clean.startswith("unknown") or len(members) > 8:
                continue
            if cat_clean not in SMALL_CATEGORIES:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    # Skip if already linked via group chain (consecutive group members)
                    # — only add cross-group mesh for same-category non-consecutive pairs
                    add_link(
                        _elem_id(a["symbol"]),
                        _elem_id(b["symbol"]),
                        "analogous_to",
                        "1.5",
                        f"Both are {category}s — share characteristic chemical and physical properties "
                        f"of this element family.",
                    )

        return inserted


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_periodic_table(
    raw_json: str | Path,
    output_db: str | Path,
) -> sqlite3.Connection:
    """Convenience wrapper used by scripts and CLI."""
    parser = PeriodicTableParser(raw_json)
    return parser.parse(output_db)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse Periodic Table JSON → RRP bundle")
    ap.add_argument("raw_json", help="Path to PeriodicTableJSON.json")
    ap.add_argument("output_db", help="Path for output .db file")
    args = ap.parse_args()

    print(f"Parsing {args.raw_json}")
    conn = parse_periodic_table(args.raw_json, args.output_db)
    conn.close()
    print("Done.")
