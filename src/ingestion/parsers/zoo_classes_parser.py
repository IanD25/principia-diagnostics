"""
zoo_classes_parser.py — Parse Timeroot/ZooClasses JSON into an RRP bundle.

Source: https://github.com/Timeroot/ZooClasses
Files:  classes.json, theorems.json, conjectures.json, problems.json,
        properties.json, problem_types.json, references.json

Pass 1 — Deterministic field mapping
=====================================
ZooClasses → DS Wiki entry_type:
  classes       → reference_law   (formally defined complexity classes)
  theorems      → theorem         (proved containment/separation results)
  conjectures   → open_question   (P≠NP, etc.)
  problems      → reference_law   (canonical problems: 3SUM, SAT)
  properties    → entry_properties values (tags, not entries)
  problem_types → entry_properties taxonomy (not entries)

Section mapping:
  desc / description  → "What It Claims"
  content / formal    → "Mathematical Form"
  notes               → "Notes"
  type (class type)   → "Concept Tags" entry_properties

Link type mapping:
  related    (classes)     → analogous to   (untyped cross-link)
  impliedby  (theorems)    → derives from   (theorem follows from these)
  implies    (conjectures) → predicts for   (conjecture predicts these results)
  not_implies             → tensions with  (contradicts if true)

ID normalisation:
  Special characters in complexity class names (# ^ [ ]) are preserved
  but prefixed by source type to avoid collisions:
    class_NP, class_#P, theorem_Space-hierarchy, conj_P!=NP, prob_3SUM
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ingestion.rrp_bundle import create_rrp_bundle, bundle_stats


# ── ID normalisation ─────────────────────────────────────────────────────────

PREFIX = {
    "classes":    "cls",
    "theorems":   "thm",
    "conjectures":"conj",
    "problems":   "prob",
}


def _make_id(source_type: str, name: str) -> str:
    """
    Generate a stable, unique entry ID for this RRP bundle.
    Keeps the original name but adds a short prefix per source type.
    Keeps all characters — the RRP universe is self-contained.
    """
    prefix = PREFIX.get(source_type, source_type[:3])
    # Collapse runs of whitespace, keep everything else
    clean = re.sub(r"\s+", "_", name.strip())
    return f"{prefix}_{clean}"


# ── Entry type mapping ────────────────────────────────────────────────────────

ENTRY_TYPE = {
    "classes":    "reference_law",
    "theorems":   "theorem",
    "conjectures":"open_question",
    "problems":   "reference_law",
}

STATUS = {
    "classes":    "established",
    "theorems":   "established",
    "conjectures":"open",
    "problems":   "established",
}

CONFIDENCE = {
    "classes":    "Tier 1",
    "theorems":   "Tier 1",
    "conjectures":"Tier 3",
    "problems":   "Tier 1",
}


# ── Link type mapping ─────────────────────────────────────────────────────────

LINK_MAP = {
    "related":      "analogous to",
    "impliedby":    "derives from",
    "implies":      "predicts for",
    "not_implies":  "tensions with",
}

LINK_TIER = {
    "related":      "1.5",
    "impliedby":    "1",
    "implies":      "1.5",
    "not_implies":  "1.5",
}


# ── Parser ────────────────────────────────────────────────────────────────────

class ZooClassesParser:
    """
    Parses a ZooClasses data directory into an RRP bundle database.

    Usage:
        parser = ZooClassesParser(raw_dir="data/rrp/zoo_classes/raw")
        conn = parser.parse(output_db="data/rrp/zoo_classes/rrp_zoo_classes.db")
        print(bundle_stats(conn))
    """

    SOURCE_NAME = "ZooClasses"
    SOURCE_URL  = "https://github.com/Timeroot/ZooClasses"
    FORMAT      = "zoo_classes_json"

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self._id_map: dict[str, str] = {}   # original_name → entry_id

    # ── Public ────────────────────────────────────────────────────────────────

    def parse(self, output_db: str | Path) -> sqlite3.Connection:
        """
        Full parse: read all source files, populate bundle db, return open connection.
        """
        conn = create_rrp_bundle(
            output_db,
            name=self.SOURCE_NAME,
            source=self.SOURCE_URL,
            fmt=self.FORMAT,
        )

        # Pass 1a: insert all entries (build ID map first so links resolve)
        for source_type in ("classes", "theorems", "conjectures", "problems"):
            records = self._load(source_type)
            for rec in records:
                self._insert_entry(conn, source_type, rec)

        conn.commit()

        # Pass 1b: insert all links (IDs now known)
        for source_type in ("classes", "theorems", "conjectures"):
            records = self._load(source_type)
            for rec in records:
                self._insert_links(conn, source_type, rec)

        conn.commit()
        return conn

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self, source_type: str) -> list[dict]:
        path = self.raw_dir / f"{source_type}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def _entry_id(self, source_type: str, name: str) -> str:
        """Register and return a stable entry ID for (source_type, name)."""
        key = f"{source_type}::{name}"
        if key not in self._id_map:
            self._id_map[key] = _make_id(source_type, name)
        return self._id_map[key]

    def _resolve_id(self, name: str) -> Optional[str]:
        """
        Resolve a bare class/theorem name (from related/impliedby arrays)
        to an entry_id in this bundle. Returns None if not found.
        """
        for source_type in ("classes", "theorems", "conjectures", "problems"):
            key = f"{source_type}::{name}"
            if key in self._id_map:
                return self._id_map[key]
        return None

    def _insert_entry(self, conn: sqlite3.Connection, source_type: str, rec: dict) -> None:
        name  = rec["name"]
        eid   = self._entry_id(source_type, name)
        etype = ENTRY_TYPE[source_type]

        # ── entries row ──────────────────────────────────────────────────────
        conn.execute(
            """INSERT OR IGNORE INTO entries
               (id, title, entry_type, source_type, domain, status, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                eid, name, etype, source_type,
                "computer science",
                STATUS[source_type],
                CONFIDENCE[source_type],
            ),
        )

        # ── sections ─────────────────────────────────────────────────────────
        sections = []

        # What It Claims — from desc or description
        claim = rec.get("desc") or rec.get("description") or rec.get("content") or ""
        if claim:
            sections.append(("What It Claims", claim, 0))

        # Mathematical Form — from formal or content (for theorems/conjectures)
        formal = rec.get("formal") or (rec.get("content") if source_type in ("theorems","conjectures") else None)
        if formal and formal != claim:
            sections.append(("Mathematical Form", formal, 1))

        # Notes
        notes = rec.get("notes", "")
        if notes:
            sections.append(("Notes", notes, 2))

        for sname, content, order in sections:
            conn.execute(
                """INSERT OR IGNORE INTO sections
                   (entry_id, section_name, content, section_order)
                   VALUES (?, ?, ?, ?)""",
                (eid, sname, content, order),
            )

        # ── entry_properties ─────────────────────────────────────────────────
        properties = []

        # Class type (Language, Promise Problem, etc.)
        if "type" in rec:
            properties.append(("class_type", rec["type"]))

        # Property tags (circuit, quantum, etc.)
        for tag in rec.get("properties", []):
            properties.append(("class_property", tag))

        # Alias
        if "alias" in rec:
            properties.append(("alias", rec["alias"]))

        # Reference URL
        ref = rec.get("ref", "")
        if ref:
            properties.append(("reference_url", ref))

        # Concept tags from name (useful for embedding)
        properties.append(("concept_tags", name))

        for pname, pval in properties:
            # For repeated property names (class_property tags), make key unique
            # by appending the value
            unique_key = pname if pname != "class_property" else f"class_property_{pval}"
            conn.execute(
                """INSERT OR IGNORE INTO entry_properties
                   (entry_id, property_name, property_value)
                   VALUES (?, ?, ?)""",
                (eid, unique_key, pval),
            )

    def _insert_links(self, conn: sqlite3.Connection, source_type: str, rec: dict) -> None:
        name = rec["name"]
        eid  = self._entry_id(source_type, name)

        # Build list of (field_name, [targets]) pairs
        link_fields: list[tuple[str, list]] = []

        if source_type == "classes":
            link_fields.append(("related", rec.get("related", [])))

        elif source_type == "theorems":
            # impliedby is a single string (one theorem name), not a list
            raw_impliedby = rec.get("impliedby")
            if raw_impliedby:
                impl_list = [raw_impliedby] if isinstance(raw_impliedby, str) else raw_impliedby
                link_fields.append(("impliedby", impl_list))
            link_fields.append(("related", rec.get("related", [])))

        elif source_type == "conjectures":
            link_fields.append(("implies",     rec.get("implies", [])))
            link_fields.append(("not_implies", rec.get("not_implies", [])))

        order_counter = conn.execute("SELECT COALESCE(MAX(link_order),0)+1 FROM links").fetchone()[0]

        for field, targets in link_fields:
            if not targets:
                continue
            link_type = LINK_MAP[field]
            tier      = LINK_TIER[field]

            for target_name in targets:
                # Try to resolve to an entry in this bundle
                target_id = self._resolve_id(target_name)
                if target_id is None:
                    # Target not in this bundle — record as unresolved property
                    conn.execute(
                        """INSERT OR IGNORE INTO entry_properties
                           (entry_id, property_name, property_value)
                           VALUES (?, ?, ?)""",
                        (eid, f"unresolved_{field}", target_name),
                    )
                    continue

                # Avoid duplicates (bidirectional check)
                existing = conn.execute(
                    """SELECT COUNT(*) FROM links
                       WHERE (source_id=? AND target_id=?)
                          OR (source_id=? AND target_id=?)""",
                    (eid, target_id, target_id, eid),
                ).fetchone()[0]
                if existing:
                    continue

                conn.execute(
                    """INSERT INTO links
                       (link_type, source_id, source_label, target_id, target_label,
                        description, link_order, confidence_tier)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        link_type,
                        eid, name,
                        target_id, target_name,
                        f"ZooClasses {field}: {name} → {target_name}",
                        order_counter,
                        tier,
                    ),
                )
                order_counter += 1


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import sys
    raw_dir    = sys.argv[1] if len(sys.argv) > 1 else "data/rrp/zoo_classes/raw"
    output_db  = sys.argv[2] if len(sys.argv) > 2 else "data/rrp/zoo_classes/rrp_zoo_classes.db"

    print(f"Parsing ZooClasses from: {raw_dir}")
    print(f"Output bundle:           {output_db}")

    parser = ZooClassesParser(raw_dir)
    conn   = parser.parse(output_db)
    stats  = bundle_stats(conn)

    print("\n── Bundle stats ────────────────────────────────")
    for k, v in stats.items():
        if k == "by_type":
            print(f"  {'by_type':<30s}:")
            for t, n in v.items():
                print(f"    {t:<28s}: {n}")
        else:
            print(f"  {k:<30s}: {v}")

    # Show unresolved cross-links
    unresolved = conn.execute(
        """SELECT property_value, COUNT(*) as c
           FROM entry_properties
           WHERE property_name LIKE 'unresolved_%'
           GROUP BY property_value ORDER BY c DESC LIMIT 10"""
    ).fetchall()
    if unresolved:
        print(f"\n── Top unresolved targets (not in bundle) ──────")
        for row in unresolved:
            print(f"  {row[0]}: {row[1]} reference(s)")

    conn.close()
    print(f"\n[OK] Bundle written to: {output_db}")


if __name__ == "__main__":
    main()
