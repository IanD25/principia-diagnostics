"""
rrp_bundle.py — Creates and manages an RRP analysis bundle database.

Each RRP gets its own SQLite database that mirrors the DS Wiki schema
(same table names, same column names) so all existing diagnostic tools
(gap_analyzer, hypothesis_generator, result_validator) can be pointed
at either database without modification.

Extra tables beyond DS Wiki:
  - rrp_meta          : package-level metadata (name, source, format, ingested_at)
  - cross_universe_bridges : Pass 2 results — RRP entities connected to DS Wiki entries

The RRP db is self-contained: it IS the diagnostic artifact for that package.
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def create_rrp_bundle(db_path: str | Path, name: str, source: str, fmt: str) -> sqlite3.Connection:
    """
    Create a new RRP bundle database at db_path.
    Returns an open connection.
    name   : human-readable package name, e.g. 'ZooClasses'
    source : origin URL or file path
    fmt    : format string, e.g. 'zoo_classes_json', 'cobra_json', 'flat_json'
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _apply_schema(conn)
    _insert_meta(conn, name, source, fmt)
    return conn


def open_rrp_bundle(db_path: str | Path) -> sqlite3.Connection:
    """Open an existing RRP bundle for read/write. Applies incremental migrations."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _migrate_schema(conn)
    return conn


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    -- ── Core tables (mirror DS Wiki schema) ──────────────────────────────────

    CREATE TABLE IF NOT EXISTS entries (
        id              TEXT PRIMARY KEY,
        title           TEXT NOT NULL,
        entry_type      TEXT NOT NULL,      -- reference_law, theorem, open_question, etc.
        source_type     TEXT,               -- origin file in RRP (classes, theorems, …)
        domain          TEXT,
        status          TEXT DEFAULT 'established',
        confidence      TEXT DEFAULT 'Tier 2',
        authoring_status TEXT DEFAULT 'rrp_ingested'
    );

    CREATE TABLE IF NOT EXISTS sections (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id        TEXT NOT NULL REFERENCES entries(id),
        section_name    TEXT NOT NULL,
        content         TEXT,
        section_order   INTEGER DEFAULT 0,
        UNIQUE(entry_id, section_name)
    );

    CREATE TABLE IF NOT EXISTS links (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        link_type           TEXT NOT NULL,
        source_id           TEXT NOT NULL REFERENCES entries(id),
        source_label        TEXT,
        target_id           TEXT NOT NULL REFERENCES entries(id),
        target_label        TEXT,
        description         TEXT,
        link_order          INTEGER DEFAULT 0,
        confidence_tier     TEXT DEFAULT '1.5',
        stoichiometry_coef  REAL    -- signed coefficient for reaction-metabolite links (NULL for non-stoichiometric)
    );

    CREATE TABLE IF NOT EXISTS entry_properties (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id        TEXT NOT NULL REFERENCES entries(id),
        property_name   TEXT NOT NULL,
        property_value  TEXT,
        UNIQUE(entry_id, property_name)
    );

    -- ── RRP-specific tables ───────────────────────────────────────────────────

    CREATE TABLE IF NOT EXISTS rrp_meta (
        key             TEXT PRIMARY KEY,
        value           TEXT
    );

    CREATE TABLE IF NOT EXISTS cross_universe_bridges (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        rrp_entry_id        TEXT NOT NULL REFERENCES entries(id),
        rrp_entry_title     TEXT,
        ds_entry_id         TEXT NOT NULL,
        ds_entry_title      TEXT,
        similarity          REAL NOT NULL,
        proposed_link_type  TEXT,
        confidence_tier     TEXT,
        description         TEXT,
        created_at          TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_links_source   ON links(source_id);
    CREATE INDEX IF NOT EXISTS idx_links_target   ON links(target_id);
    CREATE INDEX IF NOT EXISTS idx_sections_entry ON sections(entry_id);
    CREATE INDEX IF NOT EXISTS idx_bridges_rrp    ON cross_universe_bridges(rrp_entry_id);
    CREATE INDEX IF NOT EXISTS idx_bridges_ds     ON cross_universe_bridges(ds_entry_id);
    """)
    conn.commit()
    _migrate_schema(conn)


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply incremental schema migrations to existing bundles (idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(links)").fetchall()}
    if "stoichiometry_coef" not in existing:
        conn.execute("ALTER TABLE links ADD COLUMN stoichiometry_coef REAL")
        conn.commit()


def _insert_meta(conn: sqlite3.Connection, name: str, source: str, fmt: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        ("package_name", name),
        ("source",       source),
        ("format",       fmt),
        ("ingested_at",  now),
        ("schema_version", "1.1"),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO rrp_meta (key, value) VALUES (?, ?)", rows
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT key, value FROM rrp_meta").fetchall()
    return {r["key"]: r["value"] for r in rows}


def bundle_stats(conn: sqlite3.Connection) -> dict:
    """Summary stats for the RRP bundle — same shape as DS Wiki coverage stats."""
    stats = {}
    for table, col in [
        ("entries", "id"),
        ("sections", "id"),
        ("links", "id"),
        ("entry_properties", "id"),
        ("cross_universe_bridges", "id"),
    ]:
        row = conn.execute(f"SELECT COUNT({col}) FROM {table}").fetchone()
        stats[f"total_{table}"] = row[0]

    rows = conn.execute(
        "SELECT entry_type, COUNT(*) FROM entries GROUP BY entry_type ORDER BY COUNT(*) DESC"
    ).fetchall()
    stats["by_type"] = {r[0]: r[1] for r in rows}

    rows = conn.execute(
        "SELECT COUNT(*) FROM entries e WHERE NOT EXISTS "
        "(SELECT 1 FROM links l WHERE l.source_id=e.id OR l.target_id=e.id)"
    ).fetchall()
    stats["isolated_entries"] = rows[0][0]

    return stats
