"""
detector.py — Detect RRP format and dataset type from paths and bundle DBs.

Two detection levels:
  1. Format detection (detect / detect_with_report):
       Inspects raw source files to determine the parser to use.
       Returns format string: zoo_classes_json, cobra_json, flat_json, etc.

  2. Dataset type classification (classify_dataset_type):
       Reads bridge fingerprint from a completed Pass-2 bundle DB.
       Returns semantic type: entity_catalog, law_catalog, metabolic_network, unknown.
       Used by the entity catalog pass to decide whether to run Pass 1.5.

Supported formats (in detection order):
  zoo_classes_json  — Timeroot/ZooClasses (classes.json + theorems.json)
  cobra_json        — COBRA/BiGG metabolic model (reactions/metabolites/genes)
  flat_json         — Generic flat JSON array (Periodic Table, etc.)
  frictionless      — Frictionless Data (datapackage.json)
  ro_crate          — RO-Crate (ro-crate-metadata.json)
  codemeta          — CodeMeta (codemeta.json)
  citation_cff      — Citation File Format (CITATION.cff)
  unknown           — Cannot determine format

Supported dataset types:
  entity_catalog    — Catalog of physical/biological entities (elements, genes, proteins)
  law_catalog       — Catalog of laws, theorems, principles (ZooClasses, DS Wiki)
  metabolic_network — Reaction/metabolite/gene network (COBRA/BiGG format)
  unknown           — Insufficient signal or Pass 2 not yet run
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional


FORMAT_MARKERS = {
    "zoo_classes_json": ["classes.json", "theorems.json"],
    "cobra_json":       None,   # detected by JSON key inspection
    "flat_json":        None,   # detected by JSON key inspection (fallback)
    "frictionless":     ["datapackage.json"],
    "ro_crate":         ["ro-crate-metadata.json"],
    "codemeta":         ["codemeta.json"],
    "citation_cff":     ["CITATION.cff"],
}


def detect(path: str | Path) -> str:
    """
    Detect the RRP format at the given path (directory or single file).
    Returns a format string from FORMAT_MARKERS or 'unknown'.
    """
    path = Path(path)

    if path.is_dir():
        return _detect_directory(path)
    elif path.is_file():
        return _detect_file(path)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def _detect_directory(directory: Path) -> str:
    files = {f.name for f in directory.iterdir()}

    # File-marker-based detection (ordered by specificity)
    for fmt, markers in FORMAT_MARKERS.items():
        if markers and all(m in files for m in markers):
            return fmt

    # Fall back to inspecting any .json file
    json_files = list(directory.glob("*.json"))
    for jf in json_files:
        fmt = _detect_file(jf)
        if fmt != "unknown":
            return fmt

    return "unknown"


def _detect_file(file: Path) -> str:
    if file.suffix not in (".json", ".JSON"):
        if file.name == "CITATION.cff":
            return "citation_cff"
        return "unknown"

    try:
        with open(file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "unknown"

    # COBRA/BiGG: top-level keys include reactions, metabolites, genes
    if isinstance(data, dict):
        keys = set(data.keys())
        if {"reactions", "metabolites", "genes"}.issubset(keys):
            return "cobra_json"
        if "@context" in data and "@graph" in data:
            return "ro_crate"
        if "name" in data and "resources" in data:
            return "frictionless"

    # Flat JSON array → generic flat_json
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return "flat_json"

    # Wrapped array: {"elements": [...]} or {"data": [...]} etc.
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                return "flat_json"

    return "unknown"


def classify_dataset_type(bundle_db: str | Path) -> dict:
    """
    Classify an RRP bundle based on its Pass 2 bridge fingerprint.
    Requires Pass 2 to have been run (cross_universe_bridges populated).

    Classification signals (all from cross_universe_bridges + entries tables):
      tier_1_5_ratio  — fraction of bridges with confidence_tier='1.5'
      mean_sim        — mean cosine similarity across all stored bridges
      source_type_count — number of distinct source_type values in entries
      max_hub_frac    — fraction of all bridges absorbed by the single top DS entry

    Rules (first match wins):
      tier_1_5_ratio < 0.01 AND mean_sim < 0.808 AND source_type_count ≤ 2
          → entity_catalog   (confidence 0.85)
      tier_1_5_ratio > 0.02 AND source_type_count ≥ 3
          → law_catalog       (confidence 0.85)
      max_hub_frac > 0.50 AND source_type_count ≤ 2
          → entity_catalog   (confidence 0.70)  # fallback: one dominant hub
      else
          → unknown           (confidence 0.50)

    Returns {
        "dataset_type": str,
        "confidence":   float,
        "signals": {
            "tier_1_5_ratio":    float,
            "mean_sim":          float,
            "source_type_count": int,
            "max_hub_frac":      float,
            "total_bridges":     int,
        }
    }
    """
    conn = sqlite3.connect(str(bundle_db))
    try:
        # Total bridges and tier-1.5 count
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN confidence_tier = '1.5' THEN 1 ELSE 0 END) as tier_1_5,
                AVG(similarity) as mean_sim
            FROM cross_universe_bridges
        """).fetchone()

        total_bridges  = row[0] or 0
        tier_1_5_count = row[1] or 0
        mean_sim       = row[2] or 0.0

        tier_1_5_ratio = tier_1_5_count / max(1, total_bridges)

        # Source type diversity
        source_type_count = conn.execute(
            "SELECT COUNT(DISTINCT source_type) FROM entries"
        ).fetchone()[0] or 0

        # Hub domination: what fraction goes to the single top DS entry?
        top_hub_row = conn.execute("""
            SELECT COUNT(*) as n
            FROM cross_universe_bridges
            GROUP BY ds_entry_id
            ORDER BY n DESC LIMIT 1
        """).fetchone()
        top_hub_count = top_hub_row[0] if top_hub_row else 0
        max_hub_frac  = top_hub_count / max(1, total_bridges)

    finally:
        conn.close()

    signals = {
        "tier_1_5_ratio":    round(tier_1_5_ratio, 4),
        "mean_sim":          round(mean_sim, 4),
        "source_type_count": source_type_count,
        "max_hub_frac":      round(max_hub_frac, 4),
        "total_bridges":     total_bridges,
    }

    if total_bridges == 0:
        return {"dataset_type": "unknown", "confidence": 0.0, "signals": signals}

    # Classification rules (first match wins)
    if tier_1_5_ratio < 0.01 and mean_sim < 0.808 and source_type_count <= 2:
        return {"dataset_type": "entity_catalog", "confidence": 0.85, "signals": signals}

    if tier_1_5_ratio > 0.02 and source_type_count >= 3:
        return {"dataset_type": "law_catalog", "confidence": 0.85, "signals": signals}

    if max_hub_frac > 0.50 and source_type_count <= 2:
        return {"dataset_type": "entity_catalog", "confidence": 0.70, "signals": signals}

    return {"dataset_type": "unknown", "confidence": 0.50, "signals": signals}


def detect_with_report(path: str | Path) -> dict:
    """Like detect(), but returns a dict with format + supporting evidence."""
    path = Path(path)
    fmt = detect(path)

    report = {"path": str(path), "format": fmt, "evidence": []}

    if path.is_dir():
        files = [f.name for f in path.iterdir()]
        report["files_found"] = sorted(files)
        if fmt in FORMAT_MARKERS and FORMAT_MARKERS[fmt]:
            report["evidence"] = [
                f"marker file present: {m}"
                for m in FORMAT_MARKERS[fmt]
                if m in files
            ]

    return report
