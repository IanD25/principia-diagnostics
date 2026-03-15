"""
ecoli_core_parser.py — Parse E. coli core metabolic model (COBRA JSON) into an RRP bundle.

Source: BiGG Models / COBRApy e_coli_core.json
File:   data/rrp/ecoli_core/raw/e_coli_core.json
Format: cobra_json

Model stats: 95 reactions, 72 metabolites, 137 genes, 2 compartments (c, e)

Architecture: ADR-001 — Reification (see ARCHITECTURE_DECISIONS.md)
  Reactions, metabolites, and genes all become entries.
  Stoichiometry becomes binary links with stoichiometry_coef.

Pass 1 — Entry + Link mapping
==============================

Entry types:
  reactions   → entry_type = 'reaction'   (source_type = 'reaction')
  metabolites → entry_type = 'metabolite' (source_type = 'metabolite_c' or 'metabolite_e')
  genes       → entry_type = 'gene'       (source_type = 'gene')

ID scheme:
  rxn_{reaction_id}    e.g. rxn_PFK, rxn_PGI
  met_{metabolite_id}  e.g. met_atp_c, met_glc__D_e
  gene_{gene_id}       e.g. gene_b1241

Link types:
  reaction --[consumes]--> metabolite   stoichiometry_coef = |coef| (substrate, coef < 0)
  reaction --[produces]--> metabolite   stoichiometry_coef = |coef| (product, coef > 0)
  reaction --[catalyzed_by]--> gene     (from gene_reaction_rule, AND-split only; OR = alternatives)
  metabolite --[same_metabolite]--> metabolite  (cytosol <-> extracellular variants, tier 1.5)

Sections per entry:
  reaction   → "What It Captures", "Stoichiometry", "Kinetics", "Genes", "Annotations"
  metabolite → "What It Captures", "Chemical Properties", "Compartment", "Annotations"
  gene       → "What It Captures", "Annotations"
"""

import json
import re
import sys
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ingestion.rrp_bundle import create_rrp_bundle, bundle_stats

# ── Default paths ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent.parent
DEFAULT_SOURCE = str(_PROJECT / "data" / "rrp" / "ecoli_core" / "raw" / "e_coli_core.json")
DEFAULT_OUTPUT = str(_PROJECT / "data" / "rrp" / "ecoli_core" / "rrp_ecoli_core.db")


# ── ID helpers ────────────────────────────────────────────────────────────────

def _rxn_id(raw: str) -> str:
    return f"rxn_{raw}"

def _met_id(raw: str) -> str:
    return f"met_{raw}"

def _gene_id(raw: str) -> str:
    return f"gene_{raw}"


# ── Section builders ──────────────────────────────────────────────────────────

def _reaction_wic(rxn: dict) -> str:
    """'What It Captures' section for a reaction."""
    name = rxn.get("name") or rxn["id"]
    subsystem = rxn.get("subsystem", "unknown")
    lb, ub = rxn.get("lower_bound", 0), rxn.get("upper_bound", 1000)
    direction = "reversible" if lb < 0 else "irreversible (forward only)"
    ec = ", ".join(rxn.get("annotation", {}).get("ec-code", [])) or "not assigned"
    return (
        f"{name} is a metabolic reaction in the {subsystem} pathway of E. coli core metabolism. "
        f"It is {direction} (bounds: {lb} to {ub}). "
        f"EC number: {ec}."
    )

def _reaction_stoich(rxn: dict) -> str:
    """Human-readable stoichiometry string."""
    mets = rxn["metabolites"]
    substrates = sorted(
        [(mid, abs(coef)) for mid, coef in mets.items() if coef < 0],
        key=lambda x: x[0]
    )
    products = sorted(
        [(mid, coef) for mid, coef in mets.items() if coef > 0],
        key=lambda x: x[0]
    )
    lhs = " + ".join(
        f"{coef:.0f} {mid}" if coef != 1.0 else mid for mid, coef in substrates
    )
    rhs = " + ".join(
        f"{coef:.0f} {mid}" if coef != 1.0 else mid for mid, coef in products
    )
    arrow = "<->" if rxn.get("lower_bound", 0) < 0 else "->"
    return f"{lhs} {arrow} {rhs}"

def _reaction_kinetics(rxn: dict) -> str:
    lb = rxn.get("lower_bound", 0)
    ub = rxn.get("upper_bound", 1000)
    rev = "Yes" if lb < 0 else "No"
    return f"Reversible: {rev}\nFlux bounds: [{lb}, {ub}] mmol/gDW/h"

def _reaction_genes(rxn: dict) -> str:
    rule = rxn.get("gene_reaction_rule", "").strip()
    if not rule:
        return "No gene association (spontaneous or unassigned)."
    return f"Gene-reaction rule: {rule}"

def _reaction_annotations(rxn: dict) -> str:
    ann = rxn.get("annotation", {})
    lines = []
    for key in ("bigg.reaction", "ec-code", "kegg.reaction", "metanetx.reaction", "rhea", "biocyc"):
        vals = ann.get(key)
        if vals:
            lines.append(f"{key}: {', '.join(str(v) for v in vals)}")
    return "\n".join(lines) if lines else "No external annotations."

def _metabolite_wic(met: dict) -> str:
    name = met.get("name") or met["id"]
    formula = met.get("formula") or "unknown"
    comp = "cytosol" if met.get("compartment") == "c" else "extracellular space"
    charge = met.get("charge")
    charge_str = f", charge {charge:+d}" if charge is not None else ""
    return (
        f"{name} is a metabolite in the E. coli core model. "
        f"Molecular formula: {formula}{charge_str}. "
        f"Located in the {comp}."
    )

def _metabolite_chem(met: dict) -> str:
    lines = []
    if met.get("formula"):
        lines.append(f"Formula: {met['formula']}")
    if met.get("charge") is not None:
        lines.append(f"Charge: {met['charge']:+d}")
    ann = met.get("annotation", {})
    for key in ("chebi", "kegg.compound", "inchi_key", "metanetx.chemical"):
        vals = ann.get(key)
        if vals:
            lines.append(f"{key}: {', '.join(str(v) for v in vals)}")
    return "\n".join(lines) if lines else "No chemical data available."

def _metabolite_compartment(met: dict) -> str:
    comp_map = {"c": "cytosol (c)", "e": "extracellular space (e)"}
    comp = met.get("compartment", "unknown")
    return f"Compartment: {comp_map.get(comp, comp)}"

def _metabolite_annotations(met: dict) -> str:
    ann = met.get("annotation", {})
    lines = []
    for key in ("bigg.metabolite", "chebi", "kegg.compound", "hmdb", "biocyc", "seed.compound"):
        vals = ann.get(key)
        if vals:
            lines.append(f"{key}: {', '.join(str(v) for v in vals)}")
    return "\n".join(lines) if lines else "No external annotations."

def _gene_wic(gene: dict) -> str:
    name = gene.get("name") or gene["id"]
    uniprot = gene.get("annotation", {}).get("uniprot", [])
    uniprot_str = f" UniProt: {', '.join(uniprot)}." if uniprot else ""
    return (
        f"{name} (locus tag: {gene['id']}) is a gene in the E. coli core metabolic model "
        f"encoding an enzyme or enzyme subunit.{uniprot_str}"
    )

def _gene_annotations(gene: dict) -> str:
    ann = gene.get("annotation", {})
    lines = []
    for key in ("ncbigene", "uniprot", "refseq_name", "ecogene", "refseq_locus_tag"):
        vals = ann.get(key)
        if vals:
            lines.append(f"{key}: {', '.join(str(v) for v in vals)}")
    return "\n".join(lines) if lines else "No external annotations."


# ── Gene-reaction rule parser ─────────────────────────────────────────────────

def _parse_gene_ids(rule: str) -> list[str]:
    """Extract all gene locus tags from a gene_reaction_rule string."""
    if not rule:
        return []
    # Gene IDs are b-numbers like b1241 or other alphanumeric locus tags
    return list(set(re.findall(r'\b[a-z]\d{4}\b', rule)))


# ── Main parse function ───────────────────────────────────────────────────────

def parse_ecoli_core(
    source_path: str = DEFAULT_SOURCE,
    output_path: str = DEFAULT_OUTPUT,
) -> dict:
    """
    Parse COBRA JSON into an RRP bundle at output_path.
    Returns bundle_stats dict.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"E. coli Core Parser")
    print(f"  Source : {source_path}")
    print(f"  Output : {output_path}")
    print()

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    reactions  = data["reactions"]
    metabolites = data["metabolites"]
    genes      = data["genes"]

    print(f"  Loaded: {len(reactions)} reactions, {len(metabolites)} metabolites, {len(genes)} genes")

    conn = create_rrp_bundle(
        db_path=output_path,
        name="E. coli Core Metabolic Model",
        source=str(source_path),
        fmt="cobra_json",
    )

    # ── Pass 1a: Insert metabolite entries ────────────────────────────────────
    print("  Pass 1a — Inserting metabolite entries...")
    met_count = 0
    for met in metabolites:
        mid = _met_id(met["id"])
        comp = met.get("compartment", "c")
        src_type = f"metabolite_{comp}"

        conn.execute(
            """INSERT OR IGNORE INTO entries
               (id, title, entry_type, source_type, domain, status, confidence)
               VALUES (?, ?, 'metabolite', ?, 'biochemistry', 'established', 'Tier 1')""",
            (mid, met.get("name") or met["id"], src_type)
        )

        # Sections
        sections = [
            ("What It Captures",    _metabolite_wic(met),           1),
            ("Chemical Properties", _metabolite_chem(met),          2),
            ("Compartment",         _metabolite_compartment(met),   3),
            ("Annotations",         _metabolite_annotations(met),   4),
        ]
        conn.executemany(
            """INSERT OR IGNORE INTO sections (entry_id, section_name, content, section_order)
               VALUES (?, ?, ?, ?)""",
            [(mid, name, content, order) for name, content, order in sections if content]
        )

        # Properties
        props = [
            ("formula",    met.get("formula")),
            ("charge",     str(met["charge"]) if met.get("charge") is not None else None),
            ("compartment", met.get("compartment")),
            ("bigg_id",    met["id"]),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO entry_properties (entry_id, property_name, property_value) VALUES (?, ?, ?)",
            [(mid, k, v) for k, v in props if v is not None]
        )
        met_count += 1

    # ── Pass 1b: Insert gene entries ──────────────────────────────────────────
    print("  Pass 1b — Inserting gene entries...")
    gene_count = 0
    for gene in genes:
        gid = _gene_id(gene["id"])

        conn.execute(
            """INSERT OR IGNORE INTO entries
               (id, title, entry_type, source_type, domain, status, confidence)
               VALUES (?, ?, 'gene', 'gene', 'biochemistry', 'established', 'Tier 1')""",
            (gid, gene.get("name") or gene["id"])
        )

        sections = [
            ("What It Captures", _gene_wic(gene),          1),
            ("Annotations",      _gene_annotations(gene),  2),
        ]
        conn.executemany(
            """INSERT OR IGNORE INTO sections (entry_id, section_name, content, section_order)
               VALUES (?, ?, ?, ?)""",
            [(gid, name, content, order) for name, content, order in sections if content]
        )

        # Properties
        ann = gene.get("annotation", {})
        props = [
            ("locus_tag", gene["id"]),
            ("gene_name", gene.get("name")),
            ("uniprot",   ", ".join(ann.get("uniprot", []))),
            ("ncbigene",  ", ".join(ann.get("ncbigene", []))),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO entry_properties (entry_id, property_name, property_value) VALUES (?, ?, ?)",
            [(gid, k, v) for k, v in props if v]
        )
        gene_count += 1

    # ── Pass 1c: Insert reaction entries + stoichiometric links ───────────────
    print("  Pass 1c — Inserting reaction entries + stoichiometric links...")
    rxn_count = 0
    stoich_link_count = 0
    gene_link_count = 0

    for rxn in reactions:
        rid = _rxn_id(rxn["id"])
        subsystem = rxn.get("subsystem", "")
        is_exchange = rxn["id"].startswith("EX_")
        src_type = "exchange_reaction" if is_exchange else "reaction"

        conn.execute(
            """INSERT OR IGNORE INTO entries
               (id, title, entry_type, source_type, domain, status, confidence)
               VALUES (?, ?, 'reaction', ?, 'biochemistry', 'established', 'Tier 1')""",
            (rid, rxn.get("name") or rxn["id"], src_type)
        )

        sections = [
            ("What It Captures", _reaction_wic(rxn),         1),
            ("Stoichiometry",    _reaction_stoich(rxn),       2),
            ("Kinetics",         _reaction_kinetics(rxn),     3),
            ("Genes",            _reaction_genes(rxn),        4),
            ("Annotations",      _reaction_annotations(rxn),  5),
        ]
        conn.executemany(
            """INSERT OR IGNORE INTO sections (entry_id, section_name, content, section_order)
               VALUES (?, ?, ?, ?)""",
            [(rid, name, content, order) for name, content, order in sections if content]
        )

        # Properties
        lb, ub = rxn.get("lower_bound", 0), rxn.get("upper_bound", 1000)
        props = [
            ("bigg_id",            rxn["id"]),
            ("subsystem",          subsystem),
            ("lower_bound",        str(lb)),
            ("upper_bound",        str(ub)),
            ("reversible",         "true" if lb < 0 else "false"),
            ("gene_reaction_rule", rxn.get("gene_reaction_rule", "")),
            ("ec_code",            ", ".join(rxn.get("annotation", {}).get("ec-code", []))),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO entry_properties (entry_id, property_name, property_value) VALUES (?, ?, ?)",
            [(rid, k, v) for k, v in props if v]
        )
        rxn_count += 1

        # Stoichiometric links: reaction --[consumes/produces]--> metabolite
        for met_raw, coef in rxn["metabolites"].items():
            mid = _met_id(met_raw)
            link_type = "consumes" if coef < 0 else "produces"
            magnitude = abs(coef)
            conn.execute(
                """INSERT OR IGNORE INTO links
                   (link_type, source_id, source_label, target_id, target_label,
                    description, stoichiometry_coef, confidence_tier)
                   VALUES (?, ?, ?, ?, ?, ?, ?, '1')""",
                (
                    link_type, rid, rid, mid, mid,
                    f"{rxn['id']} {link_type} {met_raw} (coef={coef:+.1f})",
                    magnitude,
                )
            )
            stoich_link_count += 1

        # Gene-reaction links: reaction --[catalyzed_by]--> gene
        gene_ids = _parse_gene_ids(rxn.get("gene_reaction_rule", ""))
        for g_raw in gene_ids:
            gid = _gene_id(g_raw)
            conn.execute(
                """INSERT OR IGNORE INTO links
                   (link_type, source_id, source_label, target_id, target_label,
                    description, confidence_tier)
                   VALUES ('catalyzed_by', ?, ?, ?, ?, ?, '1')""",
                (
                    rid, rid, gid, gid,
                    f"{rxn['id']} catalyzed by {g_raw}",
                )
            )
            gene_link_count += 1

    # ── Pass 1d: Cross-compartment metabolite links ───────────────────────────
    # Link cytosolic metabolite to its extracellular counterpart (same base ID)
    print("  Pass 1d — Cross-compartment metabolite links...")
    met_by_base: dict[str, list[str]] = {}
    for met in metabolites:
        base = re.sub(r'_[ce]$', '', met["id"])
        met_by_base.setdefault(base, []).append(met["id"])

    xcomp_count = 0
    for base, variants in met_by_base.items():
        if len(variants) == 2:
            a, b = [_met_id(v) for v in sorted(variants)]
            conn.execute(
                """INSERT OR IGNORE INTO links
                   (link_type, source_id, source_label, target_id, target_label,
                    description, confidence_tier)
                   VALUES ('same_metabolite', ?, ?, ?, ?, ?, '1.5')""",
                (a, a, b, b, f"{variants[0]} and {variants[1]} are the same molecule in different compartments")
            )
            xcomp_count += 1

    conn.commit()

    stats = bundle_stats(conn)
    conn.close()

    print()
    print(f"  Metabolite entries : {met_count}")
    print(f"  Gene entries       : {gene_count}")
    print(f"  Reaction entries   : {rxn_count}")
    print(f"  ---------------------")
    print(f"  Total entries      : {stats['total_entries']}")
    print(f"  Stoich links       : {stoich_link_count}")
    print(f"  Gene-rxn links     : {gene_link_count}")
    print(f"  Cross-compartment  : {xcomp_count}")
    print(f"  Total links        : {stats['total_links']}")
    print(f"  Entry properties   : {stats['total_entry_properties']}")
    print(f"  Sections           : {stats['total_sections']}")
    print()
    print(f"[OK] Bundle written: {output_path}")

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse E. coli core COBRA JSON into RRP bundle")
    parser.add_argument("source", nargs="?", default=DEFAULT_SOURCE, help="Path to e_coli_core.json")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT, help="Path for output .db")
    args = parser.parse_args()

    parse_ecoli_core(args.source, args.output)
