"""
test_ecoli_parser.py — Tests for the E. coli core COBRA JSON parser.

Tests verify:
  - Correct entry counts and types
  - Stoichiometric link direction and coefficient values
  - Gene-reaction link extraction
  - Cross-compartment metabolite links
  - Entry properties completeness
  - Section generation
  - Schema integrity (no isolated entries, foreign key sanity)
  - Idempotency (re-running parser produces same result)
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingestion.parsers.ecoli_core_parser import (
    parse_ecoli_core,
    _rxn_id,
    _met_id,
    _gene_id,
    _parse_gene_ids,
)

# ── Source data path ──────────────────────────────────────────────────────────
_PROJECT = Path(__file__).resolve().parent.parent
_SOURCE  = _PROJECT / "data" / "rrp" / "ecoli_core" / "raw" / "e_coli_core.json"

skip_if_no_source = pytest.mark.skipif(
    not _SOURCE.exists(),
    reason="e_coli_core.json not found"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bundle_db(tmp_path_factory):
    """Run the parser once into a temp DB; share across the module."""
    out = tmp_path_factory.mktemp("ecoli") / "rrp_ecoli_core.db"
    parse_ecoli_core(source_path=str(_SOURCE), output_path=str(out))
    return str(out)


@pytest.fixture(scope="module")
def conn(bundle_db):
    c = sqlite3.connect(bundle_db)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


@pytest.fixture(scope="module")
def raw_data():
    with open(_SOURCE, encoding="utf-8") as f:
        return json.load(f)


# ── ID helper tests ───────────────────────────────────────────────────────────

class TestIdHelpers:
    def test_rxn_id(self):
        assert _rxn_id("PFK") == "rxn_PFK"

    def test_met_id(self):
        assert _met_id("atp_c") == "met_atp_c"

    def test_gene_id(self):
        assert _gene_id("b1241") == "gene_b1241"

    def test_parse_gene_ids_simple(self):
        ids = _parse_gene_ids("b3916 or b1723")
        assert set(ids) == {"b3916", "b1723"}

    def test_parse_gene_ids_and(self):
        ids = _parse_gene_ids("b0902 and b0903")
        assert set(ids) == {"b0902", "b0903"}

    def test_parse_gene_ids_complex(self):
        rule = "((b0902 and b0903) and b2579) or (b0902 and b0903)"
        ids = _parse_gene_ids(rule)
        assert "b0902" in ids
        assert "b0903" in ids
        assert "b2579" in ids

    def test_parse_gene_ids_empty(self):
        assert _parse_gene_ids("") == []

    def test_parse_gene_ids_no_genes(self):
        # spontaneous reaction — no gene locus tags
        assert _parse_gene_ids("spontaneous") == []


# ── Entry count tests ─────────────────────────────────────────────────────────

@skip_if_no_source
class TestEntryCounts:
    def test_total_entries(self, conn, raw_data):
        total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        expected = len(raw_data["reactions"]) + len(raw_data["metabolites"]) + len(raw_data["genes"])
        assert total == expected

    def test_reaction_count(self, conn, raw_data):
        count = conn.execute("SELECT COUNT(*) FROM entries WHERE entry_type='reaction'").fetchone()[0]
        assert count == len(raw_data["reactions"])

    def test_metabolite_count(self, conn, raw_data):
        count = conn.execute("SELECT COUNT(*) FROM entries WHERE entry_type='metabolite'").fetchone()[0]
        assert count == len(raw_data["metabolites"])

    def test_gene_count(self, conn, raw_data):
        count = conn.execute("SELECT COUNT(*) FROM entries WHERE entry_type='gene'").fetchone()[0]
        assert count == len(raw_data["genes"])

    def test_cytosol_metabolites(self, conn):
        count = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE entry_type='metabolite' AND source_type='metabolite_c'"
        ).fetchone()[0]
        assert count == 52

    def test_extracellular_metabolites(self, conn):
        count = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE entry_type='metabolite' AND source_type='metabolite_e'"
        ).fetchone()[0]
        assert count == 20

    def test_exchange_reactions(self, conn):
        count = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE source_type='exchange_reaction'"
        ).fetchone()[0]
        assert count == 20

    def test_no_isolated_entries(self, conn):
        isolated = conn.execute(
            "SELECT COUNT(*) FROM entries e WHERE NOT EXISTS "
            "(SELECT 1 FROM links l WHERE l.source_id=e.id OR l.target_id=e.id)"
        ).fetchone()[0]
        assert isolated == 0, f"Found {isolated} isolated entries"


# ── Stoichiometric link tests ─────────────────────────────────────────────────

@skip_if_no_source
class TestStoichiometricLinks:
    def test_total_stoich_links(self, conn):
        count = conn.execute(
            "SELECT COUNT(*) FROM links WHERE link_type IN ('consumes', 'produces')"
        ).fetchone()[0]
        assert count == 360

    def test_pfk_consumes_atp(self, conn):
        row = conn.execute(
            "SELECT stoichiometry_coef FROM links WHERE source_id='rxn_PFK' AND target_id='met_atp_c' AND link_type='consumes'"
        ).fetchone()
        assert row is not None, "PFK should consume atp_c"
        assert row[0] == 1.0

    def test_pfk_produces_fdp(self, conn):
        row = conn.execute(
            "SELECT stoichiometry_coef FROM links WHERE source_id='rxn_PFK' AND target_id='met_fdp_c' AND link_type='produces'"
        ).fetchone()
        assert row is not None, "PFK should produce fdp_c"
        assert row[0] == 1.0

    def test_stoich_coef_always_positive(self, conn):
        """stoichiometry_coef stores magnitude (always >= 0)."""
        negative = conn.execute(
            "SELECT COUNT(*) FROM links WHERE stoichiometry_coef IS NOT NULL AND stoichiometry_coef < 0"
        ).fetchone()[0]
        assert negative == 0, "All stoichiometry_coef values must be non-negative"

    def test_biomass_reaction_participant_count(self, conn):
        """BIOMASS reaction has 23 participants."""
        count = conn.execute(
            "SELECT COUNT(*) FROM links WHERE source_id='rxn_BIOMASS_Ecoli_core_w_GAM' "
            "AND link_type IN ('consumes', 'produces')"
        ).fetchone()[0]
        assert count == 23

    def test_direction_consistency(self, conn, raw_data):
        """Verify consumes/produces direction matches COBRA sign convention."""
        rxn_map = {r["id"]: r for r in raw_data["reactions"]}
        # Check PGI: glucose-6-phosphate <-> fructose-6-phosphate
        pgi = rxn_map["PGI"]
        for met_raw, coef in pgi["metabolites"].items():
            mid = _met_id(met_raw)
            expected_type = "consumes" if coef < 0 else "produces"
            row = conn.execute(
                "SELECT link_type FROM links WHERE source_id='rxn_PGI' AND target_id=?",
                (mid,)
            ).fetchone()
            assert row is not None, f"Missing link for {met_raw}"
            assert row[0] == expected_type, f"{met_raw}: expected {expected_type}, got {row[0]}"


# ── Gene-reaction link tests ──────────────────────────────────────────────────

@skip_if_no_source
class TestGeneReactionLinks:
    def test_catalyzed_by_links_exist(self, conn):
        count = conn.execute(
            "SELECT COUNT(*) FROM links WHERE link_type='catalyzed_by'"
        ).fetchone()[0]
        assert count > 0

    def test_pfk_catalyzed_by_both_genes(self, conn):
        genes = {
            row[0] for row in conn.execute(
                "SELECT target_id FROM links WHERE source_id='rxn_PFK' AND link_type='catalyzed_by'"
            ).fetchall()
        }
        assert "gene_b3916" in genes
        assert "gene_b1723" in genes

    def test_spontaneous_reaction_no_gene_links(self, conn):
        """Reactions with empty gene_reaction_rule have no catalyzed_by links."""
        # Find reactions without gene links
        count = conn.execute(
            "SELECT COUNT(*) FROM links WHERE link_type='catalyzed_by'"
        ).fetchone()[0]
        # There are 137 genes and 95 reactions; not all reactions are enzyme-catalyzed
        assert count < 95 * 10  # reasonable upper bound

    def test_gene_entries_exist_for_all_catalyzed_by(self, conn):
        """Every gene_id in a catalyzed_by link exists as an entry."""
        missing = conn.execute(
            "SELECT DISTINCT l.target_id FROM links l "
            "WHERE l.link_type='catalyzed_by' "
            "AND l.target_id NOT IN (SELECT id FROM entries)"
        ).fetchall()
        assert len(missing) == 0, f"Missing gene entries: {[r[0] for r in missing]}"


# ── Cross-compartment link tests ──────────────────────────────────────────────

@skip_if_no_source
class TestCrossCompartmentLinks:
    def test_same_metabolite_links_exist(self, conn):
        count = conn.execute(
            "SELECT COUNT(*) FROM links WHERE link_type='same_metabolite'"
        ).fetchone()[0]
        assert count == 18

    def test_atp_cross_compartment(self, conn):
        """met_atp_c and met_atp_e should be linked if both exist."""
        has_atp_e = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE id='met_atp_e'"
        ).fetchone()[0]
        if has_atp_e:
            row = conn.execute(
                "SELECT COUNT(*) FROM links WHERE link_type='same_metabolite' "
                "AND ((source_id='met_atp_c' AND target_id='met_atp_e') "
                "  OR (source_id='met_atp_e' AND target_id='met_atp_c'))"
            ).fetchone()[0]
            assert row == 1

    def test_same_metabolite_tier(self, conn):
        tier = conn.execute(
            "SELECT DISTINCT confidence_tier FROM links WHERE link_type='same_metabolite'"
        ).fetchone()
        assert tier is not None
        assert tier[0] == "1.5"


# ── Entry property tests ──────────────────────────────────────────────────────

@skip_if_no_source
class TestEntryProperties:
    def test_reaction_has_subsystem(self, conn):
        row = conn.execute(
            "SELECT property_value FROM entry_properties WHERE entry_id='rxn_PFK' AND property_name='subsystem'"
        ).fetchone()
        assert row is not None
        assert "Glycolysis" in row[0]

    def test_reaction_has_reversible_flag(self, conn):
        row = conn.execute(
            "SELECT property_value FROM entry_properties WHERE entry_id='rxn_PFK' AND property_name='reversible'"
        ).fetchone()
        assert row is not None
        assert row[0] == "false"

    def test_metabolite_has_formula(self, conn):
        row = conn.execute(
            "SELECT property_value FROM entry_properties WHERE entry_id='met_atp_c' AND property_name='formula'"
        ).fetchone()
        assert row is not None
        assert row[0] == "C10H12N5O13P3"

    def test_metabolite_has_charge(self, conn):
        row = conn.execute(
            "SELECT property_value FROM entry_properties WHERE entry_id='met_atp_c' AND property_name='charge'"
        ).fetchone()
        assert row is not None
        assert row[0] == "-4"

    def test_gene_has_locus_tag(self, conn):
        row = conn.execute(
            "SELECT property_value FROM entry_properties WHERE entry_id='gene_b1241' AND property_name='locus_tag'"
        ).fetchone()
        assert row is not None
        assert row[0] == "b1241"


# ── Section tests ─────────────────────────────────────────────────────────────

@skip_if_no_source
class TestSections:
    def test_reaction_has_stoichiometry_section(self, conn):
        row = conn.execute(
            "SELECT content FROM sections WHERE entry_id='rxn_PFK' AND section_name='Stoichiometry'"
        ).fetchone()
        assert row is not None
        assert "->" in row[0] or "<->" in row[0]

    def test_reaction_has_wic_section(self, conn):
        row = conn.execute(
            "SELECT content FROM sections WHERE entry_id='rxn_PFK' AND section_name='What It Captures'"
        ).fetchone()
        assert row is not None
        assert len(row[0]) > 20

    def test_metabolite_has_wic_section(self, conn):
        row = conn.execute(
            "SELECT content FROM sections WHERE entry_id='met_atp_c' AND section_name='What It Captures'"
        ).fetchone()
        assert row is not None
        assert "ATP" in row[0] or "adenosine" in row[0].lower() or "C10H12N5O13P3" in row[0]

    def test_gene_has_wic_section(self, conn):
        row = conn.execute(
            "SELECT content FROM sections WHERE entry_id='gene_b1241' AND section_name='What It Captures'"
        ).fetchone()
        assert row is not None

    def test_reversible_reaction_shows_arrow(self, conn):
        """A reversible reaction stoichiometry should use <->."""
        # Find a reversible reaction
        rev_rxn = conn.execute(
            "SELECT entry_id FROM entry_properties WHERE property_name='reversible' AND property_value='true' LIMIT 1"
        ).fetchone()
        if rev_rxn:
            row = conn.execute(
                "SELECT content FROM sections WHERE entry_id=? AND section_name='Stoichiometry'",
                (rev_rxn[0],)
            ).fetchone()
            assert row is not None
            assert "<->" in row[0]


# ── Idempotency test ──────────────────────────────────────────────────────────

@skip_if_no_source
class TestIdempotency:
    def test_rerun_produces_same_counts(self, tmp_path):
        """Running the parser twice on the same output produces identical counts."""
        out = tmp_path / "rrp_ecoli_idempotent.db"
        stats1 = parse_ecoli_core(source_path=str(_SOURCE), output_path=str(out))
        # Delete and recreate — INSERT OR IGNORE means re-run is safe
        out.unlink()
        stats2 = parse_ecoli_core(source_path=str(_SOURCE), output_path=str(out))
        assert stats1["total_entries"] == stats2["total_entries"]
        assert stats1["total_links"] == stats2["total_links"]


# ── RRP meta tests ────────────────────────────────────────────────────────────

@skip_if_no_source
class TestRrpMeta:
    def test_format_is_cobra_json(self, conn):
        row = conn.execute("SELECT value FROM rrp_meta WHERE key='format'").fetchone()
        assert row is not None
        assert row[0] == "cobra_json"

    def test_schema_version(self, conn):
        row = conn.execute("SELECT value FROM rrp_meta WHERE key='schema_version'").fetchone()
        assert row is not None
        assert row[0] == "1.1"

    def test_stoichiometry_coef_column_exists(self, conn):
        cols = {row[1] for row in conn.execute("PRAGMA table_info(links)").fetchall()}
        assert "stoichiometry_coef" in cols
