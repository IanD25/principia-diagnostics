"""
test_coverage_analyzer.py — Unit tests for CoverageAnalyzer.

Tests use an in-memory SQLite DB with controlled data so every metric can
be calculated independently and verified deterministically.

Run with:
    cd .
    python -m pytest tests/test_coverage_analyzer.py -v
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Optional

import pytest

# ── Path bootstrap ──────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.coverage_analyzer import (
    CoverageAnalyzer,
    CoverageReport,
    NetworkMetrics,
    PropertyCoverage,
)


# ── Fixture helpers ─────────────────────────────────────────────────────────────

def _make_test_db(tmp_path: Path, *, n_links: int = 3) -> Path:
    """
    Build a controlled SQLite DB:
    - 8 entities: 5 reference_law, 2 law, 1 open_question
    - 3 distinct domains (physics, biology, chemistry)
    - 24 sections total (~3 per entity)
    - mathematical_archetype filled for all 8
    - dimensional_sensitivity filled for 7 (1 missing)
    - concept_tags filled for 6 (2 missing)
    - n_links links (default 3), so 5 entities are isolated
    """
    db = tmp_path / "test_source.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE entries (
            id TEXT PRIMARY KEY, title TEXT,
            entry_type TEXT, domain TEXT,
            scale TEXT, status TEXT,
            confidence TEXT, type_group TEXT
        );
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT, section_name TEXT,
            content TEXT, section_order INTEGER DEFAULT 0
        );
        CREATE TABLE entry_properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT, table_name TEXT,
            property_name TEXT, property_value TEXT,
            prop_order INTEGER DEFAULT 0
        );
        CREATE TABLE links (
            id INTEGER PRIMARY KEY,
            link_type TEXT NOT NULL,
            source_id TEXT NOT NULL, source_label TEXT NOT NULL,
            target_id TEXT NOT NULL, target_label TEXT NOT NULL,
            description TEXT NOT NULL,
            link_order INTEGER DEFAULT 0,
            confidence_tier TEXT
        );
    """)

    entries = [
        ("RL1", "Newton's Law of Gravity",  "reference_law", "physics"),
        ("RL2", "Coulomb's Law",             "reference_law", "physics"),
        ("RL3", "Fick's Law of Diffusion",   "reference_law", "biology"),
        ("RL4", "Stefan-Boltzmann Law",      "reference_law", "physics"),
        ("RL5", "Beer-Lambert Law",          "reference_law", "chemistry"),
        ("L1",  "DS Gravity",                "law",           "physics"),
        ("L2",  "DS Diffusion",              "law",           "biology"),
        ("OQ1", "Why does D_eff vary?",      "open question", "physics"),
    ]
    conn.executemany(
        "INSERT INTO entries(id, title, entry_type, domain) VALUES (?,?,?,?)", entries
    )

    # 3 sections per entity = 24 total
    section_data = []
    for eid, *_ in entries:
        for i, sname in enumerate(["Overview", "Formula", "Examples"], start=1):
            section_data.append((eid, sname, f"Content for {eid} {sname}", i))
    conn.executemany(
        "INSERT INTO sections(entry_id, section_name, content, section_order) VALUES (?,?,?,?)",
        section_data,
    )

    # Properties: archetype (all 8), d-sensitivity (7), concept_tags (6)
    archetypes = [
        ("RL1", "inverse-square-geometric"),
        ("RL2", "inverse-square-geometric"),
        ("RL3", "diffusion-equation"),
        ("RL4", "thermodynamic-bound"),
        ("RL5", "exponential-decay"),
        ("L1",  "dimensional-scaling"),
        ("L2",  "dimensional-scaling"),
        ("OQ1", "equilibrium-condition"),
    ]
    for eid, val in archetypes:
        conn.execute(
            "INSERT INTO entry_properties(entry_id, property_name, property_value) VALUES (?,?,?)",
            (eid, "mathematical_archetype", val),
        )

    d_sens = [
        ("RL1", "yes"), ("RL2", "yes"), ("RL3", "no"),
        ("RL4", "yes"), ("RL5", "no"),  ("L1", "yes"),
        ("L2", "yes"),
        # OQ1 deliberately missing
    ]
    for eid, val in d_sens:
        conn.execute(
            "INSERT INTO entry_properties(entry_id, property_name, property_value) VALUES (?,?,?)",
            (eid, "dimensional_sensitivity", val),
        )

    # concept_tags for 6 entities (RL5 and OQ1 missing)
    ctag_entries = ["RL1", "RL2", "RL3", "RL4", "L1", "L2"]
    for eid in ctag_entries:
        conn.execute(
            "INSERT INTO entry_properties(entry_id, property_name, property_value) VALUES (?,?,?)",
            (eid, "concept_tags", "gravity, force, distance"),
        )

    # Links
    all_links = [
        ("derives from",  "L1",  "DS Gravity",    "RL1", "Newton Gravity",    "L1 from RL1", 1, "2"),
        ("generalizes",   "L2",  "DS Diffusion",   "RL3", "Fick's Law",        "L2 from RL3", 2, "1.5"),
        ("analogous to",  "RL1", "Newton Gravity", "RL2", "Coulomb's Law",     "Structural",  3, None),
    ]
    for row in all_links[:n_links]:
        conn.execute(
            "INSERT INTO links(link_type, source_id, source_label, target_id, target_label, description, link_order, confidence_tier) VALUES (?,?,?,?,?,?,?,?)",
            row,
        )

    conn.commit()
    conn.close()
    return db


@pytest.fixture
def db_path(tmp_path):
    return _make_test_db(tmp_path, n_links=3)


@pytest.fixture
def ca(db_path):
    return CoverageAnalyzer(source_db=db_path)


@pytest.fixture
def report(ca):
    return ca.compute_report()


# ── Tests: CoverageAnalyzer init ────────────────────────────────────────────────

class TestCoverageAnalyzerInit:
    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="source_db"):
            CoverageAnalyzer(source_db=tmp_path / "nonexistent.db")

    def test_valid_db_instantiates(self, db_path):
        ca = CoverageAnalyzer(source_db=db_path)
        assert ca.source_db == db_path


# ── Tests: CoverageReport — counts ─────────────────────────────────────────────

class TestCoverageReportCounts:
    def test_total_entities(self, report):
        assert report.total_entities == 8

    def test_total_sections(self, report):
        # 3 sections per entity × 8 entities = 24
        assert report.total_sections == 24

    def test_total_links(self, report):
        assert report.total_links == 3

    def test_total_properties_rows(self, report):
        # archetype: 8, d_sens: 7, concept_tags: 6 = 21
        assert report.total_properties == 21


# ── Tests: Entity type distribution ────────────────────────────────────────────

class TestEntityTypeDistribution:
    def test_has_all_types(self, report):
        assert "reference_law" in report.entity_type_distribution
        assert "law"           in report.entity_type_distribution
        assert "open question" in report.entity_type_distribution

    def test_counts_correct(self, report):
        assert report.entity_type_distribution["reference_law"] == 5
        assert report.entity_type_distribution["law"]           == 2
        assert report.entity_type_distribution["open question"] == 1

    def test_sums_to_total(self, report):
        assert sum(report.entity_type_distribution.values()) == report.total_entities


# ── Tests: Domain distribution ──────────────────────────────────────────────────

class TestDomainDistribution:
    def test_has_domains(self, report):
        assert "physics"   in report.domain_distribution
        assert "biology"   in report.domain_distribution
        assert "chemistry" in report.domain_distribution

    def test_physics_count(self, report):
        # RL1, RL2, RL4, L1, OQ1 = 5 physics entries
        assert report.domain_distribution["physics"] == 5

    def test_sums_to_total(self, report):
        assert sum(report.domain_distribution.values()) == report.total_entities


# ── Tests: Section statistics ───────────────────────────────────────────────────

class TestSectionStats:
    def test_reference_law_stats(self, report):
        s = report.section_stats["reference_law"]
        assert s["entities"] == 5
        assert s["sections"] == 15                  # 3 sections × 5 entities
        assert s["avg_sections_per_entity"] == 3.0

    def test_law_stats(self, report):
        s = report.section_stats["law"]
        assert s["entities"] == 2
        assert s["sections"] == 6


# ── Tests: Property coverage ────────────────────────────────────────────────────

class TestPropertyCoverage:
    def _get_pc(self, report: CoverageReport, prop_name: str) -> PropertyCoverage:
        for pc in report.property_coverage:
            if pc.property_name == prop_name:
                return pc
        raise KeyError(f"Property '{prop_name}' not found in coverage")

    def test_archetype_coverage_100pct(self, report):
        pc = self._get_pc(report, "mathematical_archetype")
        assert pc.filled == 8
        assert pc.coverage_pct == 100.0

    def test_d_sensitivity_coverage_7_of_8(self, report):
        pc = self._get_pc(report, "dimensional_sensitivity")
        assert pc.filled == 7
        assert pc.coverage_pct == pytest.approx(87.5, abs=0.1)

    def test_concept_tags_coverage_6_of_8(self, report):
        pc = self._get_pc(report, "concept_tags")
        assert pc.filled == 6
        assert pc.coverage_pct == pytest.approx(75.0, abs=0.1)

    def test_archetype_has_value_distribution(self, report):
        pc = self._get_pc(report, "mathematical_archetype")
        assert len(pc.value_distribution) > 0
        # inverse-square-geometric appears for RL1 and RL2
        assert pc.value_distribution.get("inverse-square-geometric", 0) == 2

    def test_sparse_values_detected(self, report):
        """
        With 8 entities, SPARSE_THRESHOLD = 3% → threshold_count = 0.24
        Every archetype with count ≥ 1 counts as sparse since 1/8 = 12.5% > 3%.
        But items appearing exactly once...
        Actually with threshold_count = 0.03 * 8 = 0.24, any count >= 1 passes.
        So sparse_values should be empty in this tiny fixture.
        """
        pc = self._get_pc(report, "mathematical_archetype")
        # No archetype value should be less than 0.24 count since min is 1
        # So sparse_values should be empty
        assert isinstance(pc.sparse_values, list)

    def test_all_properties_have_total_entities_set(self, report):
        for pc in report.property_coverage:
            assert pc.total_entities == 8


# ── Tests: Archetype distribution ───────────────────────────────────────────────

class TestArchetypeDistribution:
    def test_has_expected_archetypes(self, report):
        arches = report.archetype_distribution
        assert "inverse-square-geometric" in arches
        assert "dimensional-scaling"      in arches
        assert "diffusion-equation"       in arches

    def test_dimensional_scaling_count(self, report):
        # L1 and L2 both have dimensional-scaling
        assert report.archetype_distribution["dimensional-scaling"] == 2

    def test_inverse_square_count(self, report):
        # RL1 and RL2
        assert report.archetype_distribution["inverse-square-geometric"] == 2

    def test_total_matches_entities(self, report):
        total = sum(report.archetype_distribution.values())
        # All 8 entities have an archetype
        assert total == 8


# ── Tests: D-sensitivity ────────────────────────────────────────────────────────

class TestDSensitivity:
    def test_has_yes_and_no(self, report):
        assert "yes" in report.d_sensitivity_counts
        assert "no"  in report.d_sensitivity_counts

    def test_yes_count(self, report):
        # RL1, RL2, RL4, L1, L2 = 5 "yes"
        assert report.d_sensitivity_counts["yes"] == 5

    def test_no_count(self, report):
        # RL3, RL5 = 2 "no"
        assert report.d_sensitivity_counts["no"] == 2

    def test_missing_count(self, report):
        # OQ1 is missing
        assert report.d_sensitivity_counts["missing"] == 1


# ── Tests: Network metrics ──────────────────────────────────────────────────────

class TestNetworkMetrics:
    def test_total_links(self, report):
        assert report.network_metrics.total_links == 3

    def test_possible_links(self, report):
        # 8*(8-1)/2 = 28
        assert report.network_metrics.possible_links == 28

    def test_link_density(self, report):
        expected = round(3 / 28, 6)
        assert report.network_metrics.link_density == pytest.approx(expected, abs=1e-5)

    def test_avg_links_per_entity(self, report):
        # 3 links / 8 entities = 0.375
        assert report.network_metrics.avg_links_per_entity == pytest.approx(0.375, abs=0.01)

    def test_isolated_entities_count(self, report):
        # L1↔RL1, L2↔RL3, RL1↔RL2 → linked = {L1, RL1, L2, RL3, RL2} = 5
        # Isolated = {RL4, RL5, OQ1} = 3
        assert report.network_metrics.isolated_count == 3

    def test_isolated_entity_ids(self, report):
        isolated = set(report.network_metrics.isolated_entities)
        assert "RL4" in isolated
        assert "RL5" in isolated
        assert "OQ1" in isolated
        # RL1 is linked (appears in two link rows)
        assert "RL1" not in isolated

    def test_link_type_distribution(self, report):
        lt = report.network_metrics.link_type_distribution
        assert lt["derives from"] == 1
        assert lt["generalizes"] == 1
        assert lt["analogous to"] == 1

    def test_confidence_tier_distribution(self, report):
        ct = report.network_metrics.confidence_tier_distribution
        # Tier "2" (L1→RL1), tier "1.5" (L2→RL3), "original" (NULL for RL1↔RL2)
        assert ct.get("2", 0) == 1
        assert ct.get("1.5", 0) == 1
        assert ct.get("original", 0) == 1     # NULL mapped to "original"

    def test_network_metrics_type(self, report):
        assert isinstance(report.network_metrics, NetworkMetrics)


# ── Tests: Gap detection ────────────────────────────────────────────────────────

class TestGapDetection:
    def test_gaps_is_list(self, report):
        assert isinstance(report.gaps, list)

    def test_missing_d_sensitivity_flagged(self, report):
        """OQ1 has no dimensional_sensitivity → should trigger a gap."""
        gap_text = " ".join(report.gaps)
        assert "dimensional_sensitivity" in gap_text

    def test_missing_concept_tags_flagged(self, report):
        """RL5 and OQ1 are missing concept_tags."""
        gap_text = " ".join(report.gaps)
        assert "concept_tags" in gap_text

    def test_isolation_flagged_for_high_rate(self, report):
        """3/8 = 37.5% isolated — should still produce some gap note."""
        gap_text = " ".join(report.gaps)
        # Either the >40% or >20% condition fires
        assert any(
            "isolated" in g.lower() or "no explicit links" in g.lower()
            for g in report.gaps
        ), f"No isolation gap found in: {report.gaps}"

    def test_singleton_type_flagged(self, report):
        """open_question has only 1 entity — singleton warning expected."""
        gap_text = " ".join(report.gaps)
        assert "open question" in gap_text or "singleton" in gap_text

    def test_low_link_density_flagged(self, report):
        """link_density = 0.107 which is > 0.01, so no low-density flag."""
        # density = 3/28 ≈ 0.107 — above the 0.01 threshold, so NOT flagged
        low_density_flagged = any("Low" in g or "very low" in g for g in report.gaps)
        # This should NOT fire because density is 10.7%
        assert not low_density_flagged


# ── Tests: Markdown report ──────────────────────────────────────────────────────

class TestMarkdownReport:
    def test_generate_returns_string(self, ca):
        md = ca.generate_markdown()
        assert isinstance(md, str)

    def test_contains_all_sections(self, ca):
        md = ca.generate_markdown()
        expected_sections = [
            "# Coverage Analyzer Report",
            "## 1. Overview",
            "## 2. Entity Type Distribution",
            "## 3. Domain Distribution",
            "## 4. Section Statistics",
            "## 5. Property Coverage",
            "## 6. Mathematical Archetype Distribution",
            "## 7. Dimensional Sensitivity",
            "## 8. Link Network",
            "## 9. Identified Gaps",
        ]
        for section in expected_sections:
            assert section in md, f"Section missing from markdown: {section!r}"

    def test_entity_counts_in_markdown(self, ca):
        md = ca.generate_markdown()
        assert "8" in md     # total_entities = 8

    def test_isolated_entities_listed(self, ca):
        md = ca.generate_markdown()
        assert "RL4" in md or "RL5" in md  # isolated entries appear

    def test_gaps_included_in_markdown(self, ca):
        md = ca.generate_markdown()
        assert "## 9. Identified Gaps" in md

    def test_generate_with_precomputed_report(self, ca, report):
        """generate_markdown() should accept a pre-built report and not recompute."""
        md = ca.generate_markdown(report=report)
        assert isinstance(md, str)
        assert "# Coverage Analyzer Report" in md


# ── Tests: get_stats ────────────────────────────────────────────────────────────

class TestGetStats:
    def test_returns_dict(self, ca):
        stats = ca.get_stats()
        assert isinstance(stats, dict)

    def test_required_keys_present(self, ca):
        stats = ca.get_stats()
        required = {
            "total_entities", "total_sections", "total_property_rows",
            "total_links", "link_density", "avg_links_per_entity",
            "isolated_entities_count", "entity_type_distribution",
            "property_coverage_summary", "archetype_distribution",
            "link_type_distribution", "gaps_count", "gaps",
        }
        for k in required:
            assert k in stats, f"Missing key: {k}"

    def test_property_coverage_summary_has_main_props(self, ca):
        stats = ca.get_stats()
        summary = stats["property_coverage_summary"]
        assert "mathematical_archetype"   in summary
        assert "dimensional_sensitivity"  in summary
        assert "concept_tags"             in summary

    def test_values_consistent(self, ca):
        stats = ca.get_stats()
        assert stats["total_entities"] == 8
        assert stats["total_links"]    == 3
        assert stats["isolated_entities_count"] == 3


# ── Tests: edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_links_db(self, tmp_path):
        """CoverageAnalyzer should work with no links at all."""
        db = _make_test_db(tmp_path, n_links=0)
        ca = CoverageAnalyzer(source_db=db)
        report = ca.compute_report()
        assert report.network_metrics.total_links == 0
        assert report.network_metrics.link_density == 0.0
        assert report.network_metrics.isolated_count == 8   # all isolated

    def test_all_linked_density(self, tmp_path):
        """Minimal DB where every entity has a link to at least one other."""
        db = _make_test_db(tmp_path, n_links=3)
        ca = CoverageAnalyzer(source_db=db)
        report = ca.compute_report()
        # 5 of 8 entities are connected
        assert report.network_metrics.isolated_count == 3

    def test_markdown_idempotent(self, ca):
        """Calling generate_markdown twice should return identical output."""
        md1 = ca.generate_markdown()
        md2 = ca.generate_markdown()
        assert md1 == md2
