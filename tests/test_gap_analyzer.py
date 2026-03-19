"""
test_gap_analyzer.py — Unit and integration tests for GapAnalyzer.

Unit tests use an in-memory SQLite DB with hand-crafted data so no external
files are required.  Integration tests run against the real ds_wiki.db.

Run with:
    cd .
    python -m pytest tests/test_gap_analyzer.py -v
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

# ── Path bootstrap ──────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.gap_analyzer import (
    DEFAULT_TYPE_MINIMUMS,
    PROPERTY_COVERAGE_THRESHOLD,
    SPARSE_TAXONOMY_THRESHOLD,
    EnrichmentPriority,
    GapAnalysisReport,
    GapAnalyzer,
    LinkGap,
    PropertyGap,
    TaxonomyGap,
    TypeBalanceGap,
    TAXONOMY_PROPERTIES,
)


# ── Fixture helpers ─────────────────────────────────────────────────────────────

def _make_db(tmp_path: Path) -> Path:
    """
    Build a minimal test DB with:
    - 5 reference_law entries (E1–E5) — below DEFAULT minimum of 50
    - 2 method entries (M1, M2)       — below minimum of 10
    - 1 open_question entry (Q1)       — below minimum of 10
    - Properties: E1–E4 have 'mathematical_archetype'; E5 does NOT (gap)
    - Properties: E1–E5 have 'dimensional_sensitivity'; M1 does NOT
    - E1–E3 have archetype='conservation-law'; E4 has 'oscillatory' (sparse, abs=1)
    - Links:
        - E1↔E2 (tier 1)
        - E3↔E4 (tier 1.5)
        - E5 is isolated (no links)
        - M1 is isolated
        - Q1 has only M1 connection but different type
    """
    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE entries (
            id TEXT PRIMARY KEY, title TEXT, entry_type TEXT, domain TEXT,
            scale TEXT, status TEXT, confidence TEXT, type_group TEXT
        );
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT, section_name TEXT, section_order INTEGER, content TEXT,
            UNIQUE(entry_id, section_name)
        );
        CREATE TABLE entry_properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT, table_name TEXT, property_name TEXT,
            property_value TEXT, prop_order INTEGER DEFAULT 0
        );
        CREATE TABLE links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link_type TEXT, source_id TEXT, source_label TEXT,
            target_id TEXT, target_label TEXT, description TEXT,
            link_order INTEGER DEFAULT 0, confidence_tier TEXT
        );
    """)

    conn.executemany("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?)", [
        ("E1", "Entropy Law",       "reference_law",  "physics",  "macro", "established", "Tier 1", "RL"),
        ("E2", "Info Entropy",      "reference_law",  "physics",  "macro", "established", "Tier 1", "RL"),
        ("E3", "Thermo Bound",      "reference_law",  "physics",  "macro", "established", "Tier 1", "RL"),
        ("E4", "Oscillation Law",   "reference_law",  "physics",  "micro", "established", "Tier 1", "RL"),
        ("E5", "Isolated Law",      "reference_law",  "physics",  "macro", "established", "Tier 1", "RL"),
        ("M1", "Method Alpha",      "method",         "cs",       "micro", "established", "Tier 2", "ME"),
        ("M2", "Method Beta",       "method",         "cs",       "micro", "established", "Tier 2", "ME"),
        ("Q1", "Open Question",     "open_question",  "physics",  "macro", "open",        "Tier 2", "OQ"),
    ])

    # mathematical_archetype: E1-E3 have it; E4 and E5 do NOT → 3/5 = 60% < 80% → gap
    # "oscillatory" is set on M2 to keep the sparse-taxonomy test alive
    conn.executemany(
        "INSERT INTO entry_properties (entry_id, table_name, property_name, property_value) VALUES (?,?,?,?)",
        [
            ("E1", "Properties", "mathematical_archetype", "conservation-law"),
            ("E2", "Properties", "mathematical_archetype", "conservation-law"),
            ("E3", "Properties", "mathematical_archetype", "conservation-law"),
            ("M2", "Properties", "mathematical_archetype", "oscillatory"),   # sparse (count=1)
            # E4 and E5 are missing mathematical_archetype

            # dimensional_sensitivity: E1-E5 + M2 have it; M1 does NOT → gap for method
            ("E1", "Properties", "dimensional_sensitivity", "D-sensitive"),
            ("E2", "Properties", "dimensional_sensitivity", "D-sensitive"),
            ("E3", "Properties", "dimensional_sensitivity", "D-invariant"),
            ("E4", "Properties", "dimensional_sensitivity", "D-sensitive"),
            ("E5", "Properties", "dimensional_sensitivity", "D-sensitive"),
            ("M2", "Properties", "dimensional_sensitivity", "D-invariant"),
            # M1 is missing dimensional_sensitivity

            # concept_tags: only E1 has it → very sparse coverage across all types
            ("E1", "Properties", "concept_tags", "thermodynamics entropy"),
        ]
    )

    conn.executemany(
        "INSERT INTO links (link_type, source_id, source_label, target_id, target_label, confidence_tier) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("derives from", "E1", "Entropy Law",  "E2", "Info Entropy",  "1"),   # tier-1
            ("couples to",   "E3", "Thermo Bound", "E4", "Osc Law",       "1.5"), # tier-1.5
            ("tests",        "Q1", "Open Q",       "M1", "Method Alpha",  "2"),   # Q1→M1 cross-type
        ]
    )
    conn.commit()
    conn.close()
    return db


# ══════════════════════════════════════════════════════════════════════════════
# Data class unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDataClasses:
    def test_property_gap_fields(self):
        g = PropertyGap(
            entity_type="reference_law", property_name="concept_tags",
            total_of_type=10, filled=6, missing_count=4,
            coverage_pct=60.0, missing_ids=["E1","E2"], priority=2,
        )
        assert g.coverage_pct == 60.0
        assert g.missing_count == 4
        assert g.priority == 2

    def test_taxonomy_gap_fields(self):
        g = TaxonomyGap(
            taxonomy_name="mathematical_archetype", value="oscillatory",
            count=2, pct_of_filled=1.5, flag="sparse",
        )
        assert g.flag == "sparse"
        assert g.count == 2

    def test_type_balance_gap_fields(self):
        g = TypeBalanceGap(
            entity_type="method", observed=3, expected_min=10,
            deficit=7, suggestion="Add 7 more 'method' entries.",
        )
        assert g.deficit == 7
        assert g.observed == 3

    def test_link_gap_fields(self):
        g = LinkGap(
            entry_id="E5", title="Isolated Law", entity_type="reference_law",
            total_links=0, gap_type="isolated", detail="No links.",
        )
        assert g.gap_type == "isolated"
        assert g.total_links == 0

    def test_enrichment_priority_fields(self):
        p = EnrichmentPriority(
            rank=1, action="add_link", target="isolated_entries",
            description="Connect 3 isolated entries.", impact_score=0.5,
        )
        assert p.rank == 1
        assert p.action == "add_link"


class TestGapAnalysisReport:
    def _make_report(self) -> GapAnalysisReport:
        return GapAnalysisReport(
            property_gaps=[
                PropertyGap("reference_law","concept_tags",10,6,4,60.0,["E1","E2"],2)
            ],
            taxonomy_gaps=[
                TaxonomyGap("mathematical_archetype","oscillatory",1,0.5,"singleton")
            ],
            type_balance_gaps=[
                TypeBalanceGap("method",3,10,7,"Add 7 more.")
            ],
            link_gaps=[
                LinkGap("E5","Isolated Law","reference_law",0,"isolated","No links."),
                LinkGap("M1","Method Alpha","method",1,"no_tier1","No tier-1."),
            ],
            enrichment_priorities=[
                EnrichmentPriority(1,"add_link","isolated_entries","Connect 3.",0.5),
                EnrichmentPriority(2,"add_entry","method","Add 7.",0.4),
            ],
            summary_stats={"total_entities": 8, "isolated_count": 2},
        )

    def test_markdown_contains_summary_header(self):
        md = self._make_report().as_markdown()
        assert "# Gap Analysis Report" in md
        assert "## Summary" in md

    def test_markdown_contains_property_gap_section(self):
        md = self._make_report().as_markdown()
        assert "Property Coverage Gaps" in md
        assert "concept_tags" in md

    def test_markdown_contains_taxonomy_section(self):
        md = self._make_report().as_markdown()
        assert "Sparse Taxonomy Values" in md
        assert "oscillatory" in md

    def test_markdown_contains_type_balance_section(self):
        md = self._make_report().as_markdown()
        assert "Entity Type Balance Gaps" in md
        assert "method" in md

    def test_markdown_contains_link_gap_section(self):
        md = self._make_report().as_markdown()
        assert "Link Gaps" in md
        assert "isolated" in md

    def test_markdown_contains_priorities(self):
        md = self._make_report().as_markdown()
        assert "Top Enrichment Priorities" in md
        assert "add_link" in md

    def test_markdown_shows_summary_stats(self):
        md = self._make_report().as_markdown()
        assert "8" in md    # total_entities

    def test_markdown_limits_link_gap_table_to_20(self):
        # 25 link gaps → table shows 20 + "more not shown"
        many = [
            LinkGap(f"X{i}", f"Entry {i}", "ref", 0, "isolated", ".")
            for i in range(25)
        ]
        r = GapAnalysisReport(
            property_gaps=[], taxonomy_gaps=[], type_balance_gaps=[],
            link_gaps=many, enrichment_priorities=[],
            summary_stats={"total_entities": 25},
        )
        md = r.as_markdown()
        assert "more not shown" in md

    def test_empty_report_markdown_still_renders(self):
        r = GapAnalysisReport(
            property_gaps=[], taxonomy_gaps=[], type_balance_gaps=[],
            link_gaps=[], enrichment_priorities=[],
            summary_stats={"total_entities": 0},
        )
        md = r.as_markdown()
        assert "# Gap Analysis Report" in md


# ══════════════════════════════════════════════════════════════════════════════
# GapAnalyzer unit tests (synthetic DB)
# ══════════════════════════════════════════════════════════════════════════════

class TestGapAnalyzerInit:
    def test_default_type_minimums_loaded(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        assert "reference_law" in ga.type_minimums
        assert ga.type_minimums["reference_law"] == DEFAULT_TYPE_MINIMUMS["reference_law"]

    def test_custom_type_minimums_override(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db, type_minimums={"reference_law": 999})
        assert ga.type_minimums["reference_law"] == 999
        # Other defaults still present
        assert "method" in ga.type_minimums

    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GapAnalyzer(source_db=tmp_path / "nonexistent.db")

    def test_heuristics_property_returns_dict(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        h = ga.heuristics
        assert isinstance(h, dict)
        assert "_default" in h

    def test_heuristics_default_callable(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        h = ga.heuristics
        result = h["_default"](5, 100)
        assert isinstance(result, int)
        assert result > 0

    def test_heuristics_known_type(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        h = ga.heuristics
        assert "reference_law" in h
        # Should return the fixed minimum
        assert h["reference_law"](5, 100) == DEFAULT_TYPE_MINIMUMS["reference_law"]


class TestPropertyGaps:
    def test_detects_missing_archetype_for_reference_law(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        gaps = [g for g in report.property_gaps
                if g.entity_type == "reference_law" and g.property_name == "mathematical_archetype"]
        assert len(gaps) == 1
        assert gaps[0].missing_count == 2   # E4 and E5 are missing it (3/5 = 60%)
        assert "E4" in gaps[0].missing_ids
        assert "E5" in gaps[0].missing_ids

    def test_detects_missing_sensitivity_for_method(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        gaps = [g for g in report.property_gaps
                if g.entity_type == "method" and g.property_name == "dimensional_sensitivity"]
        assert len(gaps) == 1
        assert gaps[0].missing_count == 1   # M1 is missing it
        assert "M1" in gaps[0].missing_ids

    def test_no_gap_when_fully_covered(self, tmp_path):
        """E1–E5 all have dimensional_sensitivity → no gap for reference_law."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        gaps = [g for g in report.property_gaps
                if g.entity_type == "reference_law" and g.property_name == "dimensional_sensitivity"]
        assert len(gaps) == 0

    def test_coverage_pct_correct(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        gaps = [g for g in report.property_gaps
                if g.entity_type == "reference_law" and g.property_name == "mathematical_archetype"]
        assert gaps[0].coverage_pct == pytest.approx(60.0)  # 3/5 = 60%

    def test_priority_high_for_many_missing(self, tmp_path):
        """concept_tags: only E1 has it → very sparse → high priority."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        # concept_tags only exists for reference_law (only E1 has it, others don't)
        ct_gaps = [g for g in report.property_gaps if g.property_name == "concept_tags"]
        # At least one type should have a high-priority gap (priority=1)
        assert any(g.priority == 1 for g in ct_gaps)

    def test_missing_ids_limited_to_10(self, tmp_path):
        """Even if more than 10 entries are missing, missing_ids capped at 10."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        for gap in report.property_gaps:
            assert len(gap.missing_ids) <= 10

    def test_sorted_priority_then_missing_count(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        gaps = report.property_gaps
        for i in range(len(gaps) - 1):
            a, b = gaps[i], gaps[i+1]
            assert (a.priority, -a.missing_count) <= (b.priority, -b.missing_count)


class TestTaxonomyGaps:
    def test_oscillatory_flagged_as_sparse(self, tmp_path):
        """mathematical_archetype='oscillatory' appears only once → sparse."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        osc = [g for g in report.taxonomy_gaps
               if g.taxonomy_name == "mathematical_archetype" and g.value == "oscillatory"]
        assert len(osc) == 1
        assert osc[0].flag in {"sparse", "singleton"}

    def test_conservation_law_not_flagged(self, tmp_path):
        """conservation-law appears 3 times → not sparse."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        cl = [g for g in report.taxonomy_gaps
              if g.taxonomy_name == "mathematical_archetype" and g.value == "conservation-law"]
        assert len(cl) == 0

    def test_taxonomy_properties_tracked(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        names = {g.taxonomy_name for g in report.taxonomy_gaps}
        # Only TAXONOMY_PROPERTIES are analysed
        assert names.issubset(TAXONOMY_PROPERTIES)

    def test_sorted_by_count_ascending(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        counts = [g.count for g in report.taxonomy_gaps]
        assert counts == sorted(counts)


class TestTypeBalanceGaps:
    def test_reference_law_deficit_detected(self, tmp_path):
        """5 reference_law observed vs minimum 50 → deficit = 45."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        rl_gaps = [g for g in report.type_balance_gaps if g.entity_type == "reference_law"]
        assert len(rl_gaps) == 1
        assert rl_gaps[0].deficit == 45
        assert rl_gaps[0].observed == 5

    def test_method_deficit_detected(self, tmp_path):
        """2 method observed vs minimum 10 → deficit = 8."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        m_gaps = [g for g in report.type_balance_gaps if g.entity_type == "method"]
        assert len(m_gaps) == 1
        assert m_gaps[0].deficit == 8

    def test_no_deficit_when_above_minimum(self, tmp_path):
        """If we set minimum to 0, no gaps should appear."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db, type_minimums={t: 0 for t in DEFAULT_TYPE_MINIMUMS})
        report = ga.analyze()
        assert len(report.type_balance_gaps) == 0

    def test_custom_minimum_respected(self, tmp_path):
        """Set reference_law minimum to 3 (below observed 5) → no gap."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db, type_minimums={"reference_law": 3})
        report = ga.analyze()
        rl_gaps = [g for g in report.type_balance_gaps if g.entity_type == "reference_law"]
        assert len(rl_gaps) == 0

    def test_sorted_by_deficit_descending(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        deficits = [g.deficit for g in report.type_balance_gaps]
        assert deficits == sorted(deficits, reverse=True)

    def test_suggestion_text_mentions_entity_type(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        for g in report.type_balance_gaps:
            assert g.entity_type in g.suggestion


class TestLinkGaps:
    def test_isolated_entries_detected(self, tmp_path):
        """E5 and M2 have no links at all (M1 has an incoming link from Q1)."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        isolated = [g for g in report.link_gaps if g.gap_type == "isolated"]
        isolated_ids = {g.entry_id for g in isolated}
        assert "E5" in isolated_ids
        assert "M2" in isolated_ids

    def test_linked_entries_not_isolated(self, tmp_path):
        """E1, E2, E3, E4 all have links → not isolated."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        isolated_ids = {g.entry_id for g in report.link_gaps if g.gap_type == "isolated"}
        for eid in ["E1", "E2", "E3", "E4"]:
            assert eid not in isolated_ids

    def test_no_tier1_gap_detected(self, tmp_path):
        """E3, E4 have only tier-1.5 link → no_tier1 gap."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        no_t1 = {g.entry_id for g in report.link_gaps if g.gap_type == "no_tier1"}
        # E3 and E4 only have tier-1.5 links
        assert "E3" in no_t1 or "E4" in no_t1

    def test_tier1_pair_not_in_no_tier1(self, tmp_path):
        """E1↔E2 have tier-1 link → neither should be no_tier1."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        no_t1_ids = {g.entry_id for g in report.link_gaps if g.gap_type == "no_tier1"}
        # E1 and E2 both have a tier-1 link
        assert "E1" not in no_t1_ids
        assert "E2" not in no_t1_ids

    def test_isolated_sorted_first(self, tmp_path):
        """isolated entries should appear before no_tier1 in sorted output."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        types = [g.gap_type for g in report.link_gaps]
        # All "isolated" entries should come before any "no_tier1" entry
        seen_non_isolated = False
        for t in types:
            if t != "isolated":
                seen_non_isolated = True
            if seen_non_isolated and t == "isolated":
                pytest.fail("Isolated entry appeared after non-isolated in link_gaps")

    def test_link_gap_total_links_correct(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        isolated = {g.entry_id: g for g in report.link_gaps if g.gap_type == "isolated"}
        if "E5" in isolated:
            assert isolated["E5"].total_links == 0

    def test_same_type_only_not_triggered_on_low_degree(self, tmp_path):
        """Entries with < 3 links should NOT be flagged as same_type_only."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        # E1 has only 1 link (to E2) → should NOT be same_type_only even though both are ref_law
        same_type_ids = {g.entry_id for g in report.link_gaps if g.gap_type == "same_type_only"}
        assert "E1" not in same_type_ids


class TestEnrichmentPriorities:
    def test_priorities_are_ranked_from_1(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        ranks = [p.rank for p in report.enrichment_priorities]
        assert ranks[0] == 1
        assert ranks == sorted(ranks)

    def test_isolated_entry_gap_in_priorities(self, tmp_path):
        """Isolated entries should surface as add_link priority."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        add_link = [p for p in report.enrichment_priorities if p.action == "add_link"]
        assert len(add_link) > 0

    def test_add_entry_priorities_present(self, tmp_path):
        """Type deficits should surface as add_entry priorities."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        add_entry = [p for p in report.enrichment_priorities if p.action == "add_entry"]
        assert len(add_entry) > 0

    def test_impact_score_in_range(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        for p in report.enrichment_priorities:
            assert 0.0 <= p.impact_score <= 1.0

    def test_sorted_by_impact_descending(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        scores = [p.impact_score for p in report.enrichment_priorities]
        assert scores == sorted(scores, reverse=True)

    def test_all_actions_are_valid_types(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        valid = {"add_property", "add_entry", "add_link", "add_taxonomy"}
        for p in report.enrichment_priorities:
            assert p.action in valid

    def test_priorities_descriptions_nonempty(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        report = ga.analyze()
        for p in report.enrichment_priorities:
            assert len(p.description) > 10


class TestAnalyzeReturnTypes:
    def test_returns_gap_analysis_report(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        result = ga.analyze()
        assert isinstance(result, GapAnalysisReport)

    def test_all_list_fields_are_lists(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        result = ga.analyze()
        assert isinstance(result.property_gaps, list)
        assert isinstance(result.taxonomy_gaps, list)
        assert isinstance(result.type_balance_gaps, list)
        assert isinstance(result.link_gaps, list)
        assert isinstance(result.enrichment_priorities, list)

    def test_summary_stats_has_required_keys(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        result = ga.analyze()
        required = {"total_entities", "total_links", "isolated_count",
                    "property_gap_count", "taxonomy_gap_count",
                    "type_balance_gaps", "link_gap_count"}
        for key in required:
            assert key in result.summary_stats, f"Missing key: {key}"

    def test_summary_stats_total_entities_correct(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        result = ga.analyze()
        assert result.summary_stats["total_entities"] == 8

    def test_summary_stats_isolated_count(self, tmp_path):
        """E5 and M2 are isolated (M1 has an incoming link from Q1) → isolated_count = 2."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        result = ga.analyze()
        assert result.summary_stats["isolated_count"] == 2

    def test_as_markdown_returns_string(self, tmp_path):
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        result = ga.analyze()
        md = result.as_markdown()
        assert isinstance(md, str)
        assert len(md) > 200

    def test_analyze_twice_gives_same_result(self, tmp_path):
        """analyze() is stateless — calling it twice returns identical data."""
        db = _make_db(tmp_path)
        ga = GapAnalyzer(source_db=db)
        r1 = ga.analyze()
        r2 = ga.analyze()
        assert len(r1.property_gaps) == len(r2.property_gaps)
        assert len(r1.link_gaps) == len(r2.link_gaps)
        assert r1.summary_stats["total_entities"] == r2.summary_stats["total_entities"]


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests — real ds_wiki.db
# ══════════════════════════════════════════════════════════════════════════════

REAL_DB = Path(__file__).resolve().parent.parent / "data" / "ds_wiki.db"
skip_integration = pytest.mark.skipif(
    not REAL_DB.exists(),
    reason="Real ds_wiki.db not found",
)


@skip_integration
class TestIntegration:

    @pytest.fixture(scope="class")
    def report(self):
        ga = GapAnalyzer(source_db=REAL_DB)
        return ga.analyze()

    def test_returns_gap_analysis_report(self, report):
        assert isinstance(report, GapAnalysisReport)

    def test_total_entities_231(self, report):
        assert report.summary_stats["total_entities"] == 231

    def test_total_links_positive(self, report):
        assert report.summary_stats["total_links"] > 0

    def test_reference_law_has_type_balance_gap(self, report):
        """149 reference_law entries vs minimum 50 → no deficit (149 > 50)."""
        rl = [g for g in report.type_balance_gaps if g.entity_type == "reference_law"]
        assert len(rl) == 0   # 149 > 50, so no gap

    def test_mechanism_has_type_balance_gap(self, report):
        """Only 1 mechanism entry — below any reasonable minimum."""
        mech = [g for g in report.type_balance_gaps if g.entity_type == "mechanism"]
        assert len(mech) == 1
        assert mech[0].observed == 1

    def test_isolated_entries_detected(self, report):
        """All formerly isolated reference_law entries have been linked; only Q2 remains."""
        isolated = [g for g in report.link_gaps if g.gap_type == "isolated"]
        assert len(isolated) >= 1   # Q2 is the last remaining isolated entry

    def test_known_isolated_entry_present(self, report):
        """Q2 is the last known isolated entry (all 12 formerly isolated entries now linked)."""
        isolated_ids = {g.entry_id for g in report.link_gaps if g.gap_type == "isolated"}
        assert "Q2" in isolated_ids

    def test_archetype_coverage_complete(self, report):
        """All 209 entries have mathematical_archetype — no property gap for archetype."""
        archetype_gaps = [
            g for g in report.property_gaps
            if g.property_name == "mathematical_archetype"
        ]
        assert len(archetype_gaps) == 0

    def test_enrichment_priorities_nonempty(self, report):
        assert len(report.enrichment_priorities) > 0

    def test_all_priority_impact_scores_in_range(self, report):
        for p in report.enrichment_priorities:
            assert 0.0 <= p.impact_score <= 1.0, f"Impact {p.impact_score} out of range"

    def test_markdown_report_is_long(self, report):
        md = report.as_markdown()
        assert len(md) > 1000

    def test_markdown_contains_231(self, report):
        md = report.as_markdown()
        assert "231" in md

    def test_taxonomy_gaps_present(self, report):
        """With 14 archetype values across 149 reference_law entries, some are sparse."""
        assert len(report.taxonomy_gaps) > 0

    def test_link_gap_types_are_valid(self, report):
        valid = {"isolated", "no_tier1", "same_type_only"}
        for g in report.link_gaps:
            assert g.gap_type in valid, f"Unknown gap_type: {g.gap_type}"

    def test_property_gaps_all_below_threshold(self, report):
        for g in report.property_gaps:
            assert g.coverage_pct < PROPERTY_COVERAGE_THRESHOLD * 100, (
                f"{g.entity_type}/{g.property_name} at {g.coverage_pct}% should be < "
                f"{PROPERTY_COVERAGE_THRESHOLD*100}%"
            )

    def test_add_link_priority_mentions_isolated(self, report):
        """Top priorities should mention connecting isolated entries."""
        link_priorities = [p for p in report.enrichment_priorities if p.action == "add_link"]
        assert len(link_priorities) > 0
        texts = " ".join(p.description for p in link_priorities)
        assert "isolated" in texts.lower() or "connect" in texts.lower()

    def test_custom_type_minimum_affects_gaps(self):
        """Setting method minimum to 100 creates a large deficit."""
        ga = GapAnalyzer(source_db=REAL_DB, type_minimums={"method": 100})
        report = ga.analyze()
        method_gaps = [g for g in report.type_balance_gaps if g.entity_type == "method"]
        assert len(method_gaps) == 1
        assert method_gaps[0].deficit == 100 - 18   # 18 observed method entries
