"""
test_domain_boundaries.py — Phase 4.2 domain boundary validation tests.

Tests:
    TestDomainValidation     — single bridge domain crossing rules
    TestDomainAdjacency      — domain adjacency matrix
    TestDomainBoundaryReport — full RRP boundary scan (synthetic DB)
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.domain_boundaries import (  # noqa: E402
    DomainStatus,
    DomainValidation,
    DomainBoundaryReport,
    validate_bridge_domain,
    check_domain_boundaries,
    _extract_primary_domain,
    _extract_all_domains,
    _domains_adjacent,
    _any_domain_adjacent,
    CROSS_DOMAIN_VALID,
    SAME_DOMAIN_PREFERRED,
    TENSION_TYPES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TestDomainExtraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainExtraction:
    def test_primary_single(self):
        assert _extract_primary_domain("physics") == "physics"

    def test_primary_multi(self):
        assert _extract_primary_domain("physics · chemistry") == "physics"

    def test_primary_empty(self):
        assert _extract_primary_domain("") == "unknown"

    def test_all_single(self):
        assert _extract_all_domains("physics") == {"physics"}

    def test_all_multi(self):
        assert _extract_all_domains("physics · chemistry") == {"physics", "chemistry"}

    def test_all_empty(self):
        assert _extract_all_domains("") == {"unknown"}


# ═══════════════════════════════════════════════════════════════════════════════
# TestDomainAdjacency
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainAdjacency:
    def test_same_domain(self):
        assert _domains_adjacent("physics", "physics") is True

    def test_physics_chemistry(self):
        assert _domains_adjacent("physics", "chemistry") is True

    def test_physics_biology(self):
        # Not directly adjacent
        assert _domains_adjacent("physics", "biology") is False

    def test_math_cs(self):
        assert _domains_adjacent("mathematics", "computer science") is True

    def test_formal_logic_math(self):
        assert _domains_adjacent("formal logic", "mathematics") is True

    def test_biology_networks(self):
        assert _domains_adjacent("biology", "networks") is True

    def test_multi_domain_any_adjacent(self):
        # physics · chemistry has physics, which is adjacent to mathematics
        assert _any_domain_adjacent({"physics", "chemistry"}, {"mathematics"}) is True

    def test_multi_domain_none_adjacent(self):
        assert _any_domain_adjacent({"biology"}, {"formal logic"}) is False


# ═══════════════════════════════════════════════════════════════════════════════
# TestDomainValidation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainValidation:
    def test_analogous_to_cross_domain_valid(self):
        v = validate_bridge_domain("E1", "MATH4", "analogous to", "biology", "mathematics")
        assert v.status == DomainStatus.VALID

    def test_couples_to_cross_domain_valid(self):
        v = validate_bridge_domain("E1", "CHEM5", "couples to", "biology", "chemistry")
        assert v.status == DomainStatus.VALID

    def test_tests_cross_domain_valid(self):
        v = validate_bridge_domain("E1", "TD3", "tests", "biology", "physics")
        assert v.status == DomainStatus.VALID

    def test_tensions_with_cross_domain_valid(self):
        v = validate_bridge_domain("E1", "QM5", "tensions with", "biology", "physics")
        assert v.status == DomainStatus.VALID

    def test_derives_from_same_domain_valid(self):
        v = validate_bridge_domain("E1", "E2", "derives from", "physics", "physics")
        assert v.status == DomainStatus.VALID

    def test_derives_from_adjacent_domain_valid(self):
        v = validate_bridge_domain("E1", "MATH4", "derives from", "physics", "mathematics")
        assert v.status == DomainStatus.VALID

    def test_derives_from_non_adjacent_warning(self):
        v = validate_bridge_domain("E1", "BIO1", "derives from", "formal logic", "biology")
        assert v.status == DomainStatus.WARNING
        assert "non-adjacent" in v.reason

    def test_generalizes_non_adjacent_warning(self):
        v = validate_bridge_domain("E1", "ES1", "generalizes", "formal logic", "earth sciences")
        assert v.status == DomainStatus.WARNING

    def test_implements_adjacent_valid(self):
        v = validate_bridge_domain("E1", "CS1", "implements", "mathematics", "computer science")
        assert v.status == DomainStatus.VALID

    def test_unknown_link_type_warning(self):
        v = validate_bridge_domain("E1", "E2", "related", "physics", "physics")
        assert v.status == DomainStatus.WARNING
        assert "unclassified" in v.reason

    def test_multi_domain_source(self):
        # physics · chemistry source → mathematics target via derives_from
        # physics is adjacent to mathematics, so valid
        v = validate_bridge_domain("E1", "MATH1", "derives from", "physics · chemistry", "mathematics")
        assert v.status == DomainStatus.VALID


# ═══════════════════════════════════════════════════════════════════════════════
# TestDomainBoundaryReport — synthetic DB
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainBoundaryReport:
    @pytest.fixture
    def dbs(self, tmp_path):
        """Create synthetic RRP and wiki DBs for domain boundary testing."""
        # Wiki DB
        wiki_path = tmp_path / "wiki.db"
        wconn = sqlite3.connect(wiki_path)
        wconn.execute("""
            CREATE TABLE entries (
                id TEXT PRIMARY KEY, title TEXT, entry_type TEXT,
                domain TEXT, formality_tier INTEGER DEFAULT 2
            )
        """)
        wconn.executemany(
            "INSERT INTO entries (id, title, entry_type, domain, formality_tier) VALUES (?,?,?,?,?)",
            [
                ("TD3", "Second Law", "reference_law", "physics", 1),
                ("MATH4", "Godel", "reference_law", "mathematics", 1),
                ("BIO1", "Mendelian Laws", "reference_law", "biology", 2),
            ],
        )
        wconn.commit()
        wconn.close()

        # RRP DB
        rrp_path = tmp_path / "rrp.db"
        rconn = sqlite3.connect(rrp_path)
        rconn.execute("""
            CREATE TABLE entries (
                id TEXT PRIMARY KEY, title TEXT, entry_type TEXT, domain TEXT
            )
        """)
        rconn.executemany(
            "INSERT INTO entries (id, title, entry_type, domain) VALUES (?,?,?,?)",
            [
                ("R1", "Reaction A", "instantiation", "chemistry"),
                ("R2", "Model B", "method", "formal logic"),
            ],
        )
        rconn.execute("""
            CREATE TABLE cross_universe_bridges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rrp_entry_id TEXT, ds_entry_id TEXT,
                similarity REAL, proposed_link_type TEXT
            )
        """)
        rconn.executemany(
            "INSERT INTO cross_universe_bridges (rrp_entry_id, ds_entry_id, similarity, proposed_link_type) VALUES (?,?,?,?)",
            [
                ("R1", "TD3", 0.82, "couples to"),      # chemistry→physics: valid (cross-domain OK)
                ("R1", "BIO1", 0.78, "derives from"),    # chemistry→biology: adjacent, valid
                ("R2", "BIO1", 0.76, "derives from"),    # formal logic→biology: non-adjacent, warning
                ("R2", "MATH4", 0.88, "analogous to"),   # any domain: valid
            ],
        )
        rconn.commit()
        rconn.close()

        return rrp_path, wiki_path

    def test_report_counts(self, dbs):
        rrp_path, wiki_path = dbs
        report = check_domain_boundaries(rrp_path, wiki_path)
        assert report.total_bridges == 4
        assert report.valid_count == 3    # couples to, derives_from adjacent, analogous to
        assert report.warning_count == 1  # derives_from non-adjacent
        assert report.violation_count == 0

    def test_report_domain_coverage(self, dbs):
        rrp_path, wiki_path = dbs
        report = check_domain_boundaries(rrp_path, wiki_path)
        assert "physics" in report.domain_coverage
        assert "biology" in report.domain_coverage
        assert "mathematics" in report.domain_coverage

    def test_violation_rate(self, dbs):
        rrp_path, wiki_path = dbs
        report = check_domain_boundaries(rrp_path, wiki_path)
        assert report.violation_rate == 0.0

    def test_warnings_detail(self, dbs):
        rrp_path, wiki_path = dbs
        report = check_domain_boundaries(rrp_path, wiki_path)
        warnings = report.warnings
        assert len(warnings) == 1
        assert warnings[0].source_id == "R2"
        assert warnings[0].target_id == "BIO1"
        assert "non-adjacent" in warnings[0].reason
