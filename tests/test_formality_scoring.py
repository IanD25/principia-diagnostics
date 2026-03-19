"""
test_formality_scoring.py — Phase 4.3 formality-tier-aware scoring tests.

Tests:
    TestFormalityWeights            — weight and threshold lookup functions
    TestFormalityBridgeFilter       — formality-aware bridge quality scoring
    TestFormalityStructuralAlignment — formality-weighted signed scores
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.fisher_bridge_filter import (  # noqa: E402
    BridgeQualityScore,
    FORMALITY_WEIGHTS,
    FORMALITY_ETA_THRESHOLDS,
    formality_weight,
    formality_eta_threshold,
    score_bridge,
    _load_formality_tiers,
)
from analysis.fisher_diagnostics import (  # noqa: E402
    FisherResult,
    FisherSweepResult,
    RegimeType,
    KernelType,
)
from analysis.structural_alignment import (  # noqa: E402
    BridgeAlignment,
    _formality_weight_sa,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TestFormalityWeights
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormalityWeights:
    def test_tier1_weight(self):
        assert formality_weight(1) == 1.0

    def test_tier2_weight(self):
        assert formality_weight(2) == 0.85

    def test_tier3_weight(self):
        assert formality_weight(3) == 0.70

    def test_unknown_tier_defaults_to_tier2(self):
        assert formality_weight(99) == 0.85

    def test_tier1_eta_threshold(self):
        assert formality_eta_threshold(1) == 0.60

    def test_tier2_eta_threshold(self):
        assert formality_eta_threshold(2) == 0.65

    def test_tier3_eta_threshold(self):
        assert formality_eta_threshold(3) == 0.75

    def test_sa_weight_matches(self):
        """Structural alignment weights should match bridge filter weights."""
        assert _formality_weight_sa(1) == 1.0
        assert _formality_weight_sa(2) == 0.85
        assert _formality_weight_sa(3) == 0.70


# ═══════════════════════════════════════════════════════════════════════════════
# TestFormalityBridgeFilter
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormalityBridgeFilter:
    @pytest.fixture
    def mock_sweep(self):
        """Create a mock FisherSweepResult with known eta values."""
        _common = dict(
            kernel_type=KernelType.EXPONENTIAL, alpha=1.0,
            sv_profile=[1.0], raw_sigmas=[1.0], n_vertices=10,
        )
        sweep = MagicMock(spec=FisherSweepResult)
        sweep.results = {
            "PHYS1": FisherResult(
                node_id="PHYS1", d_eff=5, pr=3.2, eta=0.30,
                regime=RegimeType.RADIAL_DOMINATED,
                center_degree=8, skipped=False, **_common,
            ),
            "BIO1": FisherResult(
                node_id="BIO1", d_eff=3, pr=2.1, eta=0.62,
                regime=RegimeType.ISOTROPIC,
                center_degree=4, skipped=False, **_common,
            ),
            "SOFT1": FisherResult(
                node_id="SOFT1", d_eff=2, pr=1.5, eta=0.72,
                regime=RegimeType.NOISE_DOMINATED,
                center_degree=3, skipped=False, **_common,
            ),
        }
        return sweep

    def test_tier1_trust_score_full_weight(self, mock_sweep):
        """Tier 1 bridge: trust_score = (1-eta) * sim * 1.0"""
        score = score_bridge("R1", "PHYS1", 0.85, mock_sweep, ds_formality_tier=1)
        expected = (1.0 - 0.30) * 0.85 * 1.0
        assert abs(score.trust_score - expected) < 1e-6
        assert score.formality_tier == 1

    def test_tier2_trust_score_reduced(self, mock_sweep):
        """Tier 2 bridge: trust_score = (1-eta) * sim * 0.85"""
        score = score_bridge("R1", "BIO1", 0.80, mock_sweep, ds_formality_tier=2)
        expected = (1.0 - 0.62) * 0.80 * 0.85
        assert abs(score.trust_score - expected) < 1e-6
        assert score.formality_tier == 2

    def test_tier3_trust_score_reduced_more(self, mock_sweep):
        """Tier 3 bridge: trust_score = (1-eta) * sim * 0.70"""
        score = score_bridge("R1", "SOFT1", 0.75, mock_sweep, ds_formality_tier=3)
        expected = (1.0 - 0.72) * 0.75 * 0.70
        assert abs(score.trust_score - expected) < 1e-6
        assert score.formality_tier == 3

    def test_tier1_stricter_threshold(self, mock_sweep):
        """Tier 1 uses eta_threshold=0.60, so eta=0.62 is NOT structured for Tier 1."""
        score = score_bridge("R1", "BIO1", 0.80, mock_sweep, ds_formality_tier=1)
        assert score.is_structured is False  # 0.62 >= 0.60

    def test_tier2_default_threshold(self, mock_sweep):
        """Tier 2 uses eta_threshold=0.65, so eta=0.62 IS structured."""
        score = score_bridge("R1", "BIO1", 0.80, mock_sweep, ds_formality_tier=2)
        assert score.is_structured is True  # 0.62 < 0.65

    def test_tier3_permissive_threshold(self, mock_sweep):
        """Tier 3 uses eta_threshold=0.75, so eta=0.72 IS structured."""
        score = score_bridge("R1", "SOFT1", 0.75, mock_sweep, ds_formality_tier=3)
        assert score.is_structured is True  # 0.72 < 0.75

    def test_degenerate_node_always_worst(self, mock_sweep):
        """Missing node always gets eta=1.0 regardless of tier."""
        score = score_bridge("R1", "MISSING", 0.90, mock_sweep, ds_formality_tier=1)
        assert score.eta == 1.0
        assert score.trust_score == 0.0
        assert score.is_structured is False

    def test_as_dict_includes_tier(self, mock_sweep):
        score = score_bridge("R1", "PHYS1", 0.85, mock_sweep, ds_formality_tier=1)
        d = score.as_dict()
        assert d["formality_tier"] == 1

    def test_backward_compatible_default(self, mock_sweep):
        """Without explicit tier, defaults to Tier 2."""
        score = score_bridge("R1", "PHYS1", 0.85, mock_sweep)
        assert score.formality_tier == 2


# ═══════════════════════════════════════════════════════════════════════════════
# TestLoadFormalityTiers
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFormalityTiers:
    def test_load_from_db(self, tmp_path):
        db_path = tmp_path / "wiki.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE entries (
                id TEXT PRIMARY KEY, title TEXT, entry_type TEXT,
                domain TEXT, formality_tier INTEGER
            )
        """)
        conn.executemany(
            "INSERT INTO entries VALUES (?,?,?,?,?)",
            [
                ("FL1", "Validity", "reference_law", "formal logic", 1),
                ("BIO1", "Mendel", "reference_law", "biology", 2),
                ("Q1", "Open", "open question", "geometry", 3),
            ],
        )
        conn.commit()
        conn.close()

        tiers = _load_formality_tiers(db_path)
        assert tiers["FL1"] == 1
        assert tiers["BIO1"] == 2
        assert tiers["Q1"] == 3

    def test_load_missing_column(self, tmp_path):
        db_path = tmp_path / "wiki.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE entries (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute("INSERT INTO entries VALUES ('E1', 'Test')")
        conn.commit()
        conn.close()

        tiers = _load_formality_tiers(db_path)
        assert tiers == {}

    def test_load_none_returns_empty(self):
        tiers = _load_formality_tiers(None)
        assert tiers == {}


# ═══════════════════════════════════════════════════════════════════════════════
# TestFormalityInBridgeAlignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormalityInBridgeAlignment:
    def test_alignment_has_tier_field(self):
        ba = BridgeAlignment(
            rrp_entry_id="R1", rrp_entry_title="Test",
            ds_entry_id="PHYS1", ds_entry_title="Physics Law",
            raw_sim=0.85, polarity=1.0,
            signed_score=0.85 * 1.0 * 1.0,
            path_description="direct", hop=1,
            formality_tier=1,
        )
        assert ba.formality_tier == 1

    def test_alignment_default_tier(self):
        ba = BridgeAlignment(
            rrp_entry_id="R1", rrp_entry_title="Test",
            ds_entry_id="E1", ds_entry_title="Entry",
            raw_sim=0.80, polarity=0.5,
            signed_score=0.80 * 0.5 * 0.85,
            path_description="direct", hop=1,
        )
        assert ba.formality_tier == 2  # default

    def test_tier1_signed_score_unweighted(self):
        """Tier 1 formality weight is 1.0 — no reduction."""
        fw = _formality_weight_sa(1)
        signed = 0.85 * 1.0 * fw
        assert signed == 0.85

    def test_tier3_signed_score_reduced(self):
        """Tier 3 formality weight is 0.70 — 30% reduction."""
        fw = _formality_weight_sa(3)
        signed = 0.85 * 1.0 * fw
        assert abs(signed - 0.595) < 1e-6
