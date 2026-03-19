"""
fisher_bridge_filter.py — FIM-based bridge quality scorer.

Integrates Fisher Information Matrix geometry with the cross-universe bridge
pipeline.  Given a pre-computed FisherSweepResult for the DS Wiki graph, each
cross-universe bridge is scored by the information geometry of its DS Wiki
anchor node.

Rationale (from X0_FIM_Regimes):
  - RADIAL/ISOTROPIC anchors: structured knowledge hub, bridge is trustworthy
  - NOISE_DOMINATED anchors: high disorder (η ≥ 0.65), bridge is unreliable
  - DEGENERATE anchors: degree < 2, cannot compute FIM; treat as unverified

The trust_score = (1 − η) × cosine_sim × formality_weight combines:
  - (1 − η): reward for structured DS Wiki anchor (low disorder)
  - cosine_sim: existing embedding-similarity confidence
  - formality_weight: domain rigor multiplier from formality_tier (Phase 4.3)

Formality-tier-aware scoring (Phase 4.3):
  - Tier 1 (physics/math/logic): formality_weight=1.0, eta_threshold=0.60
  - Tier 2 (chemistry/biology):  formality_weight=0.85, eta_threshold=0.65
  - Tier 3 (soft science):       formality_weight=0.70, eta_threshold=0.75

Noise bridges are RETAINED in a separate list — they are flagged for human
review, not discarded.  This module never modifies the underlying DB.

Entry points:
    score_bridge(...)        → BridgeQualityScore  (single bridge)
    filter_bridges(...)      → (trusted, noise)    (split all bridges)
    score_bridges_from_db(...) → list[BridgeQualityScore]  (batch from DB)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from analysis.fisher_diagnostics import (
    FisherSweepResult,
    RegimeType,
    build_wiki_graph,
    sweep_graph,
    KernelType,
)


# ── Constants ─────────────────────────────────────────────────────────────────

ETA_TRUST_THRESHOLD: float = 0.65   # default noise threshold (Tier 2)
_STRUCTURED_REGIMES = {RegimeType.RADIAL_DOMINATED, RegimeType.ISOTROPIC}

# Phase 4.3 — per-tier formality weights and eta thresholds
FORMALITY_WEIGHTS: dict[int, float] = {
    1: 1.00,   # physics/math/logic — full trust in link semantics
    2: 0.85,   # chemistry/biology — moderate trust
    3: 0.70,   # soft science — lower trust in strict link semantics
}

FORMALITY_ETA_THRESHOLDS: dict[int, float] = {
    1: 0.60,   # physics/math — stricter: structured nodes expected
    2: 0.65,   # chemistry — moderate
    3: 0.75,   # soft science — permissive: noisier structure acceptable
}

def formality_weight(tier: int) -> float:
    """Return the formality weight for a given tier (default Tier 2)."""
    return FORMALITY_WEIGHTS.get(tier, FORMALITY_WEIGHTS[2])

def formality_eta_threshold(tier: int) -> float:
    """Return the eta threshold for a given formality tier."""
    return FORMALITY_ETA_THRESHOLDS.get(tier, ETA_TRUST_THRESHOLD)


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class BridgeQualityScore:
    rrp_entry_id:    str
    ds_entry_id:     str
    cosine_sim:      float          # existing similarity score from bridge table
    eta:             float          # disorder index η at DS Wiki node
    regime:          RegimeType     # X0 regime of DS Wiki anchor
    trust_score:     float          # composite: (1 − η) × cosine_sim × formality_weight
    is_structured:   bool           # True if eta < tier-adjusted threshold
    formality_tier:  int = 2        # DS Wiki anchor's formality tier (1/2/3)

    def as_dict(self) -> dict:
        return {
            "rrp_entry_id":    self.rrp_entry_id,
            "ds_entry_id":     self.ds_entry_id,
            "cosine_sim":      self.cosine_sim,
            "eta":             self.eta,
            "regime":          self.regime.value,
            "trust_score":     self.trust_score,
            "is_structured":   self.is_structured,
            "formality_tier":  self.formality_tier,
        }


# ── Core functions ────────────────────────────────────────────────────────────

def score_bridge(
    rrp_entry_id: str,
    ds_entry_id:  str,
    cosine_sim:   float,
    ds_sweep:     FisherSweepResult,
    eta_threshold: float = ETA_TRUST_THRESHOLD,
    ds_formality_tier: int = 2,
) -> BridgeQualityScore:
    """
    Score a single cross-universe bridge using the DS Wiki node's FIM geometry.

    Looks up ds_entry_id in the pre-computed FisherSweepResult.  If the node
    was skipped or is absent from the sweep, assigns eta=1.0 (worst case) and
    DEGENERATE regime — the bridge receives a low trust_score but is not dropped.

    trust_score = (1 − η) × cosine_sim × formality_weight
    is_structured = eta < formality_eta_threshold(tier)

    Phase 4.3: ds_formality_tier adjusts both the trust_score weight and the
    eta threshold used for the structured/noise gate.
    """
    fisher_result = ds_sweep.results.get(ds_entry_id)

    if fisher_result is None or fisher_result.skipped:
        # Node absent from graph or degree < 2: cannot verify, assume worst case
        eta    = 1.0
        regime = RegimeType.DEGENERATE
    else:
        eta    = fisher_result.eta
        regime = fisher_result.regime

    fw            = formality_weight(ds_formality_tier)
    trust_score   = (1.0 - eta) * cosine_sim * fw
    tier_threshold = formality_eta_threshold(ds_formality_tier)
    is_structured = eta < tier_threshold

    return BridgeQualityScore(
        rrp_entry_id=rrp_entry_id,
        ds_entry_id=ds_entry_id,
        cosine_sim=cosine_sim,
        eta=eta,
        regime=regime,
        trust_score=trust_score,
        is_structured=is_structured,
        formality_tier=ds_formality_tier,
    )


def _load_formality_tiers(wiki_db: Optional[Path] = None) -> dict[str, int]:
    """Load formality_tier for all DS Wiki entries. Returns {entry_id: tier}."""
    if wiki_db is None:
        return {}
    conn = sqlite3.connect(wiki_db)
    # Check if column exists
    cols = {row[1] for row in conn.execute("PRAGMA table_info(entries)")}
    if "formality_tier" not in cols:
        conn.close()
        return {}
    tiers = {
        row[0]: row[1]
        for row in conn.execute("SELECT id, formality_tier FROM entries")
        if row[1] is not None
    }
    conn.close()
    return tiers


def filter_bridges(
    bridges:       list[dict],
    ds_sweep:      FisherSweepResult,
    eta_threshold: float = ETA_TRUST_THRESHOLD,
    formality_tiers: Optional[dict[str, int]] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Split a list of bridge dicts into (trusted, noise) using FIM geometry.

    Each bridge dict must have keys: 'rrp_entry_id', 'ds_entry_id', 'similarity'.
    Returns (trusted_bridges, noise_bridges).

    IMPORTANT: noise_bridges are RETAINED — returned in the second list for
    human review.  Nothing is deleted.

    A bridge is trusted when the DS Wiki anchor node has eta below its
    formality-tier-adjusted threshold (Phase 4.3).
    """
    if formality_tiers is None:
        formality_tiers = {}

    trusted: list[dict] = []
    noise:   list[dict] = []

    for bridge in bridges:
        rrp_id  = bridge.get("rrp_entry_id", "")
        ds_id   = bridge.get("ds_entry_id", "")
        sim     = float(bridge.get("similarity", 0.0))
        tier    = formality_tiers.get(ds_id, 2)

        score = score_bridge(rrp_id, ds_id, sim, ds_sweep, eta_threshold,
                             ds_formality_tier=tier)

        # Attach quality metadata to the bridge dict (non-destructive copy)
        annotated = dict(bridge)
        annotated["fisher_eta"]            = score.eta
        annotated["fisher_regime"]         = score.regime.value
        annotated["fisher_trust"]          = score.trust_score
        annotated["fisher_structured"]     = score.is_structured
        annotated["fisher_formality_tier"] = score.formality_tier

        if score.is_structured:
            trusted.append(annotated)
        else:
            noise.append(annotated)

    return trusted, noise


def score_bridges_from_db(
    rrp_db:        Path,
    ds_sweep:      FisherSweepResult,
    eta_threshold: float = ETA_TRUST_THRESHOLD,
    min_sim:       float = 0.0,
    formality_tiers: Optional[dict[str, int]] = None,
) -> list[BridgeQualityScore]:
    """
    Load all cross_universe_bridges from an RRP bundle DB and score each one.

    Convenience function for CLI and reporting use.  Filters to bridges with
    similarity >= min_sim before scoring.

    Returns list sorted descending by trust_score.
    """
    if formality_tiers is None:
        formality_tiers = {}

    conn = sqlite3.connect(rrp_db)
    rows = conn.execute(
        "SELECT rrp_entry_id, ds_entry_id, similarity "
        "FROM cross_universe_bridges "
        "WHERE similarity >= ? "
        "ORDER BY similarity DESC",
        (min_sim,),
    ).fetchall()
    conn.close()

    scores = [
        score_bridge(r[0], r[1], float(r[2]), ds_sweep, eta_threshold,
                     ds_formality_tier=formality_tiers.get(r[1], 2))
        for r in rows
    ]
    return sorted(scores, key=lambda s: s.trust_score, reverse=True)


def run_bridge_filter(
    rrp_db:        Path,
    ds_wiki_db:    Path,
    eta_threshold: float = ETA_TRUST_THRESHOLD,
    kernel:        KernelType = KernelType.EXPONENTIAL,
    alpha:         float = 1.0,
) -> tuple[list[BridgeQualityScore], list[BridgeQualityScore]]:
    """
    End-to-end bridge filter: build DS Wiki graph → sweep → score all bridges.

    Loads formality_tier from ds_wiki_db for per-tier scoring (Phase 4.3).
    Returns (trusted_scores, noise_scores).
    """
    G, _ = build_wiki_graph(ds_wiki_db)
    sweep = sweep_graph(G, "ds_wiki", kernel, alpha=alpha)

    formality_tiers = _load_formality_tiers(ds_wiki_db)

    conn = sqlite3.connect(rrp_db)
    rows = conn.execute(
        "SELECT rrp_entry_id, ds_entry_id, similarity "
        "FROM cross_universe_bridges ORDER BY similarity DESC"
    ).fetchall()
    conn.close()

    trusted: list[BridgeQualityScore] = []
    noise:   list[BridgeQualityScore] = []

    for rrp_id, ds_id, sim in rows:
        tier = formality_tiers.get(ds_id, 2)
        score = score_bridge(rrp_id, ds_id, float(sim), sweep, eta_threshold,
                             ds_formality_tier=tier)
        (trusted if score.is_structured else noise).append(score)

    return trusted, noise
