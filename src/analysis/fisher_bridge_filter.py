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

The trust_score = (1 − η) × cosine_sim combines:
  - (1 − η): reward for structured DS Wiki anchor (low disorder)
  - cosine_sim: existing embedding-similarity confidence

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

ETA_TRUST_THRESHOLD: float = 0.65   # noise-dominated bridges are unreliable
_STRUCTURED_REGIMES = {RegimeType.RADIAL_DOMINATED, RegimeType.ISOTROPIC}


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class BridgeQualityScore:
    rrp_entry_id:  str
    ds_entry_id:   str
    cosine_sim:    float          # existing similarity score from bridge table
    eta:           float          # disorder index η at DS Wiki node
    regime:        RegimeType     # X0 regime of DS Wiki anchor
    trust_score:   float          # composite: (1 − η) × cosine_sim
    is_structured: bool           # True if eta < ETA_TRUST_THRESHOLD

    def as_dict(self) -> dict:
        return {
            "rrp_entry_id":  self.rrp_entry_id,
            "ds_entry_id":   self.ds_entry_id,
            "cosine_sim":    self.cosine_sim,
            "eta":           self.eta,
            "regime":        self.regime.value,
            "trust_score":   self.trust_score,
            "is_structured": self.is_structured,
        }


# ── Core functions ────────────────────────────────────────────────────────────

def score_bridge(
    rrp_entry_id: str,
    ds_entry_id:  str,
    cosine_sim:   float,
    ds_sweep:     FisherSweepResult,
    eta_threshold: float = ETA_TRUST_THRESHOLD,
) -> BridgeQualityScore:
    """
    Score a single cross-universe bridge using the DS Wiki node's FIM geometry.

    Looks up ds_entry_id in the pre-computed FisherSweepResult.  If the node
    was skipped or is absent from the sweep, assigns eta=1.0 (worst case) and
    DEGENERATE regime — the bridge receives a low trust_score but is not dropped.

    trust_score = (1 − η) × cosine_sim
    is_structured = eta < eta_threshold
    """
    fisher_result = ds_sweep.results.get(ds_entry_id)

    if fisher_result is None or fisher_result.skipped:
        # Node absent from graph or degree < 2: cannot verify, assume worst case
        eta    = 1.0
        regime = RegimeType.DEGENERATE
    else:
        eta    = fisher_result.eta
        regime = fisher_result.regime

    trust_score   = (1.0 - eta) * cosine_sim
    is_structured = eta < eta_threshold

    return BridgeQualityScore(
        rrp_entry_id=rrp_entry_id,
        ds_entry_id=ds_entry_id,
        cosine_sim=cosine_sim,
        eta=eta,
        regime=regime,
        trust_score=trust_score,
        is_structured=is_structured,
    )


def filter_bridges(
    bridges:       list[dict],
    ds_sweep:      FisherSweepResult,
    eta_threshold: float = ETA_TRUST_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """
    Split a list of bridge dicts into (trusted, noise) using FIM geometry.

    Each bridge dict must have keys: 'rrp_entry_id', 'ds_entry_id', 'similarity'.
    Returns (trusted_bridges, noise_bridges).

    IMPORTANT: noise_bridges are RETAINED — returned in the second list for
    human review.  Nothing is deleted.

    A bridge is trusted when the DS Wiki anchor node has eta < eta_threshold
    (radial or isotropic regime — structured, not noise-dominated).
    """
    trusted: list[dict] = []
    noise:   list[dict] = []

    for bridge in bridges:
        rrp_id  = bridge.get("rrp_entry_id", "")
        ds_id   = bridge.get("ds_entry_id", "")
        sim     = float(bridge.get("similarity", 0.0))

        score = score_bridge(rrp_id, ds_id, sim, ds_sweep, eta_threshold)

        # Attach quality metadata to the bridge dict (non-destructive copy)
        annotated = dict(bridge)
        annotated["fisher_eta"]       = score.eta
        annotated["fisher_regime"]    = score.regime.value
        annotated["fisher_trust"]     = score.trust_score
        annotated["fisher_structured"] = score.is_structured

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
) -> list[BridgeQualityScore]:
    """
    Load all cross_universe_bridges from an RRP bundle DB and score each one.

    Convenience function for CLI and reporting use.  Filters to bridges with
    similarity >= min_sim before scoring.

    Returns list sorted descending by trust_score.
    """
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
        score_bridge(r[0], r[1], float(r[2]), ds_sweep, eta_threshold)
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

    Returns (trusted_scores, noise_scores).
    """
    G, _ = build_wiki_graph(ds_wiki_db)
    sweep = sweep_graph(G, "ds_wiki", kernel, alpha=alpha)

    conn = sqlite3.connect(rrp_db)
    rows = conn.execute(
        "SELECT rrp_entry_id, ds_entry_id, similarity "
        "FROM cross_universe_bridges ORDER BY similarity DESC"
    ).fetchall()
    conn.close()

    trusted: list[BridgeQualityScore] = []
    noise:   list[BridgeQualityScore] = []

    for rrp_id, ds_id, sim in rows:
        score = score_bridge(rrp_id, ds_id, float(sim), sweep, eta_threshold)
        (trusted if score.is_structured else noise).append(score)

    return trusted, noise
