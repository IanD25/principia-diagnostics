"""
fisher_report.py — Two-tier PFD diagnostic report generator.

Implements Steps 3 + 5 of the six-step PFD pipeline
(FISHER_PIPELINE_REDESIGN.md) and combines results into a single report:

  Tier 1: Internal consistency of the RRP universe's own graph
  Tier 2: Bridge quality — how well the RRP integrates into DS Wiki

Entry points:
    generate_report(rrp_db, wiki_db, ...) → PFDReport
    PFDReport.as_text()                   → plain-text diagnostic report
    PFDReport.as_dict()                   → machine-readable dict for programmatic access
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from analysis.fisher_diagnostics import (
    KernelType,
    RegimeType,
    FisherSweepResult,
    build_wiki_graph,
    build_bridge_graph,
    sweep_graph,
)


# ── Report data structures ────────────────────────────────────────────────────

@dataclass
class HubEntry:
    node_id:       str
    regime:        str
    d_eff:         int
    pr:            float
    eta:           float
    center_degree: int


@dataclass
class BridgeEntry:
    rrp_id:    str
    wiki_id:   str
    similarity: float
    regime:    str      # regime of the RRP node in G_bridge


@dataclass
class AnchorLoad:
    wiki_id:    str
    n_bridges:  int     # number of distinct RRP entries bridging to this wiki node


@dataclass
class PFDReport:
    # Identity
    rrp_name:     str
    rrp_db:       str
    wiki_db:      str
    run_date:     str
    alpha:        float
    min_sim:      float

    # Tier-1 — internal consistency
    tier1_n_analyzed:    int
    tier1_n_total:       int
    tier1_coherence:     float      # fraction non-noise
    tier1_mean_d_eff:    float
    tier1_isolation_rate: float     # fraction with degree < 2
    tier1_regime_counts: dict
    tier1_verdict:       str        # INTERNALLY CONSISTENT | MARGINAL | FRAGMENTED
    tier1_top_hubs:      list       # list[HubEntry]

    # Tier-2 — bridge quality
    tier2_bridge_edges:        int
    tier2_n_bridged_rrp:       int   # RRP nodes analyzable in bridge graph
    tier2_n_rrp_total:         int
    tier2_wiki_anchors_reached: int  # unique DS Wiki nodes touched
    tier2_mean_d_eff:          float  # mean d_eff of RRP nodes in bridge graph
    tier2_regime_counts:       dict
    tier2_verdict:             str   # WELL-INTEGRATED | PARTIAL | ISOLATED
    tier2_strongest_bridges:   list  # list[BridgeEntry], top by similarity
    tier2_anchor_load:         list  # list[AnchorLoad], sorted by n_bridges desc
    tier2_uncovered_count:     int   # RRP entries with no bridge above min_sim

    # Formality (Phase 4.3)
    formality_weight:    float = 1.0    # mean formality weight of bridged anchors
    formality_breakdown: dict  = field(default_factory=dict)  # {tier: count}

    # Overall
    pfd_score:   float   = 0.0   # 0.0–1.0
    summary:     str     = ""    # 1–2 sentence natural language summary

    def as_dict(self) -> dict:
        d = asdict(self)
        # Convert nested dataclasses that asdict already handles
        return d

    def as_text(self) -> str:
        bar = "━" * 57
        thin = "─" * 57

        def _pct(n, d):
            return f"{n / max(d, 1):.1%}"

        # ── Header ────────────────────────────────────────────────────────────
        lines = [
            bar,
            "  PFD Diagnostic Report",
            bar,
            f"  RRP Universe : {self.rrp_name}",
            f"  RRP DB       : {self.rrp_db}",
            f"  DS Wiki DB   : {self.wiki_db}",
            f"  Run date     : {self.run_date}",
            f"  Kernel       : exponential  alpha={self.alpha}",
            bar,
        ]

        # ── Tier 1 ────────────────────────────────────────────────────────────
        lines += [
            "",
            "  TIER 1 — INTERNAL CONSISTENCY",
            thin,
            f"  Entries analyzed  : {self.tier1_n_analyzed} / {self.tier1_n_total}"
            f"  ({_pct(self.tier1_n_analyzed, self.tier1_n_total)} connectable)",
            f"  Internal coherence: {self.tier1_coherence:.1%}",
            f"  Mean d_eff        : {self.tier1_mean_d_eff:.2f}",
            f"  Isolation rate    : {self.tier1_isolation_rate:.1%}"
            f"  ({self.tier1_n_total - self.tier1_n_analyzed} entries with degree < 2)",
        ]

        rc = self.tier1_regime_counts
        n1 = self.tier1_n_analyzed
        lines.append(
            f"  Regime dist       : "
            f"{rc.get('radial_dominated', 0)} radial | "
            f"{rc.get('isotropic', 0)} isotropic | "
            f"{rc.get('noise_dominated', 0)} noise"
        )

        if self.tier1_top_hubs:
            lines.append("  Top internal hubs :")
            for h in self.tier1_top_hubs[:5]:
                lines.append(
                    f"    {h['node_id']:<22} d_eff={h['d_eff']}  "
                    f"PR={h['pr']:.2f}  {h['regime'].upper()[:6]}"
                )

        lines += [
            thin,
            f"  VERDICT: {self.tier1_verdict}",
        ]

        # ── Tier 2 ────────────────────────────────────────────────────────────
        lines += [
            "",
            "  TIER 2 — BRIDGE QUALITY  (vs DS Wiki)",
            thin,
            f"  Bridge edges used : {self.tier2_bridge_edges}"
            f"  (similarity ≥ {self.min_sim})",
            f"  RRP nodes bridged : {self.tier2_n_bridged_rrp} / {self.tier2_n_rrp_total}"
            f"  ({_pct(self.tier2_n_bridged_rrp, self.tier2_n_rrp_total)} reach DS Wiki)",
            f"  DS Wiki anchors   : {self.tier2_wiki_anchors_reached} unique nodes reached",
            f"  Mean bridge d_eff : {self.tier2_mean_d_eff:.2f}  (RRP nodes in bridge graph)",
            f"  Uncovered RRP     : {self.tier2_uncovered_count} entries with no bridge",
        ]

        rc2 = self.tier2_regime_counts
        lines.append(
            f"  Bridge regime dist: "
            f"{rc2.get('radial_dominated', 0)} radial | "
            f"{rc2.get('isotropic', 0)} isotropic | "
            f"{rc2.get('noise_dominated', 0)} noise"
        )

        if self.tier2_strongest_bridges:
            lines.append("  Strongest bridges :")
            for b in self.tier2_strongest_bridges[:5]:
                lines.append(
                    f"    {b['rrp_id']:<22} → {b['wiki_id']:<12}"
                    f" sim={b['similarity']:.3f}  {b['regime'].upper()[:6]}"
                )

        if self.tier2_anchor_load:
            lines.append("  Top DS Wiki anchors (most RRP bridges):")
            for a in self.tier2_anchor_load[:5]:
                lines.append(
                    f"    {a['wiki_id']:<16} {a['n_bridges']} RRP entries"
                )

        # Formality breakdown (Phase 4.3)
        if self.formality_breakdown:
            t1c = self.formality_breakdown.get(1, 0)
            t2c = self.formality_breakdown.get(2, 0)
            t3c = self.formality_breakdown.get(3, 0)
            lines.append(
                f"  Formality tiers   : "
                f"{t1c} Tier-1 | {t2c} Tier-2 | {t3c} Tier-3"
                f"  (weight={self.formality_weight:.3f})"
            )

        lines += [
            thin,
            f"  VERDICT: {self.tier2_verdict}",
        ]

        # ── Overall ───────────────────────────────────────────────────────────
        lines += [
            "",
            bar,
            f"  PFD SCORE : {self.pfd_score:.3f} / 1.000",
            f"  Summary   : {self.summary}",
            bar,
        ]

        return "\n".join(lines)


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    rrp_db:  Path,
    wiki_db: Optional[Path] = None,
    alpha:   float = 1.0,
    min_sim: float = 0.75,
    top_n:   int   = 10,
) -> PFDReport:
    """
    Run Steps 3 and 5 of the PFD pipeline and produce a PFDReport.

    Step 3: sweep_graph on the RRP internal graph → Tier-1 data
    Step 5: sweep_graph on the bridge graph → Tier-2 data

    Args:
        rrp_db:  Path to the RRP bundle SQLite DB.
        wiki_db: Path to ds_wiki.db (defaults to project SOURCE_DB).
        alpha:   Exponential kernel decay (default 1.0).
        min_sim: Minimum bridge similarity to include (default 0.75).
        top_n:   Number of top hubs to surface in each tier (default 10).

    Returns:
        PFDReport with both tiers populated.
    """
    import sys
    from pathlib import Path as _Path
    _SRC = _Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from config import SOURCE_DB

    rrp_db  = Path(rrp_db)
    wiki_db = Path(wiki_db) if wiki_db else Path(SOURCE_DB)
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    kernel   = KernelType.EXPONENTIAL

    # ── Tier 1: internal sweep ────────────────────────────────────────────────
    G_int, _ = build_wiki_graph(rrp_db)
    sweep1   = sweep_graph(G_int, f"rrp_internal:{rrp_db.stem}", kernel, alpha=alpha)

    n1_total    = G_int.number_of_nodes()
    n1_analyzed = sweep1.n_analyzed
    n1_skipped  = sweep1.n_skipped
    isolation   = n1_skipped / max(n1_total, 1)

    noise1     = sweep1.regime_counts.get("noise_dominated", 0)
    coherence  = 1.0 - noise1 / max(n1_analyzed, 1)

    if coherence >= 0.80:
        t1_verdict = "INTERNALLY CONSISTENT"
    elif coherence >= 0.60:
        t1_verdict = "MARGINAL"
    else:
        t1_verdict = "FRAGMENTED"

    t1_hubs = [
        {
            "node_id":       r.node_id,
            "regime":        r.regime.value,
            "d_eff":         r.d_eff,
            "pr":            round(r.pr, 3),
            "eta":           round(r.eta, 3),
            "center_degree": r.center_degree,
        }
        for r in sweep1.top_hubs(n=top_n)
    ]

    # ── Tier 2: bridge sweep ──────────────────────────────────────────────────
    G_br, node_source = build_bridge_graph(rrp_db, wiki_db, min_bridge_similarity=min_sim)
    sweep2 = sweep_graph(G_br, f"bridge:{rrp_db.stem}", kernel, alpha=alpha)

    n_bridge_edges = sum(
        1 for _, _, d in G_br.edges(data=True) if d.get("type") == "bridge"
    )

    rrp_results = [
        r for nid, r in sweep2.results.items()
        if node_source.get(nid) == "rrp" and not r.skipped
    ]
    n2_rrp_total   = sum(1 for v in node_source.values() if v == "rrp")
    n2_bridged     = len(rrp_results)
    n2_uncovered   = n2_rrp_total - n2_bridged

    noise2     = sum(1 for r in rrp_results if r.regime == RegimeType.NOISE_DOMINATED)
    iso2       = sum(1 for r in rrp_results if r.regime == RegimeType.ISOTROPIC)
    bridge_frac = n2_bridged / max(n2_rrp_total, 1)
    noise_frac2 = noise2 / max(n2_bridged, 1)

    if bridge_frac >= 0.70 and noise_frac2 < 0.30:
        t2_verdict = "WELL-INTEGRATED"
    elif bridge_frac >= 0.40:
        t2_verdict = "PARTIAL"
    else:
        t2_verdict = "ISOLATED"

    t2_regime_counts: dict = {}
    for r in rrp_results:
        t2_regime_counts[r.regime.value] = t2_regime_counts.get(r.regime.value, 0) + 1

    t2_mean_deff = sum(r.d_eff for r in rrp_results) / max(n2_bridged, 1)

    # Unique wiki anchors reached
    wiki_anchors_reached = len({
        nid for nid in node_source if node_source[nid] == "wiki"
        and G_br.degree(nid) > 0
        and any(
            node_source.get(nb) == "rrp"
            for nb in G_br.neighbors(nid)
        )
    })

    # Strongest bridges from the DB
    conn_rrp = sqlite3.connect(rrp_db)
    bridge_rows = conn_rrp.execute(
        "SELECT rrp_entry_id, ds_entry_id, similarity "
        "FROM cross_universe_bridges "
        "WHERE similarity >= ? "
        "ORDER BY similarity DESC LIMIT ?",
        (min_sim, top_n),
    ).fetchall()
    conn_rrp.close()

    # Build regime lookup for RRP nodes in bridge graph
    rrp_regime_lookup = {
        r.node_id: r.regime.value
        for r in rrp_results
    }

    strongest = [
        {
            "rrp_id":     row[0],
            "wiki_id":    row[1],
            "similarity": round(float(row[2]), 4),
            "regime":     rrp_regime_lookup.get(f"rrp::{row[0]}", "unknown"),
        }
        for row in bridge_rows
    ]

    # Anchor load: how many RRP entries bridge to each wiki node
    anchor_counts: dict = {}
    for _, _, d in G_br.edges(data=True):
        pass  # Can't get u,v and d in one pass without unpacking differently

    anchor_counts = {}
    for u, v, d in G_br.edges(data=True):
        if d.get("type") == "bridge":
            wiki_nid = v if node_source.get(v) == "wiki" else u
            anchor_counts[wiki_nid] = anchor_counts.get(wiki_nid, 0) + 1

    anchor_load = sorted(
        [{"wiki_id": nid.replace("wiki::", ""), "n_bridges": cnt}
         for nid, cnt in anchor_counts.items()],
        key=lambda x: -x["n_bridges"],
    )[:top_n]

    # ── Formality tier integration (Phase 4.3) ────────────────────────────────
    # Load formality tiers from DS Wiki
    wiki_conn = sqlite3.connect(wiki_db)
    wiki_cols = {row[1] for row in wiki_conn.execute("PRAGMA table_info(entries)")}
    formality_tiers: dict[str, int] = {}
    if "formality_tier" in wiki_cols:
        formality_tiers = {
            row[0]: row[1]
            for row in wiki_conn.execute("SELECT id, formality_tier FROM entries")
            if row[1] is not None
        }
    wiki_conn.close()

    # Compute mean formality weight across all bridged DS Wiki anchors
    _fw_map = {1: 1.0, 2: 0.85, 3: 0.70}
    anchor_tiers = [formality_tiers.get(nid.replace("wiki::", ""), 2)
                    for nid in anchor_counts]
    formality_breakdown_counts: dict[int, int] = {}
    for t in anchor_tiers:
        formality_breakdown_counts[t] = formality_breakdown_counts.get(t, 0) + 1
    anchor_weights = [_fw_map.get(t, 0.85) for t in anchor_tiers]
    mean_fw = sum(anchor_weights) / max(len(anchor_weights), 1) if anchor_weights else 1.0

    # ── PFD score ─────────────────────────────────────────────────────────────
    tier1_score = coherence
    tier2_raw   = bridge_frac * (1.0 - noise_frac2)
    tier2_score = tier2_raw * mean_fw
    pfd_score   = round(0.5 * tier1_score + 0.5 * tier2_score, 4)

    # ── Natural language summary ──────────────────────────────────────────────
    summary = _build_summary(
        rrp_name=rrp_db.stem,
        t1_verdict=t1_verdict,
        t2_verdict=t2_verdict,
        pfd_score=pfd_score,
        coherence=coherence,
        bridge_frac=bridge_frac,
        iso2=iso2,
        n2_bridged=n2_bridged,
    )

    return PFDReport(
        rrp_name=rrp_db.stem,
        rrp_db=str(rrp_db),
        wiki_db=str(wiki_db),
        run_date=run_date,
        alpha=alpha,
        min_sim=min_sim,
        # Tier-1
        tier1_n_analyzed=n1_analyzed,
        tier1_n_total=n1_total,
        tier1_coherence=round(coherence, 4),
        tier1_mean_d_eff=round(sweep1.mean_d_eff, 4),
        tier1_isolation_rate=round(isolation, 4),
        tier1_regime_counts=sweep1.regime_counts,
        tier1_verdict=t1_verdict,
        tier1_top_hubs=t1_hubs,
        # Tier-2
        tier2_bridge_edges=n_bridge_edges,
        tier2_n_bridged_rrp=n2_bridged,
        tier2_n_rrp_total=n2_rrp_total,
        tier2_wiki_anchors_reached=wiki_anchors_reached,
        tier2_mean_d_eff=round(t2_mean_deff, 4),
        tier2_regime_counts=t2_regime_counts,
        tier2_verdict=t2_verdict,
        tier2_strongest_bridges=strongest,
        tier2_anchor_load=anchor_load,
        tier2_uncovered_count=n2_uncovered,
        # Formality (Phase 4.3)
        formality_weight=round(mean_fw, 4),
        formality_breakdown=formality_breakdown_counts,
        # Overall
        pfd_score=pfd_score,
        summary=summary,
    )


def _build_summary(
    rrp_name: str,
    t1_verdict: str,
    t2_verdict: str,
    pfd_score: float,
    coherence: float,
    bridge_frac: float,
    iso2: int,
    n2_bridged: int,
) -> str:
    """Generate a 1–2 sentence natural language summary of the PFD report."""
    t1_adj = {
        "INTERNALLY CONSISTENT": "internally consistent",
        "MARGINAL":              "marginally structured",
        "FRAGMENTED":            "structurally fragmented",
    }.get(t1_verdict, "analyzed")

    t2_adj = {
        "WELL-INTEGRATED": "well integrated into the DS Wiki formal foundation",
        "PARTIAL":         "partially connected to the DS Wiki formal foundation",
        "ISOLATED":        "largely isolated from the DS Wiki formal foundation",
    }.get(t2_verdict, "evaluated against DS Wiki")

    iso_note = ""
    if n2_bridged > 0:
        iso_frac = iso2 / n2_bridged
        if iso_frac >= 0.25:
            iso_note = (
                f"  {iso2} bridged entries ({iso_frac:.0%}) are isotropic — "
                "genuinely instantiating multiple independent formal principles."
            )

    return (
        f"{rrp_name} is {t1_adj} ({coherence:.0%} non-noise) "
        f"and {t2_adj} ({bridge_frac:.0%} of entries reach DS Wiki above threshold). "
        f"PFD score: {pfd_score:.3f}."
        + (" " + iso_note.strip() if iso_note else "")
    )
