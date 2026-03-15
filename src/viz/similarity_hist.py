"""
similarity_hist.py — Cross-universe bridge similarity distribution.

Generates a histogram of cosine similarity scores for all bridges in an RRP
bundle. Bars are stacked and coloured by confidence tier so the tier-1.5
(sim ≥ 0.85) vs tier-2 signal is immediately visible.

Output: similarity_hist.png (matplotlib, static) +
        similarity_hist.html (plotly, interactive, self-contained, offline)
"""

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # headless — must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz._db import load_bridges, load_bundle_name, load_bridge_stats


# ── Constants ─────────────────────────────────────────────────────────────────

BIN_WIDTH   = 0.005
SIM_MIN     = 0.74    # slight left-pad so the leftmost bar isn't cut off
SIM_MAX     = 0.91    # slight right-pad
TIER_1_5_SIM = 0.85   # "analogous to" threshold

COLOR_TIER_2   = "#4393c3"   # steel blue — "couples to"
COLOR_TIER_1_5 = "#d6604d"   # warm red-orange — "analogous to"
COLOR_ANNOT    = "#555555"

ANNOTATION_LINES = [
    (TIER_1_5_SIM, "Tier 1.5 threshold (≥0.85)", COLOR_TIER_1_5, "--"),
    (0.82,         "Network viz default (≥0.82)",  COLOR_ANNOT,    ":"),
    (0.75,         "Storage floor (≥0.75)",         "#aaaaaa",      ":"),
]


# ── Generator class ───────────────────────────────────────────────────────────

class SimilarityHist:
    """
    Generate similarity distribution histogram for an RRP bundle.

    Usage:
        result = SimilarityHist("data/rrp/zoo_classes/rrp_zoo_classes.db").generate(
            output_dir="data/viz/zoo_classes",
        )
        # result: {"png": Path, "html": Path, "stats": dict}
    """

    def __init__(self, bundle_db: str | Path):
        self.bundle_db   = Path(bundle_db)
        self.bundle_name = load_bundle_name(self.bundle_db)

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        output_dir:    str | Path,
        sim_threshold: float = 0.75,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bridges = load_bridges(self.bundle_db, sim_threshold)
        stats   = load_bridge_stats(self.bundle_db)

        sims_t2  = [b.similarity for b in bridges if b.confidence_tier != "1.5"]
        sims_t15 = [b.similarity for b in bridges if b.confidence_tier == "1.5"]

        # Extend stats with filtered counts
        stats["filtered_total"]  = len(bridges)
        stats["sim_threshold"]   = sim_threshold

        png_path  = output_dir / "similarity_hist.png"
        html_path = output_dir / "similarity_hist.html"

        self._generate_png(sims_t2, sims_t15, stats, png_path, sim_threshold)
        self._generate_html(sims_t2, sims_t15, stats, html_path)

        print(f"  similarity_hist  →  {png_path.name}  {html_path.name}")
        return {"png": png_path, "html": html_path, "stats": stats}

    # ── PNG (matplotlib) ──────────────────────────────────────────────────────

    def _generate_png(
        self,
        sims_t2:       list[float],
        sims_t15:      list[float],
        stats:         dict,
        path:          Path,
        sim_threshold: float = 0.75,
    ) -> None:
        bins = np.arange(SIM_MIN, SIM_MAX + BIN_WIDTH, BIN_WIDTH)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Stacked bars: tier-2 base, tier-1.5 on top
        counts_t2,  edges = np.histogram(sims_t2,  bins=bins)
        counts_t15, _     = np.histogram(sims_t15, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        ax.bar(centers, counts_t2,  width=BIN_WIDTH * 0.85,
               color=COLOR_TIER_2,   label='Tier 2 — couples to',   alpha=0.9)
        ax.bar(centers, counts_t15, width=BIN_WIDTH * 0.85,
               color=COLOR_TIER_1_5, label='Tier 1.5 — analogous to',
               bottom=counts_t2, alpha=0.9)

        # Annotation lines
        y_max = ax.get_ylim()[1]
        for x, label, color, ls in ANNOTATION_LINES:
            if x >= sim_threshold:
                ax.axvline(x, color=color, linestyle=ls, linewidth=1.4, alpha=0.85)
                ax.text(x + 0.001, y_max * 0.97, label,
                        fontsize=7, color=color, va='top', ha='left')

        # Stats box
        t15_pct = 100 * stats["tier_1_5"] / max(stats["total"], 1)
        stats_text = (
            f"n = {stats['total']:,} bridges\n"
            f"mean = {stats['mean_sim']:.4f}\n"
            f"median = {stats['median_sim']:.4f}\n"
            f"tier 1.5 (≥0.85): {stats['tier_1_5']} ({t15_pct:.1f}%)"
        )
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))

        ax.set_xlabel("Cosine Similarity", fontsize=11)
        ax.set_ylabel("Bridge Count", fontsize=11)
        ax.set_title(
            f"{self.bundle_name}: Cross-Universe Bridge Similarity Distribution",
            fontsize=12, fontweight='bold', pad=10,
        )
        ax.set_xlim(SIM_MIN, SIM_MAX)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── HTML (plotly) ─────────────────────────────────────────────────────────

    def _generate_html(
        self,
        sims_t2:  list[float],
        sims_t15: list[float],
        stats:    dict,
        path:     Path,
    ) -> None:
        import plotly.graph_objects as go

        bins = dict(start=SIM_MIN, end=SIM_MAX, size=BIN_WIDTH)

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=sims_t2,
            xbins=bins,
            name="Tier 2 — couples to",
            marker_color=COLOR_TIER_2,
            opacity=0.9,
        ))
        fig.add_trace(go.Histogram(
            x=sims_t15,
            xbins=bins,
            name="Tier 1.5 — analogous to",
            marker_color=COLOR_TIER_1_5,
            opacity=0.9,
        ))

        fig.update_layout(barmode="stack")

        # Annotation lines
        t15_pct = 100 * stats["tier_1_5"] / max(stats["total"], 1)
        for x, label, color, _ in ANNOTATION_LINES:
            fig.add_vline(
                x=x, line_dash="dash", line_color=color, line_width=1.5,
                annotation_text=label,
                annotation_font_size=10,
                annotation_font_color=color,
            )

        stats_text = (
            f"n={stats['total']:,}  mean={stats['mean_sim']:.4f}  "
            f"median={stats['median_sim']:.4f}  "
            f"tier 1.5: {stats['tier_1_5']} ({t15_pct:.1f}%)"
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            text=stats_text,
            showarrow=False,
            font=dict(size=11),
            bgcolor="white",
            bordercolor="#cccccc",
            borderwidth=1,
            align="left",
        )

        fig.update_layout(
            title=dict(
                text=f"{self.bundle_name}: Cross-Universe Bridge Similarity Distribution",
                font=dict(size=14),
            ),
            xaxis_title="Cosine Similarity",
            yaxis_title="Bridge Count",
            xaxis=dict(range=[SIM_MIN, SIM_MAX]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=500,
            width=900,
        )
        fig.update_xaxes(gridcolor="#eeeeee")
        fig.update_yaxes(gridcolor="#eeeeee")

        fig.write_html(
            str(path),
            include_plotlyjs=True,
            full_html=True,
            config={
                "displayModeBar": True,
                "scrollZoom": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "similarity_hist",
                    "height": 500, "width": 900, "scale": 2,
                },
            },
        )
