"""
domain_heatmap.py — Cross-universe bridge density heatmap.

Rows    = RRP source type (theorems / classes / conjectures / problems)
Columns = DS Wiki type_group (RL, Q, X, H, M, T, B, F, E, Ax, Other...)

Cell values = count of bridges at sim >= sim_threshold.
Sparse type_groups (< MIN_COL_TOTAL total bridges) are collapsed into "Other".

Output: domain_heatmap.png  (matplotlib, static)
        domain_heatmap.html (plotly, interactive, self-contained)
"""

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz._db import load_bridges, load_ds_entry_meta, load_bundle_name

try:
    from config import SOURCE_DB
except ImportError:
    SOURCE_DB = None


# ── Constants ─────────────────────────────────────────────────────────────────

# Row order: RRP source types, descending by typical entry count
ROW_ORDER   = ["theorems", "classes", "conjectures", "problems"]
ROW_LABELS  = {
    "theorems":    "Theorems",
    "classes":     "Classes",
    "conjectures": "Conjectures",
    "problems":    "Problems",
}

MIN_COL_TOTAL = 5     # type_groups with fewer total bridges → collapsed to "Other"
COLORMAP      = "YlOrRd"


# ── Generator class ───────────────────────────────────────────────────────────

class DomainHeatmap:
    """
    Generate bridge-density heatmap for an RRP bundle.

    Usage:
        result = DomainHeatmap(
            "data/rrp/zoo_classes/rrp_zoo_classes.db",
            ds_wiki_db="data/ds_wiki.db",
        ).generate("data/viz/zoo_classes")
        # result: {"png": Path, "html": Path, "stats": dict}
    """

    def __init__(
        self,
        bundle_db:  str | Path,
        ds_wiki_db: str | Path | None = None,
    ):
        self.bundle_db   = Path(bundle_db)
        self.ds_wiki_db  = Path(ds_wiki_db) if ds_wiki_db else (SOURCE_DB and Path(SOURCE_DB))
        self.bundle_name = load_bundle_name(self.bundle_db)

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        output_dir:    str | Path,
        sim_threshold: float = 0.75,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bridges  = load_bridges(self.bundle_db, sim_threshold)
        matrix, rows, cols, col_totals = self._build_matrix(bridges)

        stats = {
            "rows":          len(rows),
            "cols":          len(cols),
            "total_bridges": len(bridges),
            "sim_threshold": sim_threshold,
        }

        png_path  = output_dir / "domain_heatmap.png"
        html_path = output_dir / "domain_heatmap.html"

        self._generate_png(matrix, rows, cols, stats, png_path, sim_threshold)
        self._generate_html(matrix, rows, cols, col_totals, stats, html_path, sim_threshold)

        print(f"  domain_heatmap   →  {png_path.name}  {html_path.name}")
        return {"png": png_path, "html": html_path, "stats": stats}

    # ── Matrix builder ────────────────────────────────────────────────────────

    def _build_matrix(
        self,
        bridges: list,
    ) -> tuple[np.ndarray, list[str], list[str], dict]:
        """
        Build the 2D bridge count matrix.
        Returns (matrix, row_labels, col_labels, col_totals).
        Sparse columns (total < MIN_COL_TOTAL) are collapsed into 'Other'.
        """
        # Resolve DS Wiki type_group for each bridge
        ds_ids = list({b.ds_entry_id for b in bridges})
        if self.ds_wiki_db and self.ds_wiki_db.exists():
            ds_meta = load_ds_entry_meta(self.ds_wiki_db, ds_ids)
        else:
            # Fallback: derive type_group from entry_id prefix
            ds_meta = {eid: _stub_meta(eid) for eid in ds_ids}

        # Count bridges per (source_type, type_group)
        raw_counts: dict[tuple[str, str], int] = {}
        for b in bridges:
            src = b.rrp_source_type
            tg  = ds_meta[b.ds_entry_id].type_group if b.ds_entry_id in ds_meta else "?"
            key = (src, tg)
            raw_counts[key] = raw_counts.get(key, 0) + 1

        # Compute column totals (over all source types)
        col_totals: dict[str, int] = {}
        for (src, tg), cnt in raw_counts.items():
            col_totals[tg] = col_totals.get(tg, 0) + cnt

        # Separate main columns from sparse → "Other"
        main_cols = sorted(
            [tg for tg, tot in col_totals.items() if tot >= MIN_COL_TOTAL and tg != "?"],
            key=lambda tg: -col_totals[tg],
        )
        sparse_tgs = {tg for tg in col_totals if tg not in main_cols}

        # Remap sparse type_groups to "Other"
        remapped: dict[tuple[str, str], int] = {}
        for (src, tg), cnt in raw_counts.items():
            new_tg = tg if tg in main_cols else "Other"
            key = (src, new_tg)
            remapped[key] = remapped.get(key, 0) + cnt

        # Recompute col_totals after remapping (for "Other" column ordering)
        final_col_totals: dict[str, int] = {}
        for (src, tg), cnt in remapped.items():
            final_col_totals[tg] = final_col_totals.get(tg, 0) + cnt

        # Column order: main cols by count, then "Other" at end
        cols = main_cols[:]
        if "Other" in final_col_totals and final_col_totals["Other"] > 0:
            cols.append("Other")

        # Row order: use ROW_ORDER, but only include rows that appear in the data
        src_types_present = {src for (src, _) in remapped}
        rows = [r for r in ROW_ORDER if r in src_types_present]
        # Append any unexpected source types at the end
        for src in sorted(src_types_present):
            if src not in rows:
                rows.append(src)

        # Build 2D numpy array (rows × cols)
        matrix = np.zeros((len(rows), len(cols)), dtype=int)
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                matrix[i, j] = remapped.get((row, col), 0)

        return matrix, rows, cols, final_col_totals

    # ── PNG (matplotlib) ──────────────────────────────────────────────────────

    def _generate_png(
        self,
        matrix:        np.ndarray,
        rows:          list[str],
        cols:          list[str],
        stats:         dict,
        path:          Path,
        sim_threshold: float,
    ) -> None:
        n_rows, n_cols = len(rows), len(cols)
        fig_w = max(14, n_cols * 1.1)
        fig_h = max(4,  n_rows * 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

        im = ax.imshow(matrix, cmap=COLORMAP, aspect='auto',
                       vmin=0, vmax=matrix.max() or 1)

        # Cell annotations (suppress zeros)
        for i in range(n_rows):
            for j in range(n_cols):
                v = int(matrix[i, j])
                if v > 0:
                    # white text on dark cells, black on light
                    norm_val = v / (matrix.max() or 1)
                    txt_color = "white" if norm_val > 0.6 else "black"
                    ax.text(j, i, str(v), ha='center', va='center',
                            fontsize=8, color=txt_color, fontweight='bold')

        row_labels = [ROW_LABELS.get(r, r) for r in rows]
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)

        ax.set_title(
            f"{self.bundle_name}: Cross-Universe Bridge Density\n"
            f"RRP Source Type × DS Wiki Type Group  (sim ≥ {sim_threshold})",
            fontsize=11, fontweight='bold', pad=10,
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Bridge Count", fontsize=9)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── HTML (plotly) ─────────────────────────────────────────────────────────

    def _generate_html(
        self,
        matrix:     np.ndarray,
        rows:       list[str],
        cols:       list[str],
        col_totals: dict,
        stats:      dict,
        path:       Path,
        sim_threshold: float,
    ) -> None:
        import plotly.graph_objects as go

        row_labels = [ROW_LABELS.get(r, r) for r in rows]

        # Build hover text (row × col)
        hover = []
        for i, row in enumerate(rows):
            row_hover = []
            for j, col in enumerate(cols):
                v = int(matrix[i, j])
                row_hover.append(
                    f"<b>{col}</b> × {ROW_LABELS.get(row, row)}<br>"
                    f"Bridges: <b>{v}</b><br>"
                    f"Col total: {col_totals.get(col, 0)}"
                )
            hover.append(row_hover)

        # Suppress annotations for zero cells
        annot_text = [
            [str(int(matrix[i, j])) if matrix[i, j] > 0 else ""
             for j in range(len(cols))]
            for i in range(len(rows))
        ]

        fig = go.Figure(go.Heatmap(
            z=matrix.tolist(),
            x=cols,
            y=row_labels,
            colorscale=COLORMAP,
            text=annot_text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            hoverongaps=False,
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            colorbar=dict(title=dict(text="Bridge Count", side="right")),
        ))

        fig.update_layout(
            title=dict(
                text=(
                    f"{self.bundle_name}: Cross-Universe Bridge Density<br>"
                    f"<sup>RRP Source Type × DS Wiki Type Group  (sim ≥ {sim_threshold})</sup>"
                ),
                font=dict(size=14),
            ),
            xaxis=dict(
                title="DS Wiki Type Group",
                tickangle=45,
                side="bottom",
            ),
            yaxis=dict(
                title="RRP Source Type",
                autorange="reversed",   # top row first
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=300 + len(rows) * 60,
            width=200  + len(cols) * 75,
            margin=dict(l=120, r=60, t=120, b=100),
        )

        fig.write_html(
            str(path),
            include_plotlyjs=True,
            full_html=True,
            config={
                "displayModeBar": True,
                "toImageButtonOptions": {
                    "format": "png", "filename": "domain_heatmap",
                    "scale": 2,
                },
            },
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _stub_meta(entry_id: str):
    """Derive a minimal DSEntryMeta stub from the entry_id prefix when ds_wiki.db is unavailable."""
    from viz._db import DSEntryMeta
    # Strip trailing digits to get prefix: CS4 → CS, MATH3 → MATH
    import re
    prefix = re.sub(r'\d+$', '', entry_id).rstrip('_')
    return DSEntryMeta(
        entry_id   = entry_id,
        title      = entry_id,
        type_group = prefix or "?",
        domain     = "?",
    )
