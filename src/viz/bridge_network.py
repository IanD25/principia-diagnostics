"""
bridge_network.py — Cross-universe bridge network visualization.

Bipartite graph: DS Wiki entries (left) ↔ RRP entries (right).
Edges = cross-universe bridges, coloured by link type, width by similarity.

Default threshold 0.82 keeps ~273 nodes / ~432 edges — the readable zone.
Below 0.80 the graph becomes a hairball; use the heatmap or histogram instead.

Output: bridge_network.png  (matplotlib + networkx, static, 16×20 in)
        bridge_network.html (plotly, interactive, self-contained, offline)
"""

import sys
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz._db import load_bridges, load_ds_entry_meta, load_bundle_name

try:
    from config import SOURCE_DB
except ImportError:
    SOURCE_DB = None


# ── Visual constants ──────────────────────────────────────────────────────────

# DS Wiki type_group → colour (fixed palette, 15 groups)
TYPE_GROUP_COLORS: dict[str, str] = {
    "RL":  "#1f77b4",  # blue
    "T":   "#ff7f0e",  # orange
    "X":   "#2ca02c",  # green
    "Q":   "#d62728",  # red
    "H":   "#9467bd",  # purple
    "M":   "#8c564b",  # brown
    "F":   "#e377c2",  # pink
    "B":   "#7f7f7f",  # gray
    "Ax":  "#bcbd22",  # yellow-green
    "A":   "#17becf",  # teal
    "E":   "#98df8a",  # light green
    "G":   "#ff9896",  # light red
    "OmD": "#c5b0d5",  # light purple
    "?":   "#dddddd",  # fallback
}

# RRP source_type → colour
RRP_SRC_COLORS: dict[str, str] = {
    "theorems":    "#4393c3",   # cool blue
    "classes":     "#f4a582",   # warm salmon
    "conjectures": "#b2182b",   # dark red
    "problems":    "#74c476",   # green
    "unknown":     "#cccccc",
}

# Edge link_type → colour
LINK_TYPE_COLORS: dict[str, str] = {
    "analogous to": "#7b2d8b",   # dark purple (high-sim tier 1.5)
    "couples to":   "#6baed6",   # light blue  (tier 2)
    "related":      "#aaaaaa",   # gray
}

# Similarity → plotly edge width tier
def _edge_width_tier(sim: float) -> tuple[float, str]:
    """Return (line_width, bucket_label) based on similarity."""
    if sim >= 0.88:
        return 3.0, "high"
    elif sim >= 0.83:
        return 1.8, "mid"
    else:
        return 0.8, "low"


# ── Generator class ───────────────────────────────────────────────────────────

class BridgeNetwork:
    """
    Generate a bipartite bridge network visualization.

    NOTE: Default sim_threshold = 0.82 (not 0.75 like the other viz tools).
    At 0.80 the graph has ~969 edges — a dense hairball.
    At 0.82 it has ~432 edges — readable in both PNG and HTML.
    Use similarity_hist or domain_heatmap to explore the full 0.75+ set.

    Usage:
        result = BridgeNetwork(
            "data/rrp/zoo_classes/rrp_zoo_classes.db",
            ds_wiki_db="data/ds_wiki.db",
        ).generate("data/viz/zoo_classes", sim_threshold=0.82)
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
        sim_threshold: float = 0.82,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bridges = load_bridges(self.bundle_db, sim_threshold)

        if not bridges:
            print(f"  bridge_network   →  no bridges above {sim_threshold:.2f}")
            return {"png": None, "html": None, "stats": {"edges": 0}}

        # Resolve DS Wiki metadata
        ds_ids = list({b.ds_entry_id for b in bridges})
        if self.ds_wiki_db and self.ds_wiki_db.exists():
            ds_meta = load_ds_entry_meta(self.ds_wiki_db, ds_ids)
        else:
            ds_meta = {eid: _stub_meta(eid) for eid in ds_ids}

        G, pos, ds_nodes, rrp_nodes = self._build_graph(bridges, ds_meta)

        stats = {
            "rrp_nodes":     len(rrp_nodes),
            "ds_nodes":      len(ds_nodes),
            "edges":         len(bridges),
            "sim_threshold": sim_threshold,
        }

        png_path  = output_dir / "bridge_network.png"
        html_path = output_dir / "bridge_network.html"

        self._generate_png(G, pos, ds_nodes, rrp_nodes, ds_meta, bridges, stats, png_path)
        self._generate_html(pos, ds_nodes, rrp_nodes, ds_meta, bridges, stats, html_path, sim_threshold)

        print(f"  bridge_network   →  {png_path.name}  {html_path.name}  "
              f"({stats['ds_nodes']} DS + {stats['rrp_nodes']} RRP nodes, {stats['edges']} edges)")
        return {"png": png_path, "html": html_path, "stats": stats}

    # ── Graph builder ─────────────────────────────────────────────────────────

    def _build_graph(self, bridges, ds_meta):
        """
        Build a bipartite networkx graph and compute deterministic two-column layout.
        DS nodes: x=0.0, sorted by bridge count desc, evenly spaced vertically.
        RRP nodes: x=1.0, y = mean y of their DS neighbours (minimises crossings).
        """
        G = nx.Graph()

        # Count bridges per DS node (for size + sort order)
        ds_bridge_count: dict[str, int] = {}
        for b in bridges:
            ds_bridge_count[b.ds_entry_id] = ds_bridge_count.get(b.ds_entry_id, 0) + 1

        # Collect unique node IDs (preserve order: DS sorted by count desc)
        ds_nodes  = sorted(ds_bridge_count, key=lambda x: -ds_bridge_count[x])
        rrp_set   = {b.rrp_entry_id for b in bridges}
        rrp_nodes = list(rrp_set)  # will be sorted after layout

        G.add_nodes_from(ds_nodes,  bipartite=0)
        G.add_nodes_from(rrp_nodes, bipartite=1)

        for b in bridges:
            G.add_edge(b.rrp_entry_id, b.ds_entry_id,
                       similarity         = b.similarity,
                       proposed_link_type = b.proposed_link_type,
                       confidence_tier    = b.confidence_tier)

        # ── Two-column bipartite layout ────────────────────────────────────
        pos: dict[str, tuple[float, float]] = {}

        n_ds = len(ds_nodes)
        for i, nid in enumerate(ds_nodes):
            pos[nid] = (0.0, i / max(n_ds - 1, 1))

        # RRP y = mean(y of DS neighbours)
        for nid in rrp_nodes:
            ds_nbrs = [nb for nb in G.neighbors(nid) if nb in pos]
            if ds_nbrs:
                pos[nid] = (1.0, sum(pos[nb][1] for nb in ds_nbrs) / len(ds_nbrs))
            else:
                pos[nid] = (1.0, 0.5)

        # Re-sort RRP nodes by their y position for consistent ordering
        rrp_nodes = sorted(rrp_nodes, key=lambda nid: pos[nid][1])

        return G, pos, ds_nodes, rrp_nodes

    # ── PNG (matplotlib) ──────────────────────────────────────────────────────

    def _generate_png(
        self,
        G, pos, ds_nodes, rrp_nodes, ds_meta, bridges, stats, path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(16, 20), dpi=150)
        ax.set_axis_off()

        # Build per-node look-up tables from bridges
        ds_bridge_count = {}
        rrp_src_map     = {}
        for b in bridges:
            ds_bridge_count[b.ds_entry_id] = ds_bridge_count.get(b.ds_entry_id, 0) + 1
            rrp_src_map[b.rrp_entry_id]    = b.rrp_source_type

        # Draw edges (thin, alpha)
        for b in bridges:
            x0, y0 = pos[b.rrp_entry_id]
            x1, y1 = pos[b.ds_entry_id]
            color  = LINK_TYPE_COLORS.get(b.proposed_link_type, "#aaaaaa")
            lw     = _edge_width_tier(b.similarity)[0] * 0.5  # halve for PNG density
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.35, zorder=1)

        # Draw DS nodes (squares, left column)
        for nid in ds_nodes:
            x, y   = pos[nid]
            cnt    = ds_bridge_count.get(nid, 1)
            tg     = ds_meta[nid].type_group if nid in ds_meta else "?"
            color  = TYPE_GROUP_COLORS.get(tg, TYPE_GROUP_COLORS["?"])
            size   = min(30, 8 + 4 * math.sqrt(cnt))
            ax.scatter(x, y, s=size**2 * 0.5, c=color, marker='s',
                       zorder=3, edgecolors='white', linewidths=0.5)
            label = f"{nid}: {(ds_meta[nid].title[:22] if nid in ds_meta else nid)}"
            ax.text(x - 0.02, y, label, ha='right', va='center',
                    fontsize=5.5, color='#333333', zorder=4)

        # Draw RRP nodes (circles, right column) — no individual labels (too dense)
        for nid in rrp_nodes:
            x, y  = pos[nid]
            src   = rrp_src_map.get(nid, "unknown")
            color = RRP_SRC_COLORS.get(src, RRP_SRC_COLORS["unknown"])
            ax.scatter(x, y, s=30, c=color, marker='o',
                       zorder=3, edgecolors='white', linewidths=0.4, alpha=0.7)

        # ── Legend ────────────────────────────────────────────────────────
        handles = []

        # DS type_group colours
        tgs_present = {ds_meta[n].type_group for n in ds_nodes if n in ds_meta}
        for tg in sorted(tgs_present):
            c = TYPE_GROUP_COLORS.get(tg, TYPE_GROUP_COLORS["?"])
            handles.append(mpatches.Patch(color=c, label=f"DS {tg}"))

        handles.append(mlines.Line2D([], [], color='none', label=''))   # spacer

        # RRP source type colours
        srcs_present = {rrp_src_map.get(n, "unknown") for n in rrp_nodes}
        for src in ["theorems", "classes", "conjectures", "problems", "unknown"]:
            if src in srcs_present:
                c = RRP_SRC_COLORS.get(src, RRP_SRC_COLORS["unknown"])
                handles.append(mpatches.Patch(color=c, label=f"RRP {src}"))

        handles.append(mlines.Line2D([], [], color='none', label=''))   # spacer

        # Edge link types
        for lt, c in LINK_TYPE_COLORS.items():
            handles.append(mlines.Line2D([0], [0], color=c, linewidth=2, label=lt))

        ax.legend(handles=handles, loc='upper right', fontsize=6,
                  framealpha=0.9, ncol=2, borderpad=0.5)

        ax.set_title(
            f"{self.bundle_name}: Cross-Universe Bridge Network\n"
            f"{stats['ds_nodes']} DS Wiki nodes  ×  {stats['rrp_nodes']} RRP nodes  "
            f"({stats['edges']} bridges, sim ≥ {stats['sim_threshold']})",
            fontsize=10, fontweight='bold',
        )

        ax.set_xlim(-0.35, 1.20)
        ax.set_ylim(-0.03, 1.03)

        # Column headers
        ax.text(0.0, 1.025, "← DS Wiki (science universe)", ha='center',
                fontsize=8, color='#555555', style='italic',
                transform=ax.transData)
        ax.text(1.0, 1.025, "RRP entries (research universe) →", ha='center',
                fontsize=8, color='#555555', style='italic',
                transform=ax.transData)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # ── HTML (plotly) ─────────────────────────────────────────────────────────

    def _generate_html(
        self,
        pos, ds_nodes, rrp_nodes, ds_meta, bridges, stats, path, sim_threshold,
    ) -> None:
        import plotly.graph_objects as go

        ds_bridge_count: dict[str, int] = {}
        rrp_src_map:     dict[str, str] = {}
        rrp_title_map:   dict[str, str] = {}
        for b in bridges:
            ds_bridge_count[b.ds_entry_id] = ds_bridge_count.get(b.ds_entry_id, 0) + 1
            rrp_src_map[b.rrp_entry_id]    = b.rrp_source_type
            rrp_title_map[b.rrp_entry_id]  = b.rrp_entry_title

        traces = []

        # ── Edge traces: 2 link types × 3 width tiers = up to 6 traces ────
        edge_groups: dict[tuple[str, str], list] = {}
        for b in bridges:
            lt       = b.proposed_link_type
            _, tier  = _edge_width_tier(b.similarity)
            key      = (lt, tier)
            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append(b)

        for (lt, tier), group in edge_groups.items():
            ex, ey = [], []
            for b in group:
                x0, y0 = pos[b.rrp_entry_id]
                x1, y1 = pos[b.ds_entry_id]
                ex += [x0, x1, None]
                ey += [y0, y1, None]
            lw    = {"high": 2.5, "mid": 1.4, "low": 0.6}[tier]
            color = LINK_TYPE_COLORS.get(lt, "#aaaaaa")
            traces.append(go.Scatter(
                x=ex, y=ey, mode='lines',
                line=dict(width=lw, color=color),
                opacity=0.35,
                hoverinfo='none',
                showlegend=False,
                name=f"{lt} ({tier})",
            ))

        # ── DS nodes: one trace per type_group (enables legend toggling) ──
        tgs_present = sorted({ds_meta[n].type_group for n in ds_nodes if n in ds_meta})
        for tg in tgs_present:
            nids = [n for n in ds_nodes if (n in ds_meta and ds_meta[n].type_group == tg)]
            xs   = [pos[n][0] for n in nids]
            ys   = [pos[n][1] for n in nids]
            szs  = [min(30, 8 + 4 * math.sqrt(ds_bridge_count.get(n, 1))) for n in nids]
            hover = [
                f"<b>[{n}] {ds_meta[n].title}</b><br>"
                f"Type group: {ds_meta[n].type_group}<br>"
                f"Domain: {ds_meta[n].domain}<br>"
                f"Bridges at threshold: {ds_bridge_count.get(n, 0)}"
                for n in nids
            ]
            color = TYPE_GROUP_COLORS.get(tg, TYPE_GROUP_COLORS["?"])
            traces.append(go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(
                    symbol='square',
                    size=szs,
                    color=color,
                    line=dict(width=1, color='white'),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name=f"DS: {tg}",
                legendgroup=f"ds_{tg}",
            ))

        # ── RRP nodes: one trace per source_type ──────────────────────────
        srcs_present = sorted({rrp_src_map.get(n, "unknown") for n in rrp_nodes})
        for src in srcs_present:
            nids = [n for n in rrp_nodes if rrp_src_map.get(n, "unknown") == src]
            xs   = [pos[n][0] for n in nids]
            ys   = [pos[n][1] for n in nids]
            hover = [
                f"<b>{rrp_title_map.get(n, n)}</b><br>"
                f"Source type: {src}<br>"
                f"ID: {n}"
                for n in nids
            ]
            color = RRP_SRC_COLORS.get(src, RRP_SRC_COLORS["unknown"])
            traces.append(go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(
                    symbol='circle',
                    size=7,
                    color=color,
                    opacity=0.75,
                    line=dict(width=0.5, color='white'),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name=f"RRP: {src}",
                legendgroup=f"rrp_{src}",
            ))

        # ── Legend edge type indicators (dummy traces) ────────────────────
        for lt, color in LINK_TYPE_COLORS.items():
            traces.append(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color=color, width=2),
                name=lt, showlegend=True,
            ))

        # ── Column header annotations ─────────────────────────────────────
        annotations = [
            dict(x=0.0, y=1.02, xref='x', yref='y',
                 text="← DS Wiki (science universe)",
                 showarrow=False, font=dict(size=11, color='#555555'),
                 xanchor='center'),
            dict(x=1.0, y=1.02, xref='x', yref='y',
                 text="RRP entries (research universe) →",
                 showarrow=False, font=dict(size=11, color='#555555'),
                 xanchor='center'),
        ]

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=(
                    f"{self.bundle_name}: Cross-Universe Bridge Network<br>"
                    f"<sup>{stats['ds_nodes']} DS Wiki + {stats['rrp_nodes']} RRP nodes  "
                    f"| {stats['edges']} bridges  |  sim ≥ {sim_threshold}</sup>"
                ),
                font=dict(size=13),
            ),
            xaxis=dict(visible=False, range=[-0.2, 1.2]),
            yaxis=dict(visible=False, range=[-0.03, 1.08]),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation='v', x=1.01, y=1,
                xanchor='left', yanchor='top',
                font=dict(size=10),
                tracegroupgap=4,
            ),
            annotations=annotations,
            height=900,
            width=1200,
            margin=dict(l=20, r=220, t=100, b=20),
        )

        fig.write_html(
            str(path),
            include_plotlyjs=True,
            full_html=True,
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "bridge_network",
                    "height": 900, "width": 1200, "scale": 2,
                },
            },
        )


# ── Utility ───────────────────────────────────────────────────────────────────

def _stub_meta(entry_id: str):
    """Derive a minimal DSEntryMeta stub from the entry_id prefix."""
    from viz._db import DSEntryMeta
    import re
    prefix = re.sub(r'\d+$', '', entry_id).rstrip('_')
    return DSEntryMeta(
        entry_id   = entry_id,
        title      = entry_id,
        type_group = prefix or "?",
        domain     = "?",
    )
