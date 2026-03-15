"""
tier1_dashboard.py — Tier-1 visualization suite.

Generates PNG charts and interactive D3.js graphs for internal RRP analysis:
  - Coherence dashboard (pie + D_eff gauge)
  - Regime distribution bar chart
  - Interactive network topology graph (D3.js)

Usage:
    dashboard = Tier1Dashboard(rrp_db)
    dashboard.generate_coherence_png(output_path)
    dashboard.generate_regime_png(output_path)
    dashboard.generate_network_html(output_path)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from dataclasses import asdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
import networkx as nx
import numpy as np

from analysis.fisher_diagnostics import (
    sweep_graph,
    RegimeType,
)


class Tier1Dashboard:
    """Generates Tier-1 internal RRP visualizations."""

    def __init__(self, rrp_db: str | Path):
        """Initialize with RRP database path.

        Args:
            rrp_db: Path to RRP SQLite database

        Raises:
            FileNotFoundError: If RRP database doesn't exist
        """
        self.rrp_db = Path(rrp_db)
        if not self.rrp_db.exists():
            raise FileNotFoundError(f"RRP database not found: {rrp_db}")

        self._graph = None
        self._sweep = None
        self._regime_counts = None

    @property
    def graph(self) -> nx.DiGraph:
        """Load and cache RRP graph."""
        if self._graph is None:
            conn = sqlite3.connect(self.rrp_db)
            entries = load_rrp_entries(conn)
            links = load_rrp_links(conn)

            G = nx.DiGraph()
            for entry in entries:
                G.add_node(entry["id"], **entry)
            for link in links:
                G.add_edge(
                    link["source_id"],
                    link["target_id"],
                    **link,
                )

            conn.close()
            self._graph = G

        return self._graph

    @property
    def sweep(self):
        """Run sweep analysis and cache result."""
        if self._sweep is None:
            graph_source = f"rrp_internal:{self.rrp_db.stem}"
            self._sweep = sweep_graph(self.graph, graph_source=graph_source, alpha=1.0)
        return self._sweep

    @property
    def regime_counts(self) -> dict:
        """Get regime distribution from sweep result."""
        if self._regime_counts is None:
            # Use pre-computed regime_counts from sweep
            counts = self.sweep.regime_counts.copy()
            # Ensure all keys exist
            self._regime_counts = {
                "radial": counts.get("radial_dominated", 0),
                "isotropic": counts.get("isotropic", 0),
                "noise": counts.get("noise_dominated", 0),
                "degenerate": counts.get("degenerate", 0),
            }

        return self._regime_counts

    # ── Coherence Dashboard ────────────────────────────────────────────────────

    def _compute_coherence(self) -> float:
        """Compute coherence from sweep results (1 - noise_fraction)."""
        noise_count = self.sweep.regime_counts.get("noise_dominated", 0)
        total_analyzed = self.sweep.n_analyzed
        if total_analyzed == 0:
            return 0.0
        return 1.0 - (noise_count / total_analyzed)

    def generate_coherence_png(self, output_path: str | Path) -> Path:
        """Generate coherence dashboard PNG (pie + gauge).

        Args:
            output_path: Output file path

        Returns:
            Path to generated PNG
        """
        fig, (ax_pie, ax_gauge) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Coherence Overview", fontsize=16, fontweight="bold")

        # ── Left: Pie chart (noise vs. signal) ──────────────────────────────

        coherence = self._compute_coherence()
        noise_pct = (1.0 - coherence) * 100
        signal_pct = coherence * 100

        colors = ["#2ECC71", "#F39C12"]  # Green (signal), orange (noise)
        sizes = [signal_pct, noise_pct]
        labels = [f"Signal\n{signal_pct:.1f}%", f"Noise\n{noise_pct:.1f}%"]

        wedges, texts, autotexts = ax_pie.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="",
            startangle=90,
            textprops={"fontsize": 12, "weight": "bold"},
        )

        ax_pie.set_title("Signal vs. Noise", fontsize=13, pad=15)

        # ── Right: D_eff gauge ─────────────────────────────────────────────

        d_eff = self.sweep.mean_d_eff
        max_d_eff = 20.0

        # Draw semicircular gauge
        theta = np.linspace(0, np.pi, 100)
        ax_gauge.plot(np.cos(theta), np.sin(theta), "k-", lw=2)

        # Color gradient zones
        zones = [
            (0, 0.25, "#1E3A8A"),  # Blue (planar)
            (0.25, 0.5, "#3B82F6"),
            (0.5, 0.75, "#60A5FA"),
            (0.75, 1.0, "#DC2626"),  # Red (distributed)
        ]

        for start, end, color in zones:
            theta_zone = np.linspace(start * np.pi, end * np.pi, 30)
            ax_gauge.fill_between(
                np.cos(theta_zone),
                0,
                np.sin(theta_zone),
                alpha=0.3,
                color=color,
            )

        # Draw needle
        needle_angle = (d_eff / max_d_eff) * np.pi
        needle_x = 0.8 * np.cos(needle_angle)
        needle_y = 0.8 * np.sin(needle_angle)
        ax_gauge.arrow(
            0,
            0,
            needle_x,
            needle_y,
            head_width=0.08,
            head_length=0.08,
            fc="red",
            ec="red",
        )

        # Labels
        ax_gauge.text(-0.9, -0.15, "1\n(Planar)", ha="center", fontsize=10)
        ax_gauge.text(0.9, -0.15, f"{max_d_eff}\n(Distributed)", ha="center", fontsize=10)
        ax_gauge.text(0, 0.5, f"D_eff = {d_eff:.2f}", ha="center", fontsize=14, weight="bold")

        ax_gauge.set_xlim(-1.2, 1.2)
        ax_gauge.set_ylim(-0.3, 1.2)
        ax_gauge.set_aspect("equal")
        ax_gauge.axis("off")
        ax_gauge.set_title("Effective Dimensionality", fontsize=13, pad=15)

        plt.tight_layout()
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    # ── Regime Distribution ────────────────────────────────────────────────

    def generate_regime_png(self, output_path: str | Path) -> Path:
        """Generate regime distribution bar chart PNG.

        Args:
            output_path: Output file path

        Returns:
            Path to generated PNG
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        counts = self.regime_counts
        regimes = ["Radial", "Isotropic", "Noise", "Degenerate"]
        values = [
            counts["radial"],
            counts["isotropic"],
            counts["noise"],
            counts["degenerate"],
        ]
        colors = ["#3B82F6", "#06B6D4", "#F59E0B", "#EF4444"]

        bars = ax.bar(regimes, values, color=colors, edgecolor="black", linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=12,
                weight="bold",
            )

        ax.set_ylabel("Number of Nodes", fontsize=12, weight="bold")
        ax.set_title("Network Regime Distribution", fontsize=14, weight="bold", pad=20)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis="y", alpha=0.3)

        # Add total and percentages
        total = sum(values)
        summary_text = "\n".join(
            [
                f"{regimes[i]}: {v} ({v/total*100:.1f}%)"
                for i, v in enumerate(values)
            ]
        )
        ax.text(
            0.98,
            0.97,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    # ── Interactive Network Graph ──────────────────────────────────────────

    def generate_network_html(self, output_path: str | Path) -> Path:
        """Generate interactive D3.js network graph.

        Args:
            output_path: Output HTML file path

        Returns:
            Path to generated HTML
        """
        output_path = Path(output_path)

        # Prepare node and link data for D3
        G = self.graph
        nodes = []
        node_map = {}

        # Load node titles for context
        conn = sqlite3.connect(self.rrp_db)
        node_titles = {}
        rows = conn.execute("SELECT id, title FROM entries").fetchall()
        for row in rows:
            node_titles[row[0]] = row[1]
        conn.close()

        for i, node_id in enumerate(G.nodes()):
            result = self.sweep.results.get(node_id)
            regime = "unknown"
            d_eff = 0
            degree = G.degree(node_id)
            title = node_titles.get(node_id, node_id)

            if result and not result.skipped:
                d_eff = result.d_eff
                regime_map = {
                    RegimeType.RADIAL_DOMINATED: "radial",
                    RegimeType.ISOTROPIC: "isotropic",
                    RegimeType.NOISE_DOMINATED: "noise",
                    RegimeType.DEGENERATE: "degenerate",
                }
                regime = regime_map.get(result.regime, "unknown")

            node_map[node_id] = i
            nodes.append(
                {
                    "id": node_id,
                    "title": title,
                    "regime": regime,
                    "degree": degree,
                    "d_eff": d_eff,
                }
            )

        links = []
        for source, target in G.edges():
            links.append(
                {
                    "source": source,  # Use node ID, not index
                    "target": target,  # Use node ID, not index
                }
            )

        # Generate D3.js HTML
        html_content = self._generate_d3_html(nodes, links)

        output_path.write_text(html_content, encoding="utf-8")
        return output_path

    def _generate_d3_html(self, nodes: list, links: list) -> str:
        """Generate D3.js HTML string."""
        nodes_json = json.dumps(nodes)
        links_json = json.dumps(links)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Topology</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; }}
        #container {{ display: flex; height: 100vh; }}
        #controls {{
            width: 250px; padding: 20px; background: white; border-right: 1px solid #ddd;
            overflow-y: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        #canvas {{ flex: 1; background: white; }}
        .control-group {{ margin-bottom: 20px; }}
        .control-group label {{ display: block; font-weight: bold; margin-bottom: 8px; font-size: 12px; color: #333; }}
        .control-group input[type="checkbox"] {{ margin-right: 8px; cursor: pointer; }}
        .control-item {{ display: flex; align-items: center; margin: 6px 0; font-size: 12px; }}
        input[type="text"] {{ width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }}
        input[type="range"] {{ width: 100%; }}
        .stat-box {{ background: #f9f9f9; padding: 12px; border-radius: 4px; margin-bottom: 12px; font-size: 12px; border-left: 3px solid #3B82F6; }}
        .stat-label {{ font-weight: bold; color: #333; }}
        .stat-value {{ color: #666; margin-top: 4px; }}
        .legend {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 11px; }}
        .legend-item {{ display: flex; align-items: center; margin: 6px 0; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 50%; margin-right: 8px; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h3 style="margin-bottom: 16px; font-size: 14px;">Interactive Explorer</h3>

            <div class="stat-box">
                <div class="stat-label">Total Nodes</div>
                <div class="stat-value" id="nodeCount">0</div>
            </div>

            <div class="stat-box">
                <div class="stat-label">Total Links</div>
                <div class="stat-value" id="linkCount">0</div>
            </div>

            <div class="control-group">
                <label>Search Entry:</label>
                <input type="text" id="searchInput" placeholder="Enter node ID...">
            </div>

            <div class="control-group">
                <label>Show Regime Types:</label>
                <div class="control-item">
                    <input type="checkbox" id="showRadial" checked>
                    <label for="showRadial" style="margin: 0;">Radial</label>
                </div>
                <div class="control-item">
                    <input type="checkbox" id="showIsotropic" checked>
                    <label for="showIsotropic" style="margin: 0;">Isotropic</label>
                </div>
                <div class="control-item">
                    <input type="checkbox" id="showNoise" checked>
                    <label for="showNoise" style="margin: 0;">Noise</label>
                </div>
                <div class="control-item">
                    <input type="checkbox" id="showDegenerate" checked>
                    <label for="showDegenerate" style="margin: 0;">Degenerate</label>
                </div>
            </div>

            <div class="legend">
                <strong>Regimes:</strong>
                <div class="legend-item">
                    <div class="legend-color" style="background: #3B82F6;"></div>
                    <span>Radial (hubs)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #06B6D4;"></div>
                    <span>Isotropic (balanced)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #F59E0B;"></div>
                    <span>Noise (random)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #EF4444;"></div>
                    <span>Degenerate (isolated)</span>
                </div>
            </div>

            <div style="margin-top: 20px; font-size: 11px; color: #888; line-height: 1.6;">
                <strong>Controls:</strong><br>
                • Drag nodes to move<br>
                • Scroll to zoom<br>
                • Hover for details<br>
                • Search to highlight
            </div>
        </div>
        <svg id="canvas"></svg>
    </div>

    <script>
        const nodes = {nodes_json};
        const links = {links_json};

        const regimeColors = {{
            'radial': '#3B82F6',
            'isotropic': '#06B6D4',
            'noise': '#F59E0B',
            'degenerate': '#EF4444',
            'unknown': '#999999'
        }};

        const width = document.getElementById('canvas').parentElement.clientWidth;
        const height = document.getElementById('canvas').parentElement.clientHeight;

        const svg = d3.select('#canvas')
            .attr('width', width)
            .attr('height', height);

        // Simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(50))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 2));

        function nodeRadius(d) {{
            return (5 + Math.sqrt(d.degree) * 2) * 1.7;
        }}

        // Zoom
        const zoom = d3.zoom().on('zoom', (event) => {{
            g.attr('transform', event.transform);
        }});
        svg.call(zoom);

        const g = svg.append('g');

        // Links
        const link = g.selectAll('line')
            .data(links)
            .enter()
            .append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', 1);

        // Nodes
        const node = g.selectAll('circle')
            .data(nodes)
            .enter()
            .append('circle')
            .attr('r', d => nodeRadius(d))
            .attr('fill', d => regimeColors[d.regime])
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .call(d3.drag()
                .on('start', dragStart)
                .on('drag', dragged)
                .on('end', dragEnd))
            .on('mouseover', function(event, d) {{
                d3.select(this).attr('stroke-width', 3).attr('stroke', '#000');
                tooltip.text(`${{d.title || d.id}} (ID: ${{d.id}}, degree: ${{d.degree}})`)
                    .style('opacity', 1)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY + 10) + 'px');
            }})
            .on('mouseout', function() {{
                d3.select(this).attr('stroke-width', 2).attr('stroke', '#fff');
                tooltip.style('opacity', 0);
            }});

        // Tooltip
        const tooltip = d3.select('body').append('div')
            .style('position', 'absolute')
            .style('padding', '6px 12px')
            .style('background', 'rgba(0, 0, 0, 0.8)')
            .style('color', '#fff')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .style('opacity', 0);

        function dragStart(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragEnd(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        simulation.on('tick', () => {{
            link
                .attr('x1', d => nodes[d.source.index].x)
                .attr('y1', d => nodes[d.source.index].y)
                .attr('x2', d => nodes[d.target.index].x)
                .attr('y2', d => nodes[d.target.index].y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
        }});

        // Update stats
        document.getElementById('nodeCount').textContent = nodes.length;
        document.getElementById('linkCount').textContent = links.length;

        // Filter controls
        const showRadial = document.getElementById('showRadial');
        const showIsotropic = document.getElementById('showIsotropic');
        const showNoise = document.getElementById('showNoise');
        const showDegenerate = document.getElementById('showDegenerate');
        const searchInput = document.getElementById('searchInput');

        function isNodeVisible(d, regimesShown, query) {{
            if (!regimesShown[d.regime]) return false;
            if (query && !d.id.toLowerCase().includes(query.toLowerCase())) return false;
            return true;
        }}

        function updateVisibility() {{
            const regimesShown = {{
                'radial': showRadial.checked,
                'isotropic': showIsotropic.checked,
                'noise': showNoise.checked,
                'degenerate': showDegenerate.checked,
            }};
            const query = searchInput.value;

            // Pin hidden nodes at current position; unpin re-shown nodes
            nodes.forEach(d => {{
                const visible = isNodeVisible(d, regimesShown, query);
                if (!visible) {{
                    // Save current position and pin in place so simulation ignores them
                    if (d._savedFx === undefined) {{
                        d._savedFx = d.fx;
                        d._savedFy = d.fy;
                    }}
                    d.fx = d.x;
                    d.fy = d.y;
                    d._hidden = true;
                }} else if (d._hidden) {{
                    // Restore previous pin state (or unpin)
                    d.fx = d._savedFx || null;
                    d.fy = d._savedFy || null;
                    delete d._savedFx;
                    delete d._savedFy;
                    delete d._hidden;
                }}
            }});

            node.style('display', d => isNodeVisible(d, regimesShown, query) ? null : 'none');

            link.style('display', d => {{
                const sourceVisible = isNodeVisible(d.source, regimesShown, query);
                const targetVisible = isNodeVisible(d.target, regimesShown, query);
                return (sourceVisible && targetVisible) ? null : 'none';
            }});

            // Restart simulation with lower alpha for smooth transition
            simulation.alpha(0.3).restart();
        }}

        showRadial.addEventListener('change', updateVisibility);
        showIsotropic.addEventListener('change', updateVisibility);
        showNoise.addEventListener('change', updateVisibility);
        showDegenerate.addEventListener('change', updateVisibility);
        searchInput.addEventListener('input', updateVisibility);
    </script>
</body>
</html>"""


def load_rrp_entries(conn: sqlite3.Connection) -> list[dict]:
    """Load RRP entries from database."""
    rows = conn.execute("""
        SELECT id, entry_type, title
        FROM entries
    """).fetchall()
    return [
        {"id": r[0], "entry_type": r[1], "title": r[2]}
        for r in rows
    ]


def load_rrp_links(conn: sqlite3.Connection) -> list[dict]:
    """Load RRP links from database."""
    rows = conn.execute("""
        SELECT source_id, target_id, link_type
        FROM links
    """).fetchall()
    return [
        {"source_id": r[0], "target_id": r[1], "link_type": r[2]}
        for r in rows
    ]
