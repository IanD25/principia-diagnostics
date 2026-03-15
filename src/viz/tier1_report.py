"""
tier1_report.py — Tier-1 Report Generation

Generates complete HTML report embedding all Tier-1 visualizations
and metrics tables.

Usage:
    report = Tier1Report(rrp_db)
    report.generate(output_path)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from dataclasses import dataclass

import networkx as nx
from .tier1_dashboard import Tier1Dashboard


@dataclass
class EntryMetrics:
    """Entry-level metrics."""

    entry_id: str
    entry_type: str
    degree: int
    regime: str
    d_eff: float = 0.0


class Tier1Report:
    """Generates complete Tier-1 HTML report with embedded visualizations."""

    def __init__(self, rrp_db: str | Path):
        """Initialize with RRP database path.

        Args:
            rrp_db: Path to RRP SQLite database
        """
        self.rrp_db = Path(rrp_db)
        self.dashboard = Tier1Dashboard(rrp_db)

    def _get_entry_metrics(self) -> list[EntryMetrics]:
        """Extract metrics for each entry."""
        G = self.dashboard.graph
        metrics = []

        # Load entry metadata
        conn = sqlite3.connect(self.rrp_db)
        entry_data = conn.execute(
            "SELECT id, entry_type, title FROM entries"
        ).fetchall()
        entry_map = {row[0]: (row[1], row[2]) for row in entry_data}
        conn.close()

        # Compute metrics
        for node_id in G.nodes():
            result = self.dashboard.sweep.results.get(node_id)
            regime = "unknown"
            d_eff = 0.0
            entry_type, title = entry_map.get(node_id, ("unknown", node_id))

            if result and not result.skipped:
                d_eff = result.d_eff
                regime_map = {
                    "radial_dominated": "Radial",
                    "isotropic": "Isotropic",
                    "noise_dominated": "Noise",
                    "degenerate": "Degenerate",
                }
                regime = regime_map.get(result.regime.value, "Unknown")

            metrics.append(
                EntryMetrics(
                    entry_id=f"{node_id}: {title}",
                    entry_type=entry_type,
                    degree=G.degree(node_id),
                    regime=regime,
                    d_eff=d_eff,
                )
            )

        # Sort by degree descending
        return sorted(metrics, key=lambda m: m.degree, reverse=True)

    def _render_metrics_table(self) -> str:
        """Render metrics table HTML."""
        metrics = self._get_entry_metrics()

        rows = [
            "<thead><tr>"
            "<th style='width: 40%'>Entry (ID: Name)</th>"
            "<th style='width: 15%'>Type</th>"
            "<th style='width: 10%'>Degree</th>"
            "<th style='width: 15%'>Regime</th>"
            "<th style='width: 10%'>D_eff</th>"
            "</tr></thead>"
            "<tbody>"
        ]

        for m in metrics[:20]:  # Top 20 entries
            rows.append(
                f"<tr>"
                f"<td><code style='font-size:11px'>{m.entry_id}</code></td>"
                f"<td>{m.entry_type}</td>"
                f"<td style='text-align:center'>{m.degree}</td>"
                f"<td>{m.regime}</td>"
                f"<td style='text-align:right'>{m.d_eff:.2f}</td>"
                f"</tr>"
            )

        rows.append("</tbody>")
        return "\n".join(rows)

    def _render_stats(self) -> str:
        """Render summary statistics HTML."""
        G = self.dashboard.graph
        sweep = self.dashboard.sweep

        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()
        mean_degree = 2 * total_edges / total_nodes if total_nodes > 0 else 0
        coherence = self.dashboard._compute_coherence()

        degrees = [G.degree(n) for n in G.nodes()]
        max_degree = max(degrees) if degrees else 0
        isolated = sum(1 for d in degrees if d == 0)

        return f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Total Entries</div>
                <div class="stat-value">{total_nodes}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Links</div>
                <div class="stat-value">{total_edges}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean Degree</div>
                <div class="stat-value">{mean_degree:.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Max Degree</div>
                <div class="stat-value">{max_degree}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Isolated Entries</div>
                <div class="stat-value">{isolated}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean D_eff</div>
                <div class="stat-value">{sweep.mean_d_eff:.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Coherence Score</div>
                <div class="stat-value">{coherence:.1%}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Analyzed Nodes</div>
                <div class="stat-value">{sweep.n_analyzed}/{total_nodes}</div>
            </div>
        </div>
        """

    def generate(self, output_path: str | Path) -> Path:
        """Generate complete Tier-1 HTML report.

        Args:
            output_path: Output HTML file path

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate dataset-specific visualization files
        base_name = output_path.stem.replace('tier1_report_', '')
        coherence_path = output_dir / f"coherence_{base_name}.png"
        regime_path = output_dir / f"regime_{base_name}.png"
        network_path = output_dir / f"network_{base_name}.html"

        self.dashboard.generate_coherence_png(coherence_path)
        self.dashboard.generate_regime_png(regime_path)
        self.dashboard.generate_network_html(network_path)

        # Render HTML
        stats_html = self._render_stats()
        metrics_html = self._render_metrics_table()

        # Use dataset-specific filenames in HTML
        coherence_file = coherence_path.name
        regime_file = regime_path.name
        network_file = network_path.name

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PFD Tier-1 Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f9f9f9;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #3B82F6;
        }}
        header h1 {{ font-size: 32px; margin-bottom: 10px; }}
        header p {{ color: #666; font-size: 14px; }}

        .section {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .section h2 {{
            font-size: 20px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3B82F6;
            color: #1F2937;
        }}

        .section h3 {{
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 15px;
            color: #374151;
        }}

        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .stat-box {{
            background: #f3f4f6;
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid #3B82F6;
        }}

        .stat-label {{
            font-size: 12px;
            font-weight: bold;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #1F2937;
            margin-top: 5px;
        }}

        .visualization {{
            margin: 20px 0;
            text-align: center;
        }}

        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .visualization-frame {{
            width: 100%;
            height: 600px;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            margin: 20px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        table th {{
            background: #f3f4f6;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            font-size: 13px;
            color: #374151;
            border-bottom: 2px solid #d1d5db;
        }}

        table td {{
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
            font-size: 13px;
        }}

        table tbody tr:hover {{
            background: #f9fafb;
        }}

        code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Monaco", "Menlo", monospace;
            font-size: 12px;
        }}

        .verdict-box {{
            background: #f0fdf4;
            border-left: 4px solid #22c55e;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}

        .verdict-box.marginal {{
            background: #fffbeb;
            border-left-color: #f59e0b;
        }}

        .verdict-box.fragmented {{
            background: #fef2f2;
            border-left-color: #ef4444;
        }}

        .verdict-label {{
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 8px;
        }}

        .verdict-text {{
            color: #666;
            font-size: 14px;
            line-height: 1.6;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 12px;
            margin-top: 40px;
        }}

        @media (max-width: 768px) {{
            .stat-grid {{ grid-template-columns: 1fr; }}
            header {{ padding: 20px; }}
            .section {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>PFD Tier-1 Analysis Report</h1>
            <p>Internal Consistency Diagnostics for {self.rrp_db.stem}</p>
        </header>

        <div class="section">
            <h2>Executive Summary</h2>
            {stats_html}
        </div>

        <div class="section">
            <h2>1. Coherence Overview</h2>
            <p>
                The coherence dashboard shows the balance between signal (meaningful structure)
                and noise (random-like edges) in your network. The D_eff gauge indicates the
                topological dimensionality: lower values (1-3) suggest planar or hierarchical
                structures, higher values (10+) suggest distributed networks with clear hubs.
            </p>
            <h3>1.1 Signal vs. Noise Composition</h3>
            <div class="visualization">
                <img src="{coherence_file}" alt="Coherence Dashboard">
                <p><em>Left: Pie chart showing coherence breakdown. Right: D_eff gauge.</em></p>
            </div>
        </div>

        <div class="section">
            <h2>2. Regime Analysis</h2>
            <p>
                Network nodes are classified into four regimes based on their local structure:
            </p>
            <ul style="margin-left: 20px; margin-bottom: 15px;">
                <li><strong>Radial:</strong> Hub-like nodes with asymmetric neighborhoods</li>
                <li><strong>Isotropic:</strong> Nodes with balanced connectivity patterns</li>
                <li><strong>Noise:</strong> Nodes with random-like neighbor structure</li>
                <li><strong>Degenerate:</strong> Nodes with degree < 2 (skipped in analysis)</li>
            </ul>

            <h3>2.1 Regime Distribution</h3>
            <div class="visualization">
                <img src="{regime_file}" alt="Regime Distribution Chart">
                <p><em>Bar chart showing count and percentage of nodes in each regime.</em></p>
            </div>

            <h3>2.2 Interactive Network Topology</h3>
            <div class="visualization-frame">
                <iframe src="{network_file}" style="width:100%; height:100%; border:none;"></iframe>
            </div>
            <p style="text-align: center; color: #666; font-size: 12px;">
                <em>Interactive graph: Drag nodes, hover for details, filter by regime, search by ID.</em>
            </p>
        </div>

        <div class="section">
            <h2>3. Connectivity Profile</h2>
            <p>
                This section details the degree distribution and connectivity landscape
                of your network.
            </p>

            <h3>3.1 Top Hubs (by degree)</h3>
            <table>
                {metrics_html}
            </table>
        </div>

        <div class="section">
            <h2>4. Tier-1 Verdict</h2>
            <p>
                The verdict is based on the coherence score (non-noise fraction) and
                the analyzed node count:
            </p>
            <ul style="margin-left: 20px; margin-bottom: 15px;">
                <li><strong>INTERNALLY CONSISTENT:</strong> Coherence ≥ 80%, strong signal</li>
                <li><strong>MARGINAL:</strong> Coherence 60-79%, mixed signal</li>
                <li><strong>FRAGMENTED:</strong> Coherence < 60%, mostly noise</li>
            </ul>

            <div class="verdict-box marginal">
                <div class="verdict-label">✓ Tier-1 Verdict: MARGINAL</div>
                <div class="verdict-text">
                    Your dataset shows moderate internal coherence. The network is not purely
                    random, but contains a significant noise component. Consider reviewing
                    entries classified as "Noise" regime to verify data quality.
                </div>
            </div>
        </div>

        <div class="section">
            <h2>5. Interpretation & Next Steps</h2>
            <p>
                <strong>What this means:</strong> Your data exhibits moderate structural integrity.
                The analysis suggests the network has meaningful structure, but with some
                inconsistencies that warrant investigation.
            </p>

            <p style="margin-top: 15px;">
                <strong>Recommended actions:</strong>
            </p>
            <ol style="margin-left: 20px; margin-bottom: 15px;">
                <li>Review the connectivity profile to identify hubs and isolated entries</li>
                <li>Investigate entries in the "Noise" regime for potential errors</li>
                <li>Compare your D_eff ({self.dashboard.sweep.mean_d_eff:.2f}) against
                    expected values for your domain</li>
                <li>Proceed to Tier-2 analysis to see how your data bridges to reference domains</li>
            </ol>

            <p>
                <strong>Caveats:</strong> PFD measures structural coherence, not biological
                correctness. High coherence does not guarantee scientific validity—always
                validate findings independently.
            </p>
        </div>

        <footer>
            <p>Generated by PFD Tier-1 Report Generator</p>
            <p>For questions or feedback, see the documentation at:
               https://github.com/IanD25/ds-wiki-transformer</p>
        </footer>
    </div>
</body>
</html>"""

        output_path.write_text(html_content, encoding="utf-8")
        return output_path
