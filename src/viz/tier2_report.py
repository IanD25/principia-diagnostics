"""
tier2_report.py — Tier-2 Report Generation

Generates a complete HTML report embedding all three Tier-2 cross-universe
visualizations and bridge statistics tables.

Usage:
    report = Tier2Report(bundle_db, ds_wiki_db)
    report.generate(output_path)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from viz._db import load_bridges, load_bridge_stats, load_bundle_name, load_ds_entry_meta
from viz.similarity_hist import SimilarityHist
from viz.domain_heatmap import DomainHeatmap
from viz.bridge_network import BridgeNetwork


class Tier2Report:
    """Generates complete Tier-2 HTML report with embedded cross-universe visualizations."""

    def __init__(self, bundle_db: str | Path, ds_wiki_db: str | Path):
        self.bundle_db  = Path(bundle_db)
        self.ds_wiki_db = Path(ds_wiki_db)
        self.bundle_name = load_bundle_name(self.bundle_db)

    # ── Stats helpers ─────────────────────────────────────────────────────────

    def _get_coverage(self) -> tuple[int, int, float]:
        """Returns (n_bridged_rrp, n_rrp_total, bridge_frac)."""
        conn = sqlite3.connect(self.bundle_db)
        n_total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        n_bridged = conn.execute(
            "SELECT COUNT(DISTINCT rrp_entry_id) FROM cross_universe_bridges WHERE similarity >= 0.75"
        ).fetchone()[0]
        conn.close()
        frac = n_bridged / max(n_total, 1)
        return n_bridged, n_total, round(frac, 4)

    def _get_top_ds_anchor(self) -> tuple[str, int]:
        """Returns (ds_entry_id, n_bridges) for the most-referenced DS Wiki node."""
        conn = sqlite3.connect(self.bundle_db)
        row = conn.execute(
            """SELECT ds_entry_id, COUNT(*) as n
               FROM cross_universe_bridges WHERE similarity >= 0.75
               GROUP BY ds_entry_id ORDER BY n DESC LIMIT 1"""
        ).fetchone()
        conn.close()
        if row:
            return row[0], row[1]
        return "—", 0

    def _get_verdict(self, bridge_frac: float, mean_sim: float) -> tuple[str, str]:
        """Returns (verdict_label, css_class)."""
        if bridge_frac >= 0.70 and mean_sim >= 0.80:
            return "WELL-INTEGRATED", "consistent"
        elif bridge_frac >= 0.40 or mean_sim >= 0.77:
            return "PARTIAL", "marginal"
        else:
            return "ISOLATED", "fragmented"

    def _render_stats(self, bridge_stats: dict, net_stats: dict) -> str:
        n_bridged, n_total, bridge_frac = self._get_coverage()
        top_anchor, top_count = self._get_top_ds_anchor()
        tier_1_5_pct = 100 * bridge_stats["tier_1_5"] / max(bridge_stats["total"], 1)

        # Resolve top anchor title
        if top_anchor != "—":
            meta = load_ds_entry_meta(self.ds_wiki_db, [top_anchor])
            top_label = meta[top_anchor].title
        else:
            top_label = "—"

        return f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Total Bridges</div>
                <div class="stat-value">{bridge_stats['total']:,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">High-Confidence (≥0.85)</div>
                <div class="stat-value">{bridge_stats['tier_1_5']} <span class="stat-sub">({tier_1_5_pct:.1f}%)</span></div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Mean Similarity</div>
                <div class="stat-value">{bridge_stats['mean_sim']:.4f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">DS Wiki Anchors</div>
                <div class="stat-value">{net_stats['ds_nodes']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">RRP Entries Bridged</div>
                <div class="stat-value">{n_bridged} / {n_total}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Coverage</div>
                <div class="stat-value">{bridge_frac:.1%}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Top Anchor</div>
                <div class="stat-value" style="font-size:13px;">{top_label}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Top Anchor Bridges</div>
                <div class="stat-value">{top_count}</div>
            </div>
        </div>
        """

    def _render_top_bridges_table(self, n: int = 20) -> str:
        bridges = load_bridges(self.bundle_db, sim_threshold=0.75)[:n]
        ds_ids = list({b.ds_entry_id for b in bridges})
        ds_meta = load_ds_entry_meta(self.ds_wiki_db, ds_ids)

        rows = [
            "<thead><tr>"
            "<th style='width:30%'>RRP Entry</th>"
            "<th style='width:25%'>DS Wiki Entry</th>"
            "<th style='width:8%'>Sim</th>"
            "<th style='width:10%'>Tier</th>"
            "<th style='width:15%'>Link Type</th>"
            "</tr></thead>"
            "<tbody>"
        ]

        for b in bridges:
            ds = ds_meta[b.ds_entry_id]
            tier_badge = (
                "<span class='badge badge-t15'>1.5</span>"
                if b.confidence_tier == "1.5"
                else "<span class='badge badge-t2'>2</span>"
            )
            sim_bar_width = int((b.similarity - 0.75) / (1.0 - 0.75) * 100)
            rows.append(
                f"<tr>"
                f"<td><code style='font-size:10px'>{b.rrp_entry_id}</code>"
                f"<div style='font-size:11px;color:#666;margin-top:2px'>{b.rrp_entry_title[:50]}</div></td>"
                f"<td><code style='font-size:10px'>{b.ds_entry_id}</code>"
                f"<div style='font-size:11px;color:#666;margin-top:2px'>{ds.title[:40]}</div></td>"
                f"<td style='text-align:center'>"
                f"  <div class='sim-bar'><div class='sim-fill' style='width:{sim_bar_width}%'></div></div>"
                f"  <span style='font-size:11px'>{b.similarity:.4f}</span>"
                f"</td>"
                f"<td style='text-align:center'>{tier_badge}</td>"
                f"<td style='font-size:11px'>{b.proposed_link_type}</td>"
                f"</tr>"
            )

        rows.append("</tbody>")
        return "\n".join(rows)

    def generate(
        self,
        output_path: str | Path,
        net_threshold: float = 0.82,
        sim_threshold: float = 0.75,
    ) -> Path:
        """Generate complete Tier-2 HTML report.

        Args:
            output_path:   Output HTML file path
            net_threshold: Similarity cutoff for bridge network visualization
            sim_threshold: Similarity cutoff for histogram + heatmap

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        base = output_path.stem.replace("tier2_report_", "")

        # Generate visualization files with dataset-specific names
        hist_png  = output_dir / f"t2_hist_{base}.png"
        hist_html = output_dir / f"t2_hist_{base}.html"
        heat_png  = output_dir / f"t2_heat_{base}.png"
        heat_html = output_dir / f"t2_heat_{base}.html"
        net_png   = output_dir / f"t2_net_{base}.png"
        net_html  = output_dir / f"t2_net_{base}.html"

        class _FixedOut:
            """Redirect generate() to fixed paths."""
            pass

        hist_result = SimilarityHist(self.bundle_db).generate(
            output_dir=output_dir, sim_threshold=sim_threshold
        )
        # Rename to dataset-specific names
        if hist_result["png"].exists():
            hist_result["png"].rename(hist_png)
        if hist_result["html"].exists():
            hist_result["html"].rename(hist_html)

        heat_result = DomainHeatmap(self.bundle_db, self.ds_wiki_db).generate(
            output_dir=output_dir, sim_threshold=sim_threshold
        )
        if heat_result["png"].exists():
            heat_result["png"].rename(heat_png)
        if heat_result["html"].exists():
            heat_result["html"].rename(heat_html)

        net_result = BridgeNetwork(self.bundle_db, self.ds_wiki_db).generate(
            output_dir=output_dir, sim_threshold=net_threshold
        )
        if net_result["png"].exists():
            net_result["png"].rename(net_png)
        if net_result["html"].exists():
            net_result["html"].rename(net_html)

        # Collect stats
        bridge_stats = load_bridge_stats(self.bundle_db)
        net_stats = net_result["stats"]
        _, _, bridge_frac = self._get_coverage()
        verdict, verdict_css = self._get_verdict(bridge_frac, bridge_stats["mean_sim"])

        stats_html = self._render_stats(bridge_stats, net_stats)
        table_html = self._render_top_bridges_table()

        verdict_messages = {
            "WELL-INTEGRATED": (
                "This dataset is well-integrated into the DS Wiki formal foundation. "
                f"Over {bridge_frac:.0%} of RRP entries have at least one cross-universe bridge "
                f"above threshold (sim ≥ {sim_threshold}), with strong mean similarity "
                f"({bridge_stats['mean_sim']:.3f}). The dataset's concepts map reliably onto "
                "established formal structures."
            ),
            "PARTIAL": (
                f"This dataset is partially connected to DS Wiki ({bridge_frac:.0%} coverage). "
                "Some entries have strong bridges but others are poorly anchored. Review the "
                "domain heatmap to identify which entry types are under-connected and consider "
                "adding more descriptive metadata to improve bridge quality."
            ),
            "ISOLATED": (
                f"This dataset has low coverage ({bridge_frac:.0%} of entries bridged). "
                "Most entries do not match any DS Wiki formal structure above threshold. "
                "This may indicate a highly specialized domain not yet represented in DS Wiki, "
                "or that entry descriptions need enrichment to enable semantic matching."
            ),
        }
        verdict_text = verdict_messages[verdict]

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PFD Tier-2 Report — {self.bundle_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6; color: #333; background: #f9f9f9;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{
            background: white; padding: 30px; border-radius: 8px;
            margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #7C3AED;
        }}
        header h1 {{ font-size: 32px; margin-bottom: 10px; }}
        header p {{ color: #666; font-size: 14px; }}
        .section {{
            background: white; padding: 30px; border-radius: 8px;
            margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            font-size: 20px; margin-bottom: 20px; padding-bottom: 10px;
            border-bottom: 2px solid #7C3AED; color: #1F2937;
        }}
        .section h3 {{
            font-size: 16px; margin-top: 20px; margin-bottom: 15px; color: #374151;
        }}
        .stat-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px; margin-bottom: 20px;
        }}
        .stat-box {{
            background: #f3f4f6; padding: 15px; border-radius: 6px;
            border-left: 3px solid #7C3AED;
        }}
        .stat-label {{
            font-size: 12px; font-weight: bold; color: #666;
            text-transform: uppercase; letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 20px; font-weight: bold; color: #1F2937; margin-top: 5px;
        }}
        .stat-sub {{ font-size: 13px; font-weight: normal; color: #888; }}
        .visualization {{ margin: 20px 0; text-align: center; }}
        .visualization img {{
            max-width: 100%; height: auto; border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .visualization-frame {{
            width: 100%; height: 600px; border: 1px solid #e5e7eb;
            border-radius: 6px; margin: 20px 0;
        }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        table th {{
            background: #f3f4f6; padding: 12px; text-align: left;
            font-weight: bold; font-size: 13px; color: #374151;
            border-bottom: 2px solid #d1d5db;
        }}
        table td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; font-size: 13px; }}
        table tbody tr:hover {{ background: #f9fafb; }}
        code {{
            background: #f3f4f6; padding: 2px 5px; border-radius: 3px;
            font-family: "Monaco", "Menlo", monospace; font-size: 11px;
        }}
        .badge {{
            display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 11px; font-weight: bold;
        }}
        .badge-t15 {{ background: #fde68a; color: #92400e; }}
        .badge-t2  {{ background: #dbeafe; color: #1e40af; }}
        .sim-bar {{
            width: 100%; height: 4px; background: #e5e7eb; border-radius: 2px;
            margin-bottom: 3px;
        }}
        .sim-fill {{
            height: 100%; background: #7C3AED; border-radius: 2px;
        }}
        .verdict-box {{
            background: #f0fdf4; border-left: 4px solid #22c55e;
            padding: 20px; border-radius: 6px; margin: 20px 0;
        }}
        .verdict-box.marginal {{ background: #fffbeb; border-left-color: #f59e0b; }}
        .verdict-box.fragmented {{ background: #fef2f2; border-left-color: #ef4444; }}
        .verdict-label {{ font-weight: bold; font-size: 16px; margin-bottom: 8px; }}
        .verdict-text {{ color: #555; font-size: 14px; line-height: 1.7; }}
        footer {{
            text-align: center; padding: 20px; color: #999;
            font-size: 12px; margin-top: 40px;
        }}
        @media (max-width: 768px) {{
            .stat-grid {{ grid-template-columns: 1fr; }}
            header, .section {{ padding: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>PFD Tier-2 Report</h1>
            <p>Cross-Universe Bridge Analysis — {self.bundle_name}</p>
        </header>

        <div class="section">
            <h2>Executive Summary</h2>
            <p style="margin-bottom: 15px;">
                Tier-2 measures how well the RRP dataset connects to the DS Wiki formal
                knowledge graph. Each <strong>bridge</strong> is a semantic link between
                an RRP entry and a DS Wiki foundation node, scored by cosine similarity.
                High bridge density and quality indicate the dataset is grounded in
                established formal structures.
            </p>
            {stats_html}
        </div>

        <div class="section">
            <h2>1. Bridge Similarity Distribution</h2>
            <p>
                Histogram of cosine similarity scores for all cross-universe bridges.
                Bars are colour-coded by confidence tier:
            </p>
            <ul style="margin-left: 20px; margin-bottom: 15px;">
                <li><strong>Tier 1.5 (≥0.85, red-orange):</strong> "Analogous to" — strong structural match</li>
                <li><strong>Tier 2 (0.75–0.85, blue):</strong> "Couples to" — meaningful but weaker match</li>
            </ul>
            <h3>1.1 Static Overview</h3>
            <div class="visualization">
                <img src="{hist_png.name}" alt="Bridge Similarity Histogram">
                <p><em>Stacked histogram with tier thresholds and network visualization cutoff marked.</em></p>
            </div>
            <h3>1.2 Interactive Histogram</h3>
            <div class="visualization-frame">
                <iframe src="{hist_html.name}" style="width:100%; height:100%; border:none;"></iframe>
            </div>
            <p style="text-align:center; color:#666; font-size:12px; margin-top:5px;">
                <em>Hover bars for exact counts. Use toolbar to zoom or export PNG.</em>
            </p>
        </div>

        <div class="section">
            <h2>2. Domain Connection Heatmap</h2>
            <p>
                Shows which DS Wiki knowledge domains each RRP entry type connects to.
                Rows = RRP source types (theorems, classes, etc.).
                Columns = DS Wiki type groups (RL, CHEM, BIO, etc.).
                Darker cells = more bridges.
            </p>
            <h3>2.1 Static Overview</h3>
            <div class="visualization">
                <img src="{heat_png.name}" alt="Domain Connection Heatmap">
                <p><em>Cell values show bridge count at sim ≥ {sim_threshold}. Sparse columns collapsed into "Other".</em></p>
            </div>
            <h3>2.2 Interactive Heatmap</h3>
            <div class="visualization-frame">
                <iframe src="{heat_html.name}" style="width:100%; height:100%; border:none;"></iframe>
            </div>
            <p style="text-align:center; color:#666; font-size:12px; margin-top:5px;">
                <em>Hover cells for exact counts. Identify which DS Wiki domains anchor your data.</em>
            </p>
        </div>

        <div class="section">
            <h2>3. Bridge Network</h2>
            <p>
                Bipartite force-directed graph showing RRP entries (left) connected to
                DS Wiki anchor nodes (right) at sim ≥ {net_threshold}.
                Node size scales with bridge count. Highly-connected DS Wiki anchors
                appear as large central hubs.
            </p>
            <h3>3.1 Static Overview</h3>
            <div class="visualization">
                <img src="{net_png.name}" alt="Bridge Network">
                <p><em>{net_stats['rrp_nodes']} RRP nodes + {net_stats['ds_nodes']} DS Wiki anchors + {net_stats['edges']} bridge edges shown.</em></p>
            </div>
            <h3>3.2 Interactive Network</h3>
            <div class="visualization-frame">
                <iframe src="{net_html.name}" style="width:100%; height:100%; border:none;"></iframe>
            </div>
            <p style="text-align:center; color:#666; font-size:12px; margin-top:5px;">
                <em>Drag nodes to explore structure. Hover for entry titles and similarity scores.</em>
            </p>
        </div>

        <div class="section">
            <h2>4. Top Bridges</h2>
            <p>
                The strongest cross-universe bridges ranked by cosine similarity.
                These represent the tightest connections between this dataset and the
                DS Wiki formal foundation.
            </p>
            <table>
                {table_html}
            </table>
        </div>

        <div class="section">
            <h2>5. Tier-2 Verdict</h2>
            <p>Verdict thresholds:</p>
            <ul style="margin-left: 20px; margin-bottom: 15px;">
                <li><strong>WELL-INTEGRATED:</strong> ≥70% coverage AND mean sim ≥ 0.80</li>
                <li><strong>PARTIAL:</strong> ≥40% coverage OR mean sim ≥ 0.77</li>
                <li><strong>ISOLATED:</strong> &lt;40% coverage AND mean sim &lt; 0.77</li>
            </ul>
            <div class="verdict-box {verdict_css}">
                <div class="verdict-label">&#10003; Tier-2 Verdict: {verdict}</div>
                <div class="verdict-text">{verdict_text}</div>
            </div>
        </div>

        <div class="section">
            <h2>6. Interpretation & Next Steps</h2>
            <p>
                <strong>What bridges mean:</strong> A bridge connects an RRP entry to a DS Wiki
                node via semantic embedding similarity. Tier-1.5 bridges (≥0.85) indicate the
                RRP concept is structurally analogous to a formal foundation. Tier-2 (0.75–0.85)
                indicates a meaningful coupling worth investigating.
            </p>
            <p style="margin-top: 15px;">
                <strong>Recommended actions:</strong>
            </p>
            <ol style="margin-left: 20px; margin-bottom: 15px;">
                <li>Review the domain heatmap to identify which DS Wiki domains anchor your data</li>
                <li>Investigate uncovered entries (those with no bridges) for missing metadata</li>
                <li>Use top bridges in Section 4 to validate semantic alignment manually</li>
                <li>Combine with Tier-1 results for the full PFD Score (0.0–1.0)</li>
            </ol>
            <p>
                <strong>Caveats:</strong> Bridge similarity is based on BGE embeddings — semantic
                proximity does not guarantee formal equivalence. Always verify high-value bridges
                with domain expertise before drawing scientific conclusions.
            </p>
        </div>

        <footer>
            <p>Generated by PFD Tier-2 Report Generator</p>
            <p>For questions or feedback, see the documentation at:
               https://github.com/IanD25/ds-wiki-transformer</p>
        </footer>
    </div>
</body>
</html>"""

        output_path.write_text(html_content, encoding="utf-8")
        return output_path
