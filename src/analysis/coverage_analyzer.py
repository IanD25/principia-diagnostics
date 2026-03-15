"""
coverage_analyzer.py — Measure knowledge base completeness and produce a
structured coverage report.

SCOPE: Currently DS Wiki–specific (expects ds_wiki.db schema with domain, tier,
       entry_type columns). Planned generalisation: make schema columns configurable
       so coverage commentary can be added to any RRP Tier-1 or Tier-2 report.
       Track in: Phase 3 / report enrichment milestone.

Computes
--------
1. Entity type distribution (how many of each entry_type)
2. Domain distribution (how many entries per domain)
3. Section statistics (total sections, avg per type)
4. Property coverage — for each property_name, what % of entities have it
5. Archetype distribution — value counts for mathematical_archetype
6. Link network metrics:
     - link density = M / (N*(N-1)/2)
     - avg links per entity
     - isolated entities (no links in either direction)
     - link type distribution
     - confidence tier distribution
7. Generates a human-readable markdown report with a gap summary.

Entry points
------------
    ca = CoverageAnalyzer(source_db)
    report = ca.compute_report()        # CoverageReport dataclass
    md     = ca.generate_markdown()     # str

source_db accepts pathlib.Path or str.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import SOURCE_DB  # noqa: E402


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PropertyCoverage:
    property_name: str
    total_entities:  int           # total distinct entity_ids in entries table
    filled:          int           # entities that have this property
    coverage_pct:    float         # filled / total_entities * 100
    value_distribution: Dict[str, int] = field(default_factory=dict)
    sparse_values:      List[str]  = field(default_factory=list)
    # values whose share < sparse_threshold (default 3 %) of all entities with this property


@dataclass
class NetworkMetrics:
    total_entities:              int
    total_links:                 int
    possible_links:              int                # N*(N-1)/2
    link_density:                float              # total_links / possible_links
    avg_links_per_entity:        float
    isolated_entities:           List[str]
    isolated_count:              int
    link_type_distribution:      Dict[str, int]
    confidence_tier_distribution: Dict[str, int]


@dataclass
class CoverageReport:
    # Counts
    total_entities:      int
    total_sections:      int
    total_properties:    int   # rows in entry_properties
    total_links:         int

    # Distributions
    entity_type_distribution: Dict[str, int]
    domain_distribution:      Dict[str, int]
    section_stats:            Dict[str, dict]   # per entity_type

    # Coverage
    property_coverage:        List[PropertyCoverage]
    archetype_distribution:   Dict[str, int]    # shortcut for mathematical_archetype
    d_sensitivity_counts:     Dict[str, int]    # "yes" / "no" / missing

    # Network
    network_metrics:          NetworkMetrics

    # Gap summary — populated by compute_report()
    gaps:                     List[str] = field(default_factory=list)

    # Markdown report — generated on demand
    _markdown: Optional[str] = field(default=None, repr=False)


# ── Core class ─────────────────────────────────────────────────────────────────

class CoverageAnalyzer:
    """
    Reads ds_wiki.db and produces comprehensive coverage metrics.

    Parameters
    ----------
    source_db : path to ds_wiki.db (read-only, default = config.SOURCE_DB)
    """

    # Properties whose value distribution is tracked in detail
    VOCABULARY_PROPERTIES = {
        "mathematical_archetype",
        "dimensional_sensitivity",
        "concept_tags",
    }
    # Threshold below which a value is flagged as "sparse" (fraction of entity total)
    SPARSE_THRESHOLD = 0.03

    def __init__(self, source_db: Path | str = SOURCE_DB) -> None:
        self.source_db = Path(source_db)
        if not self.source_db.exists():
            raise FileNotFoundError(f"source_db not found: {self.source_db}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.source_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _entity_type_distribution(self, conn: sqlite3.Connection) -> Dict[str, int]:
        rows = conn.execute(
            "SELECT entry_type, COUNT(*) c FROM entries GROUP BY entry_type ORDER BY c DESC"
        ).fetchall()
        return {r["entry_type"] or "unknown": r["c"] for r in rows}

    def _domain_distribution(self, conn: sqlite3.Connection) -> Dict[str, int]:
        rows = conn.execute(
            "SELECT domain, COUNT(*) c FROM entries GROUP BY domain ORDER BY c DESC"
        ).fetchall()
        return {(r["domain"] or "unknown"): r["c"] for r in rows}

    def _section_stats(self, conn: sqlite3.Connection) -> Dict[str, dict]:
        rows = conn.execute("""
            SELECT e.entry_type, COUNT(DISTINCT e.id) entities, COUNT(s.id) sections
            FROM entries e
            LEFT JOIN sections s ON s.entry_id = e.id
            GROUP BY e.entry_type
            ORDER BY entities DESC
        """).fetchall()
        stats = {}
        for r in rows:
            et = r["entry_type"] or "unknown"
            stats[et] = {
                "entities": r["entities"],
                "sections": r["sections"],
                "avg_sections_per_entity": round(
                    r["sections"] / r["entities"], 1
                ) if r["entities"] else 0,
            }
        return stats

    def _property_coverage(
        self,
        conn: sqlite3.Connection,
        total_entities: int,
    ) -> List[PropertyCoverage]:
        """
        For each distinct property_name in entry_properties, compute:
        - how many entities have it
        - value distribution (for controlled-vocabulary props)
        - sparse values
        """
        # Distinct property names
        prop_rows = conn.execute(
            "SELECT DISTINCT property_name FROM entry_properties ORDER BY property_name"
        ).fetchall()
        prop_names = [r["property_name"] for r in prop_rows]

        coverage_list: List[PropertyCoverage] = []

        for pname in prop_names:
            # Count distinct entities that have this property (non-empty value)
            filled = conn.execute(
                "SELECT COUNT(DISTINCT entry_id) FROM entry_properties "
                "WHERE property_name=? AND property_value IS NOT NULL AND property_value != ''",
                (pname,),
            ).fetchone()[0]

            pct = round(filled / total_entities * 100, 1) if total_entities else 0.0

            pc = PropertyCoverage(
                property_name=pname,
                total_entities=total_entities,
                filled=filled,
                coverage_pct=pct,
            )

            # Value distribution for tracked properties
            if pname in self.VOCABULARY_PROPERTIES:
                val_rows = conn.execute(
                    "SELECT property_value, COUNT(*) c FROM entry_properties "
                    "WHERE property_name=? AND property_value IS NOT NULL AND property_value != '' "
                    "GROUP BY property_value ORDER BY c DESC",
                    (pname,),
                ).fetchall()
                pc.value_distribution = {r["property_value"]: r["c"] for r in val_rows}
                # Flag sparse values
                threshold_count = self.SPARSE_THRESHOLD * total_entities
                pc.sparse_values = [
                    v for v, c in pc.value_distribution.items()
                    if c < threshold_count
                ]

            coverage_list.append(pc)

        return coverage_list

    def _archetype_distribution(self, conn: sqlite3.Connection) -> Dict[str, int]:
        rows = conn.execute(
            "SELECT property_value, COUNT(*) c FROM entry_properties "
            "WHERE property_name='mathematical_archetype' "
            "GROUP BY property_value ORDER BY c DESC"
        ).fetchall()
        return {r["property_value"]: r["c"] for r in rows}

    def _d_sensitivity_counts(self, conn: sqlite3.Connection, total: int) -> Dict[str, int]:
        rows = conn.execute(
            "SELECT LOWER(property_value) v, COUNT(*) c FROM entry_properties "
            "WHERE property_name='dimensional_sensitivity' "
            "GROUP BY LOWER(property_value)"
        ).fetchall()
        counts: Dict[str, int] = {r["v"]: r["c"] for r in rows}
        filled = sum(counts.values())
        counts["missing"] = total - filled
        return counts

    def _network_metrics(self, conn: sqlite3.Connection, total_entities: int) -> NetworkMetrics:
        # Total links
        total_links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]

        # Max possible undirected pairs
        possible = total_entities * (total_entities - 1) // 2
        density  = round(total_links / possible, 6) if possible else 0.0
        avg_links = round(total_links / total_entities, 2) if total_entities else 0.0

        # Isolated entities: appear neither as source nor target
        isolated_rows = conn.execute("""
            SELECT e.id FROM entries e
            WHERE e.id NOT IN (SELECT DISTINCT source_id FROM links)
            AND   e.id NOT IN (SELECT DISTINCT target_id FROM links)
            ORDER BY e.id
        """).fetchall()
        isolated = [r["id"] for r in isolated_rows]

        # Link type distribution
        lt_rows = conn.execute(
            "SELECT link_type, COUNT(*) c FROM links GROUP BY link_type ORDER BY c DESC"
        ).fetchall()
        link_types = {r["link_type"]: r["c"] for r in lt_rows}

        # Confidence tier distribution (handle NULL)
        ct_rows = conn.execute(
            "SELECT COALESCE(confidence_tier, 'original') tier, COUNT(*) c "
            "FROM links GROUP BY tier ORDER BY c DESC"
        ).fetchall()
        conf_tiers = {r["tier"]: r["c"] for r in ct_rows}

        return NetworkMetrics(
            total_entities=total_entities,
            total_links=total_links,
            possible_links=possible,
            link_density=density,
            avg_links_per_entity=avg_links,
            isolated_entities=isolated,
            isolated_count=len(isolated),
            link_type_distribution=link_types,
            confidence_tier_distribution=conf_tiers,
        )

    def _identify_gaps(self, report: "CoverageReport") -> List[str]:
        """
        Heuristic gap detection — generates actionable text recommendations.
        """
        gaps: List[str] = []

        # 1. Properties with <100% coverage among DS-native entries
        #    (reference_laws often have fewer custom properties; focus on DS-native)
        for pc in report.property_coverage:
            if pc.property_name in {"mathematical_archetype", "dimensional_sensitivity", "concept_tags"}:
                if pc.coverage_pct < 100.0:
                    missing = pc.total_entities - pc.filled
                    gaps.append(
                        f"Property '{pc.property_name}': {missing} "
                        f"{'entity' if missing == 1 else 'entities'} missing "
                        f"({pc.coverage_pct:.1f}% covered)."
                    )

        # 2. Sparse archetype values (< SPARSE_THRESHOLD * N entries)
        n = report.total_entities
        for arch, cnt in report.archetype_distribution.items():
            pct = cnt / n * 100
            if pct < self.SPARSE_THRESHOLD * 100:
                gaps.append(
                    f"Archetype '{arch}' under-represented: only {cnt} "
                    f"{'entry' if cnt == 1 else 'entries'} ({pct:.1f}%). "
                    "Consider adding cross-links to bring it into the cluster."
                )

        # 3. High isolation — more than 40% entities have no links
        iso_pct = report.network_metrics.isolated_count / n * 100 if n else 0
        if iso_pct > 40:
            gaps.append(
                f"{report.network_metrics.isolated_count} entities "
                f"({iso_pct:.1f}%) are isolated (no links). "
                "Run the Hypothesis Generator to surface candidate connections."
            )
        elif iso_pct > 20:
            gaps.append(
                f"{report.network_metrics.isolated_count} entities "
                f"({iso_pct:.1f}%) have no explicit links — run HypothesisGenerator "
                "to identify candidates."
            )

        # 4. Under-represented domains
        domain_dist = report.domain_distribution
        total_domain_entities = sum(domain_dist.values())
        for dom, cnt in domain_dist.items():
            pct = cnt / total_domain_entities * 100
            if pct < 2.0:
                gaps.append(
                    f"Domain '{dom}' has only {cnt} "
                    f"{'entry' if cnt == 1 else 'entries'} ({pct:.1f}%). "
                    "Consider adding more entries or cross-links to this domain."
                )

        # 5. Entity type imbalance — warn if any type is a singleton
        for etype, cnt in report.entity_type_distribution.items():
            if cnt == 1:
                gaps.append(
                    f"Entity type '{etype}' has only one entry. "
                    "A singleton type limits pairwise analysis for this category."
                )

        # 6. Low link density
        if report.network_metrics.link_density < 0.01:
            gaps.append(
                f"Link density is very low ({report.network_metrics.link_density:.4f}). "
                "Only {:.1f}% of possible entity pairs are explicitly linked.".format(
                    report.network_metrics.link_density * 100
                )
            )

        return gaps

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_report(self) -> CoverageReport:
        """
        Read ds_wiki.db and compute the full coverage report.
        Returns a CoverageReport dataclass with all metrics populated.
        """
        conn = self._conn()

        total_entities  = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        total_sections  = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
        total_props     = conn.execute("SELECT COUNT(*) FROM entry_properties").fetchone()[0]
        total_links     = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]

        entity_type_dist = self._entity_type_distribution(conn)
        domain_dist      = self._domain_distribution(conn)
        section_stats    = self._section_stats(conn)
        prop_coverage    = self._property_coverage(conn, total_entities)
        archetype_dist   = self._archetype_distribution(conn)
        d_sens_counts    = self._d_sensitivity_counts(conn, total_entities)
        net_metrics      = self._network_metrics(conn, total_entities)

        conn.close()

        report = CoverageReport(
            total_entities=total_entities,
            total_sections=total_sections,
            total_properties=total_props,
            total_links=total_links,
            entity_type_distribution=entity_type_dist,
            domain_distribution=domain_dist,
            section_stats=section_stats,
            property_coverage=prop_coverage,
            archetype_distribution=archetype_dist,
            d_sensitivity_counts=d_sens_counts,
            network_metrics=net_metrics,
        )

        report.gaps = self._identify_gaps(report)
        return report

    def generate_markdown(self, report: Optional[CoverageReport] = None) -> str:
        """
        Generate a human-readable markdown coverage report.
        If `report` is None, calls compute_report() first.
        """
        if report is None:
            report = self.compute_report()

        nm = report.network_metrics

        lines = [
            "# Coverage Analyzer Report",
            "",
            "## 1. Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total entities | {report.total_entities} |",
            f"| Total sections | {report.total_sections} |",
            f"| Total property rows | {report.total_properties} |",
            f"| Total links | {report.total_links} |",
            f"| Link density | {nm.link_density:.6f} "
            f"({nm.total_links} / {nm.possible_links} possible pairs) |",
            f"| Avg links per entity | {nm.avg_links_per_entity:.2f} |",
            f"| Isolated entities | {nm.isolated_count} "
            f"({nm.isolated_count / report.total_entities * 100:.1f}%) |",
            "",
            "---",
            "",
            "## 2. Entity Type Distribution",
            "",
            "| Entity Type | Count | % |",
            "|-------------|-------|---|",
        ]
        for etype, cnt in report.entity_type_distribution.items():
            pct = cnt / report.total_entities * 100
            lines.append(f"| {etype} | {cnt} | {pct:.1f}% |")

        lines += [
            "",
            "---",
            "",
            "## 3. Domain Distribution",
            "",
            "| Domain | Count |",
            "|--------|-------|",
        ]
        for dom, cnt in report.domain_distribution.items():
            lines.append(f"| {dom} | {cnt} |")

        lines += [
            "",
            "---",
            "",
            "## 4. Section Statistics (per Entity Type)",
            "",
            "| Entity Type | Entities | Sections | Avg Sections |",
            "|-------------|----------|----------|--------------|",
        ]
        for etype, s in report.section_stats.items():
            lines.append(
                f"| {etype} | {s['entities']} | {s['sections']} "
                f"| {s['avg_sections_per_entity']} |"
            )

        lines += [
            "",
            "---",
            "",
            "## 5. Property Coverage",
            "",
            "| Property | Filled | Total | % |",
            "|----------|--------|-------|---|",
        ]
        # Sort: standard properties first, then others alphabetically
        standard = {"mathematical_archetype", "dimensional_sensitivity", "concept_tags"}
        std_props = [pc for pc in report.property_coverage if pc.property_name in standard]
        other_props = [pc for pc in report.property_coverage if pc.property_name not in standard]
        for pc in std_props + other_props:
            lines.append(
                f"| {pc.property_name} | {pc.filled} | {pc.total_entities} | {pc.coverage_pct:.1f}% |"
            )

        lines += [
            "",
            "---",
            "",
            "## 6. Mathematical Archetype Distribution",
            "",
            "| Archetype | Count | % |",
            "|-----------|-------|---|",
        ]
        total_arch = sum(report.archetype_distribution.values())
        for arch, cnt in sorted(report.archetype_distribution.items(), key=lambda x: -x[1]):
            pct = cnt / total_arch * 100 if total_arch else 0
            lines.append(f"| {arch} | {cnt} | {pct:.1f}% |")

        lines += [
            "",
            "---",
            "",
            "## 7. Dimensional Sensitivity",
            "",
            "| Value | Count |",
            "|-------|-------|",
        ]
        for val, cnt in sorted(report.d_sensitivity_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {val} | {cnt} |")

        lines += [
            "",
            "---",
            "",
            "## 8. Link Network",
            "",
            "### Link Type Distribution",
            "",
            "| Link Type | Count |",
            "|-----------|-------|",
        ]
        for lt, cnt in sorted(nm.link_type_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"| {lt} | {cnt} |")

        lines += [
            "",
            "### Confidence Tier Distribution",
            "",
            "| Tier | Count |",
            "|------|-------|",
        ]
        for tier, cnt in sorted(nm.confidence_tier_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"| {tier} | {cnt} |")

        # Isolated entities
        if nm.isolated_entities:
            lines += [
                "",
                f"### Isolated Entities ({nm.isolated_count} total)",
                "",
                "These entities have no links in either direction:",
                "",
            ]
            for eid in nm.isolated_entities[:20]:
                lines.append(f"- `{eid}`")
            if len(nm.isolated_entities) > 20:
                lines.append(f"- _(and {len(nm.isolated_entities) - 20} more)_")

        # Gaps
        lines += [
            "",
            "---",
            "",
            "## 9. Identified Gaps and Recommendations",
            "",
        ]
        if report.gaps:
            for gap in report.gaps:
                lines.append(f"- ⚠️  {gap}")
        else:
            lines.append("✅ No significant gaps detected.")

        lines += ["", "---", ""]
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """
        Return a compact statistics dictionary (suitable for MCP tool output
        or quick inspection). Does not generate the full markdown report.
        """
        report = self.compute_report()
        nm = report.network_metrics

        # Property coverage summary (only the main 3)
        prop_summary = {}
        for pc in report.property_coverage:
            if pc.property_name in {"mathematical_archetype", "dimensional_sensitivity", "concept_tags"}:
                prop_summary[pc.property_name] = {
                    "filled": pc.filled,
                    "coverage_pct": pc.coverage_pct,
                }

        return {
            "total_entities":      report.total_entities,
            "total_sections":      report.total_sections,
            "total_property_rows": report.total_properties,
            "total_links":         report.total_links,
            "link_density":        nm.link_density,
            "avg_links_per_entity": nm.avg_links_per_entity,
            "isolated_entities_count": nm.isolated_count,
            "isolated_entity_ids": nm.isolated_entities[:10],  # first 10 only
            "entity_type_distribution": report.entity_type_distribution,
            "top_5_domains": dict(list(report.domain_distribution.items())[:5]),
            "property_coverage_summary": prop_summary,
            "archetype_distribution":    report.archetype_distribution,
            "link_type_distribution":    nm.link_type_distribution,
            "confidence_tier_distribution": nm.confidence_tier_distribution,
            "gaps_count": len(report.gaps),
            "gaps": report.gaps,
        }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run Coverage Analyzer on DS wiki")
    parser.add_argument("--output", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()

    ca = CoverageAnalyzer()

    if args.output == "json":
        print(json.dumps(ca.get_stats(), indent=2))
    else:
        print(ca.generate_markdown())
