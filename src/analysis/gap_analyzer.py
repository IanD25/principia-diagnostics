"""
gap_analyzer.py — Identify knowledge-base gaps and produce ranked enrichment priorities.

SCOPE: Currently DS Wiki–specific (operates on ds_wiki.db schema: entry_type, tier,
       domain columns). Planned generalisation: parameterise the schema so this runs
       on any RRP SQLite db, contributing gap commentary to the Tier-1 report section.
       Track in: Phase 3 / report enrichment milestone.

Complements CoverageAnalyzer (which reports global counts and network metrics) by
focusing on **actionable per-type gaps and ranked recommendations**.

What it computes
----------------
1. PropertyGap   — per (entity_type × property_name) pairs whose coverage < threshold.
                   e.g. "CS entries: 8/15 missing 'concept_tags'"
2. TaxonomyGap   — controlled-vocabulary values with < SPARSE_THRESHOLD representation.
                   e.g. "'oscillatory' archetype: 2 entries (1.0%)"
3. TypeBalanceGap — entity types below their expected minimum count.
                   e.g. "open_question: 7 observed vs minimum 10 → add 3 more"
4. LinkGap        — entries that are isolated, have no tier-1 links, or only same-type links.
5. EnrichmentPriority — unified ranked list of the most impactful additions.

Entry points
------------
    ga = GapAnalyzer(source_db)
    report = ga.analyze()                # GapAnalysisReport dataclass
    print(report.as_markdown())          # human-readable summary

    # Customise expected type minimums before calling analyze():
    ga.type_minimums["open_question"] = 20
    report = ga.analyze()

source_db accepts pathlib.Path or str (defaults to config.SOURCE_DB).
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import SOURCE_DB  # noqa: E402


# ── Thresholds ─────────────────────────────────────────────────────────────────
PROPERTY_COVERAGE_THRESHOLD = 0.80   # flag (entity_type, prop) if < 80% filled
SPARSE_TAXONOMY_THRESHOLD   = 0.03   # flag archetype/domain value if < 3% of total
SPARSE_TAXONOMY_ABS         = 2      # also flag if absolute count ≤ 2

# Default minimum expected counts per entity type.
# Keys must match values in entries.entry_type exactly.
DEFAULT_TYPE_MINIMUMS: Dict[str, int] = {
    "reference_law":  50,
    "method":         10,
    "law":            10,
    "open_question":  10,
    "constraint":      5,
    "instantiation":   5,
    "axiom":           3,
    "parameter":       3,
    "theorem":         3,
    "mechanism":       3,
}

# Properties that carry controlled-vocabulary values (taxonomy analysis)
TAXONOMY_PROPERTIES = frozenset({
    "mathematical_archetype",
    "dimensional_sensitivity",
})


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class PropertyGap:
    entity_type:    str
    property_name:  str
    total_of_type:  int           # entries of this type
    filled:         int           # entries of this type that have the property
    missing_count:  int           # total_of_type - filled
    coverage_pct:   float         # filled / total_of_type * 100
    missing_ids:    List[str]     # up to first 10 entry IDs missing this property
    priority:       int           # 1=high (many missing), 2=medium, 3=low


@dataclass
class TaxonomyGap:
    taxonomy_name:  str           # property_name (e.g. "mathematical_archetype")
    value:          str           # specific value that is sparse/absent
    count:          int           # how many entries have this value
    pct_of_filled:  float         # count / total entries with this property * 100
    flag:           str           # "sparse" | "singleton"


@dataclass
class TypeBalanceGap:
    entity_type:    str
    observed:       int
    expected_min:   int
    deficit:        int           # max(0, expected_min - observed)
    suggestion:     str           # human-readable recommendation


@dataclass
class LinkGap:
    entry_id:       str
    title:          str
    entity_type:    str
    total_links:    int
    gap_type:       str           # "isolated" | "no_tier1" | "same_type_only"
    detail:         str           # brief explanation


@dataclass
class EnrichmentPriority:
    rank:           int
    action:         str           # "add_property" | "add_entry" | "add_link"
    target:         str           # entity_type, entry_id, or property_name
    description:    str
    impact_score:   float         # 0.0–1.0 (higher = more impactful)


@dataclass
class GapAnalysisReport:
    # Sub-reports
    property_gaps:        List[PropertyGap]
    taxonomy_gaps:        List[TaxonomyGap]
    type_balance_gaps:    List[TypeBalanceGap]
    link_gaps:            List[LinkGap]
    enrichment_priorities: List[EnrichmentPriority]

    # Quick-access summary stats
    summary_stats:        Dict[str, object]

    def as_markdown(self) -> str:
        """Render the full gap analysis as a markdown report."""
        lines: List[str] = [
            "# Gap Analysis Report",
            "",
            "## Summary",
            "",
        ]

        # Summary stats table
        ss = self.summary_stats
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total entities | {ss.get('total_entities', '?')} |",
            f"| Property gaps (< 80% per type) | {len(self.property_gaps)} |",
            f"| Sparse taxonomy values | {len(self.taxonomy_gaps)} |",
            f"| Type balance gaps | {len(self.type_balance_gaps)} |",
            f"| Link gaps | {len(self.link_gaps)} |",
            f"| Isolated entries | {ss.get('isolated_count', '?')} |",
            "",
        ]

        # 1. Enrichment priorities
        if self.enrichment_priorities:
            lines += ["## Top Enrichment Priorities", ""]
            lines += [
                "| Rank | Action | Target | Impact | Description |",
                "|------|--------|--------|--------|-------------|",
            ]
            for p in self.enrichment_priorities[:15]:
                lines.append(
                    f"| {p.rank} | {p.action} | `{p.target}` | "
                    f"{p.impact_score:.2f} | {p.description} |"
                )
            lines.append("")

        # 2. Property gaps
        if self.property_gaps:
            lines += ["## Property Coverage Gaps", ""]
            lines += [
                "| Entity Type | Property | Coverage | Missing | Top Missing IDs |",
                "|-------------|----------|----------|---------|----------------|",
            ]
            for g in self.property_gaps:
                ids_str = ", ".join(g.missing_ids[:5]) + ("…" if len(g.missing_ids) > 5 else "")
                lines.append(
                    f"| {g.entity_type} | {g.property_name} | "
                    f"{g.coverage_pct:.0f}% | {g.missing_count} | {ids_str} |"
                )
            lines.append("")

        # 3. Taxonomy gaps
        if self.taxonomy_gaps:
            lines += ["## Sparse Taxonomy Values", ""]
            lines += [
                "| Taxonomy | Value | Count | % of Filled | Flag |",
                "|----------|-------|-------|-------------|------|",
            ]
            for g in self.taxonomy_gaps:
                lines.append(
                    f"| {g.taxonomy_name} | {g.value} | {g.count} | "
                    f"{g.pct_of_filled:.1f}% | {g.flag} |"
                )
            lines.append("")

        # 4. Type balance gaps
        if self.type_balance_gaps:
            lines += ["## Entity Type Balance Gaps", ""]
            lines += [
                "| Entity Type | Observed | Min Expected | Deficit | Suggestion |",
                "|-------------|----------|--------------|---------|------------|",
            ]
            for g in self.type_balance_gaps:
                lines.append(
                    f"| {g.entity_type} | {g.observed} | {g.expected_min} | "
                    f"{g.deficit} | {g.suggestion} |"
                )
            lines.append("")

        # 5. Link gaps (top 20)
        if self.link_gaps:
            lines += [f"## Link Gaps ({len(self.link_gaps)} entries)", ""]
            lines += [
                "| Entry ID | Title | Type | Links | Gap Type |",
                "|----------|-------|------|-------|----------|",
            ]
            for g in self.link_gaps[:20]:
                lines.append(
                    f"| {g.entry_id} | {g.title[:40]} | {g.entity_type} | "
                    f"{g.total_links} | {g.gap_type} |"
                )
            if len(self.link_gaps) > 20:
                lines.append(f"| … | *{len(self.link_gaps) - 20} more not shown* | | | |")
            lines.append("")

        return "\n".join(lines)


# ── GapAnalyzer ────────────────────────────────────────────────────────────────

class GapAnalyzer:
    """
    Analyse a knowledge base for gaps and produce ranked enrichment priorities.

    Parameters
    ----------
    source_db     : path to ds_wiki.db (read-only)
    type_minimums : override expected minimum counts per entity type;
                    merged over DEFAULT_TYPE_MINIMUMS (your overrides win)

    Usage
    -----
        ga = GapAnalyzer()
        report = ga.analyze()
        print(report.as_markdown())

        # Customise expected counts:
        ga.type_minimums["open_question"] = 20
        report = ga.analyze()
    """

    def __init__(
        self,
        source_db:     Path | str              = SOURCE_DB,
        type_minimums: Optional[Dict[str, int]] = None,
    ) -> None:
        self.source_db = Path(source_db)
        if not self.source_db.exists():
            raise FileNotFoundError(f"source_db not found: {self.source_db}")

        # Merge user overrides over defaults
        self.type_minimums: Dict[str, int] = {**DEFAULT_TYPE_MINIMUMS}
        if type_minimums:
            self.type_minimums.update(type_minimums)

    # ── Heuristics property ────────────────────────────────────────────────────

    @property
    def heuristics(self) -> Dict[str, Callable[[int, int], int]]:
        """
        Return per-entity-type heuristic functions:
            f(observed_count, total_entities) -> expected_minimum

        The default heuristic for unlisted types flags anything with < 2% of
        total entities.  Override by mutating self.type_minimums directly.
        """
        def _default(observed: int, total: int) -> int:
            return max(3, int(total * 0.02))

        result: Dict[str, Callable[[int, int], int]] = {}
        for et, minimum in self.type_minimums.items():
            result[et] = lambda obs, tot, m=minimum: m
        result["_default"] = _default
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.source_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _property_gaps(
        self,
        conn: sqlite3.Connection,
        entity_types: Dict[str, int],
        total_entities: int,
    ) -> List[PropertyGap]:
        """
        For each (entity_type, property_name) pair: flag if coverage < threshold.
        Only includes properties that at least one entry of this type could have.
        """
        # All property names in the DB
        prop_names = [
            r["property_name"]
            for r in conn.execute(
                "SELECT DISTINCT property_name FROM entry_properties ORDER BY property_name"
            ).fetchall()
        ]

        gaps: List[PropertyGap] = []

        for et, type_count in entity_types.items():
            if type_count == 0:
                continue

            # How many entries of this type have each property
            for pname in prop_names:
                filled = conn.execute(
                    """SELECT COUNT(DISTINCT ep.entry_id)
                       FROM entry_properties ep
                       JOIN entries e ON e.id = ep.entry_id
                       WHERE e.entry_type = ?
                         AND ep.property_name = ?
                         AND ep.property_value IS NOT NULL
                         AND ep.property_value != ''""",
                    (et, pname),
                ).fetchone()[0]

                # Skip if this property doesn't exist for this type at all
                if filled == 0:
                    continue

                pct = filled / type_count * 100
                missing = type_count - filled

                if (filled / type_count) < PROPERTY_COVERAGE_THRESHOLD:
                    # Get up to 10 IDs missing this property
                    missing_ids = [
                        r["id"]
                        for r in conn.execute(
                            """SELECT e.id FROM entries e
                               WHERE e.entry_type = ?
                                 AND e.id NOT IN (
                                     SELECT ep.entry_id FROM entry_properties ep
                                     WHERE ep.property_name = ?
                                       AND ep.property_value IS NOT NULL
                                       AND ep.property_value != ''
                                 )
                               ORDER BY e.id
                               LIMIT 10""",
                            (et, pname),
                        ).fetchall()
                    ]

                    # Priority: high if >50% missing, medium if >20%, low otherwise
                    if missing / type_count > 0.5:
                        priority = 1
                    elif missing / type_count > 0.2:
                        priority = 2
                    else:
                        priority = 3

                    gaps.append(PropertyGap(
                        entity_type   = et,
                        property_name = pname,
                        total_of_type = type_count,
                        filled        = filled,
                        missing_count = missing,
                        coverage_pct  = round(pct, 1),
                        missing_ids   = missing_ids,
                        priority      = priority,
                    ))

        # Sort: priority asc (1 first), then missing_count desc
        gaps.sort(key=lambda g: (g.priority, -g.missing_count))
        return gaps

    def _taxonomy_gaps(
        self,
        conn: sqlite3.Connection,
        total_entities: int,
    ) -> List[TaxonomyGap]:
        """
        For each controlled-vocabulary property (archetype, d-sensitivity):
        flag values that appear in < SPARSE_TAXONOMY_THRESHOLD of total entries
        or with absolute count ≤ SPARSE_TAXONOMY_ABS.
        """
        gaps: List[TaxonomyGap] = []

        for pname in TAXONOMY_PROPERTIES:
            rows = conn.execute(
                """SELECT property_value, COUNT(DISTINCT entry_id) c
                   FROM entry_properties
                   WHERE property_name = ?
                     AND property_value IS NOT NULL AND property_value != ''
                   GROUP BY property_value
                   ORDER BY c ASC""",
                (pname,),
            ).fetchall()

            total_filled = sum(r["c"] for r in rows)
            if total_filled == 0:
                continue

            for r in rows:
                val   = r["property_value"]
                count = r["c"]
                pct   = count / total_filled * 100

                if count <= SPARSE_TAXONOMY_ABS:
                    flag = "singleton" if count == 1 else "sparse"
                elif (count / total_entities) < SPARSE_TAXONOMY_THRESHOLD:
                    flag = "sparse"
                else:
                    continue   # not sparse

                gaps.append(TaxonomyGap(
                    taxonomy_name = pname,
                    value         = val,
                    count         = count,
                    pct_of_filled = round(pct, 1),
                    flag          = flag,
                ))

        gaps.sort(key=lambda g: g.count)
        return gaps

    def _type_balance_gaps(
        self,
        conn: sqlite3.Connection,
        entity_types: Dict[str, int],
        total_entities: int,
    ) -> List[TypeBalanceGap]:
        """
        Compare observed entity type counts against expected minimums.
        """
        h = self.heuristics
        gaps: List[TypeBalanceGap] = []

        for et, observed in entity_types.items():
            heuristic = h.get(et, h["_default"])
            expected  = heuristic(observed, total_entities)
            deficit   = max(0, expected - observed)

            if deficit > 0:
                suggestion = (
                    f"Add {deficit} more '{et}' entries to reach minimum of {expected}."
                )
                gaps.append(TypeBalanceGap(
                    entity_type  = et,
                    observed     = observed,
                    expected_min = expected,
                    deficit      = deficit,
                    suggestion   = suggestion,
                ))

        gaps.sort(key=lambda g: -g.deficit)
        return gaps

    def _link_gaps(
        self,
        conn: sqlite3.Connection,
    ) -> List[LinkGap]:
        """
        Find entries that are:
        - isolated: no links in or out
        - no_tier1: have links but none are confidence_tier='1'
        - same_type_only: all links connect to entries of the same entity_type
        """
        # Build link index: entry_id -> list of (other_id, tier, other_type)
        rows = conn.execute(
            """SELECT l.source_id, l.target_id, l.confidence_tier,
                      es.entry_type source_type, et.entry_type target_type
               FROM links l
               LEFT JOIN entries es ON es.id = l.source_id
               LEFT JOIN entries et ON et.id = l.target_id"""
        ).fetchall()

        link_map: Dict[str, List[Tuple[str, str, str]]] = {}  # eid -> [(other, tier, other_type)]
        for r in rows:
            src, tgt, tier = r["source_id"], r["target_id"], r["confidence_tier"] or "original"
            src_type, tgt_type = r["source_type"] or "", r["target_type"] or ""
            link_map.setdefault(src, []).append((tgt, tier, tgt_type))
            link_map.setdefault(tgt, []).append((src, tier, src_type))

        entries = conn.execute(
            "SELECT id, title, entry_type FROM entries ORDER BY id"
        ).fetchall()

        gaps: List[LinkGap] = []

        for e in entries:
            eid   = e["id"]
            title = e["title"] or eid
            etype = e["entry_type"] or "unknown"

            entry_links = link_map.get(eid, [])
            total_links = len(entry_links)

            if total_links == 0:
                gaps.append(LinkGap(
                    entry_id   = eid,
                    title      = title,
                    entity_type = etype,
                    total_links = 0,
                    gap_type   = "isolated",
                    detail     = "No links in or out of this entry.",
                ))
                continue

            # No tier-1 links
            tier1_links = [l for l in entry_links if l[1] == "1"]
            if not tier1_links:
                gaps.append(LinkGap(
                    entry_id   = eid,
                    title      = title,
                    entity_type = etype,
                    total_links = total_links,
                    gap_type   = "no_tier1",
                    detail     = f"Has {total_links} link(s) but none are tier-1 (high-confidence).",
                ))
                continue

            # Same-type-only (only check if ≥ 3 links to avoid false positives on low-degree nodes)
            if total_links >= 3:
                other_types = {l[2] for l in entry_links if l[2]}
                if other_types and all(t == etype for t in other_types):
                    gaps.append(LinkGap(
                        entry_id   = eid,
                        title      = title,
                        entity_type = etype,
                        total_links = total_links,
                        gap_type   = "same_type_only",
                        detail     = (
                            f"All {total_links} links connect to other '{etype}' entries. "
                            "Consider adding cross-type connections."
                        ),
                    ))

        # Sort: isolated first, then no_tier1, then same_type_only
        order = {"isolated": 0, "no_tier1": 1, "same_type_only": 2}
        gaps.sort(key=lambda g: (order.get(g.gap_type, 9), g.entity_type, g.entry_id))
        return gaps

    def _enrichment_priorities(
        self,
        property_gaps:     List[PropertyGap],
        type_balance_gaps: List[TypeBalanceGap],
        link_gaps:         List[LinkGap],
        taxonomy_gaps:     List[TaxonomyGap],
        total_entities:    int,
    ) -> List[EnrichmentPriority]:
        """
        Produce a unified ranked list of enrichment recommendations.

        Impact score formula (0–1):
          - add_property: missing_count / total_entities (normalised per-gap)
          - add_entry:    deficit / (deficit + observed) — severity of imbalance
          - add_link:     1.0 for isolated, 0.6 for no_tier1, 0.3 for same_type_only
          - add_taxonomy: (count ≤ 2 → 0.4 else 0.2)
        """
        items: List[EnrichmentPriority] = []

        # Property gaps
        for g in property_gaps:
            impact = min(1.0, g.missing_count / max(1, total_entities))
            items.append(EnrichmentPriority(
                rank         = 0,
                action       = "add_property",
                target       = f"{g.entity_type}.{g.property_name}",
                description  = (
                    f"Fill '{g.property_name}' for {g.missing_count} {g.entity_type} "
                    f"entries ({g.coverage_pct:.0f}% covered). "
                    f"Start with: {', '.join(g.missing_ids[:3])}"
                    + ("…" if len(g.missing_ids) > 3 else ".")
                ),
                impact_score = round(impact, 3),
            ))

        # Type balance gaps
        for g in type_balance_gaps:
            impact = min(1.0, g.deficit / max(1, g.expected_min))
            items.append(EnrichmentPriority(
                rank         = 0,
                action       = "add_entry",
                target       = g.entity_type,
                description  = g.suggestion,
                impact_score = round(impact, 3),
            ))

        # Link gaps — only surface isolated + no_tier1 in priorities (same_type_only is low priority)
        isolated_count = sum(1 for g in link_gaps if g.gap_type == "isolated")
        no_tier1_count = sum(1 for g in link_gaps if g.gap_type == "no_tier1")

        if isolated_count > 0:
            isolated_ids = [g.entry_id for g in link_gaps if g.gap_type == "isolated"][:5]
            items.append(EnrichmentPriority(
                rank         = 0,
                action       = "add_link",
                target       = "isolated_entries",
                description  = (
                    f"Connect {isolated_count} isolated entries to the graph. "
                    f"Priority: {', '.join(isolated_ids)}"
                    + ("…" if isolated_count > 5 else ".")
                ),
                impact_score = round(min(1.0, isolated_count / max(1, total_entities)), 3),
            ))

        if no_tier1_count > 0:
            items.append(EnrichmentPriority(
                rank         = 0,
                action       = "add_link",
                target       = "no_tier1_entries",
                description  = (
                    f"{no_tier1_count} entries have links but no tier-1 (high-confidence) connections. "
                    "Add at least one tier-1 link per entry."
                ),
                impact_score = round(min(0.6, no_tier1_count / max(1, total_entities) * 2), 3),
            ))

        # Sparse taxonomy gaps
        for g in taxonomy_gaps:
            if g.count <= 1:
                impact = 0.4
                desc = (
                    f"Only 1 entry has '{g.taxonomy_name}={g.value}'. "
                    "Add 2+ more entries with this value or reconsider this taxonomy category."
                )
            else:
                impact = 0.2
                desc = (
                    f"'{g.taxonomy_name}={g.value}' appears in only {g.count} entries "
                    f"({g.pct_of_filled:.1f}%). Consider expanding this category."
                )
            items.append(EnrichmentPriority(
                rank         = 0,
                action       = "add_taxonomy",
                target       = f"{g.taxonomy_name}:{g.value}",
                description  = desc,
                impact_score = impact,
            ))

        # Sort by impact_score descending, then action alphabetically for stable sort
        items.sort(key=lambda x: (-x.impact_score, x.action, x.target))

        # Assign ranks
        for i, item in enumerate(items):
            item.rank = i + 1

        return items

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self) -> GapAnalysisReport:
        """
        Run the full gap analysis and return a GapAnalysisReport.

        All queries are read-only against source_db.
        """
        conn = self._conn()

        try:
            # Core counts
            total_entities = conn.execute(
                "SELECT COUNT(*) FROM entries"
            ).fetchone()[0]

            entity_types: Dict[str, int] = {
                r["entry_type"] or "unknown": r["c"]
                for r in conn.execute(
                    "SELECT entry_type, COUNT(*) c FROM entries "
                    "GROUP BY entry_type ORDER BY c DESC"
                ).fetchall()
            }

            total_links = conn.execute(
                "SELECT COUNT(*) FROM links"
            ).fetchone()[0]

            # Sub-analyses
            prop_gaps     = self._property_gaps(conn, entity_types, total_entities)
            tax_gaps      = self._taxonomy_gaps(conn, total_entities)
            type_gaps     = self._type_balance_gaps(conn, entity_types, total_entities)
            lnk_gaps      = self._link_gaps(conn)
            isolated_count = sum(1 for g in lnk_gaps if g.gap_type == "isolated")

            priorities = self._enrichment_priorities(
                prop_gaps, type_gaps, lnk_gaps, tax_gaps, total_entities
            )

            summary_stats = {
                "total_entities":   total_entities,
                "total_links":      total_links,
                "isolated_count":   isolated_count,
                "property_gap_count":  len(prop_gaps),
                "taxonomy_gap_count":  len(tax_gaps),
                "type_balance_gaps":   len(type_gaps),
                "link_gap_count":      len(lnk_gaps),
            }

        finally:
            conn.close()

        return GapAnalysisReport(
            property_gaps         = prop_gaps,
            taxonomy_gaps         = tax_gaps,
            type_balance_gaps     = type_gaps,
            link_gaps             = lnk_gaps,
            enrichment_priorities = priorities,
            summary_stats         = summary_stats,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ga     = GapAnalyzer()
    report = ga.analyze()
    print(report.as_markdown())
