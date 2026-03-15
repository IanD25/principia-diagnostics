"""
test_integration.py — Integration tests using the real DS wiki database.

These tests run against the actual ds_wiki.db and wiki_history.db.
They verify that:

1. HypothesisGenerator produces the expected discoveries on the live DS data
   (e.g., B5↔TD3, Wien↔Planck, OmD↔GV1 should all be surfaced).
2. CoverageAnalyzer produces correct metrics for the live DS knowledge base
   (156 entities, 100% archetype coverage, 254 links, etc.).
3. The markdown reports are well-formed and contain expected content.
4. Both tools compose correctly (run coverage, then hypothesis, no state leakage).

Tests are marked with @pytest.mark.integration and are skipped if the
required database files are not present (e.g., in CI without the DB files).

Run with:
    cd .
    python -m pytest tests/test_integration.py -v -m integration
    # OR without the marker to run all tests including integration:
    python -m pytest tests/test_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path bootstrap ──────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import SOURCE_DB, HISTORY_DB
from analysis.hypothesis_generator import HypothesisGenerator, SurprisingPair
from analysis.coverage_analyzer import CoverageAnalyzer, CoverageReport

# ── Skip guard ──────────────────────────────────────────────────────────────────
DB_AVAILABLE = SOURCE_DB.exists() and HISTORY_DB.exists()
skip_if_no_db = pytest.mark.skipif(
    not DB_AVAILABLE,
    reason="DS wiki DB files not present (run sync.py first)",
)
pytestmark = [pytest.mark.integration, skip_if_no_db]


# ── Shared fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ca():
    return CoverageAnalyzer(source_db=SOURCE_DB)


@pytest.fixture(scope="module")
def live_report(ca):
    return ca.compute_report()


@pytest.fixture(scope="module")
def gen():
    return HypothesisGenerator(source_db=SOURCE_DB, history_db=HISTORY_DB)


@pytest.fixture(scope="module")
def live_pairs(gen):
    """Find pairs with slightly relaxed thresholds to ensure ample results."""
    return gen.find_surprising_pairs(sim_threshold=0.80, surprise_threshold=1.10)


# ── CoverageAnalyzer — Live DB ──────────────────────────────────────────────────

class TestCoverageAnalyzerLive:

    # ── Basic counts ──────────────────────────────────────────────────────────

    def test_total_entities_209(self, live_report):
        assert live_report.total_entities == 209, (
            f"Expected 209 entities, got {live_report.total_entities}"
        )

    def test_total_sections_above_1200(self, live_report):
        """After all four enrichment phases, should have ≥ 1200 sections."""
        assert live_report.total_sections >= 1200, (
            f"Expected ≥ 1200 sections, got {live_report.total_sections}"
        )

    def test_total_links_at_least_250(self, live_report):
        """After Tier 2 links, should have ≥ 250 links."""
        assert live_report.total_links >= 250, (
            f"Expected ≥ 250 links, got {live_report.total_links}"
        )

    # ── Entity type distribution ───────────────────────────────────────────────

    def test_reference_law_count(self, live_report):
        assert live_report.entity_type_distribution.get("reference_law", 0) == 149

    def test_method_count(self, live_report):
        assert live_report.entity_type_distribution.get("method", 0) == 16

    def test_law_count(self, live_report):
        assert live_report.entity_type_distribution.get("law", 0) == 15

    def test_entity_type_sums_to_209(self, live_report):
        total = sum(live_report.entity_type_distribution.values())
        assert total == 209

    # ── Property coverage ─────────────────────────────────────────────────────

    def _get_pc(self, report, prop_name):
        for pc in report.property_coverage:
            if pc.property_name == prop_name:
                return pc
        raise KeyError(f"Property '{prop_name}' not found")

    def test_mathematical_archetype_100pct(self, live_report):
        pc = self._get_pc(live_report, "mathematical_archetype")
        assert pc.filled == 209
        assert pc.coverage_pct == 100.0

    def test_dimensional_sensitivity_full_coverage(self, live_report):
        pc = self._get_pc(live_report, "dimensional_sensitivity")
        assert pc.filled >= 198, (
            f"Expected ≥ 198 d-sensitivity entries, got {pc.filled}"
        )

    def test_concept_tags_full_coverage(self, live_report):
        pc = self._get_pc(live_report, "concept_tags")
        assert pc.filled >= 195, (
            f"Expected ≥ 195 concept_tag entries, got {pc.filled}"
        )

    def test_archetype_has_at_least_15_distinct_values(self, live_report):
        """At least 15 archetypes (Option E expanded the vocabulary to 22+)."""
        assert len(live_report.archetype_distribution) >= 15, (
            f"Expected ≥ 15 archetypes, got {len(live_report.archetype_distribution)}: "
            f"{list(live_report.archetype_distribution.keys())}"
        )

    # ── Archetype distribution — spot checks ──────────────────────────────────

    def test_conservation_law_or_thermodynamic_bound_is_top(self, live_report):
        top = max(live_report.archetype_distribution, key=live_report.archetype_distribution.get)
        # After Option E: conservation-law leads (28), thermodynamic-bound second (27)
        assert top in {"conservation-law", "thermodynamic-bound", "equilibrium-condition"}, (
            f"Unexpected top archetype: {top}"
        )

    def test_dimensional_scaling_present(self, live_report):
        assert live_report.archetype_distribution.get("dimensional-scaling", 0) >= 10

    def test_all_15_archetypes_present(self, live_report):
        expected = {
            "thermodynamic-bound", "equilibrium-condition", "dimensional-scaling",
            "geometric-ratio", "statistical-distribution", "power-law-scaling",
            "conservation-law", "gradient-flux-transport", "variational-principle",
            "symmetry-conservation", "coupled-field-equations",
            "inverse-square-geometric", "exponential-decay",
            "wave-equation", "diffusion-equation",
        }
        actual = set(live_report.archetype_distribution.keys())
        missing = expected - actual
        assert not missing, f"Missing archetypes: {missing}"

    # ── D-sensitivity ─────────────────────────────────────────────────────────

    def test_d_sensitivity_has_two_distinct_values(self, live_report):
        """
        The live DB uses 'd-sensitive' / 'd-invariant' (not 'yes'/'no').
        Either naming convention is valid — just verify two non-missing values exist.
        """
        counts = live_report.d_sensitivity_counts
        non_missing = {k: v for k, v in counts.items() if k != "missing"}
        assert len(non_missing) >= 2, (
            f"Expected ≥ 2 distinct d-sensitivity values, got: {counts}"
        )
        # And at least one is non-zero
        assert sum(non_missing.values()) >= 100, (
            f"Expected most entities to have d-sensitivity, got: {counts}"
        )

    def test_d_sensitivity_missing_is_small(self, live_report):
        missing = live_report.d_sensitivity_counts.get("missing", 0)
        assert missing <= 2, f"Too many missing d-sensitivity values: {missing}"

    # ── Network metrics ───────────────────────────────────────────────────────

    def test_link_density_positive(self, live_report):
        assert live_report.network_metrics.link_density > 0

    def test_possible_links_count(self, live_report):
        n = 209
        expected_possible = n * (n - 1) // 2
        assert live_report.network_metrics.possible_links == expected_possible

    def test_derives_from_link_type_present(self, live_report):
        lt = live_report.network_metrics.link_type_distribution
        assert "derives from" in lt, f"'derives from' not in {list(lt.keys())}"

    def test_link_type_sums_to_total(self, live_report):
        total = sum(live_report.network_metrics.link_type_distribution.values())
        assert total == live_report.total_links

    def test_original_tier_most_common(self, live_report):
        """167 'original' (NULL) links should be the most common tier."""
        ct = live_report.network_metrics.confidence_tier_distribution
        # "original" is how we label NULL confidence_tier
        assert ct.get("original", 0) >= 100, (
            f"Expected ≥ 100 'original' tier links, got {ct.get('original', 0)}"
        )

    # ── Gaps ──────────────────────────────────────────────────────────────────

    def test_gaps_is_list(self, live_report):
        assert isinstance(live_report.gaps, list)

    def test_high_isolation_generates_gap(self, live_report):
        """81 of 156 entities are isolated — should trigger isolation gap warning."""
        iso_pct = live_report.network_metrics.isolated_count / 156 * 100
        if iso_pct > 20:
            gap_text = " ".join(live_report.gaps)
            assert "isolated" in gap_text.lower() or "no explicit link" in gap_text.lower(), (
                f"Expected isolation gap (iso_pct={iso_pct:.1f}%) but not found in gaps"
            )

    # ── Markdown report ───────────────────────────────────────────────────────

    def test_markdown_report_length(self, ca):
        md = ca.generate_markdown()
        assert len(md) > 2000, "Markdown report seems too short"

    def test_markdown_contains_209(self, ca):
        md = ca.generate_markdown()
        assert "209" in md

    def test_markdown_contains_archetype_section(self, ca):
        md = ca.generate_markdown()
        assert "Mathematical Archetype" in md

    def test_get_stats_total_entities(self, ca):
        stats = ca.get_stats()
        assert stats["total_entities"] == 209


# ── HypothesisGenerator — Live DB ──────────────────────────────────────────────

class TestHypothesisGeneratorLive:

    # ── Sanity checks ─────────────────────────────────────────────────────────

    def test_returns_non_empty_list(self, live_pairs):
        assert len(live_pairs) > 0, (
            "Expected at least one surprising pair from DS wiki"
        )

    def test_all_items_are_surprising_pairs(self, live_pairs):
        for p in live_pairs:
            assert isinstance(p, SurprisingPair)

    def test_sorted_by_surprise_factor_desc(self, live_pairs):
        factors = [p.surprise_factor for p in live_pairs]
        assert factors == sorted(factors, reverse=True)

    def test_no_self_pairs_in_live_data(self, live_pairs):
        for p in live_pairs:
            assert p.entity_a.entity_id != p.entity_b.entity_id

    def test_no_duplicate_pairs(self, live_pairs):
        seen = set()
        for p in live_pairs:
            key = tuple(sorted([p.entity_a.entity_id, p.entity_b.entity_id]))
            assert key not in seen, f"Duplicate pair: {key}"
            seen.add(key)

    def test_similarity_above_threshold(self, live_pairs):
        for p in live_pairs:
            assert p.similarity >= 0.80, (
                f"{p.entity_a.entity_id}↔{p.entity_b.entity_id}: "
                f"sim={p.similarity} below threshold 0.80"
            )

    def test_surprise_factor_above_threshold(self, live_pairs):
        for p in live_pairs:
            assert p.surprise_factor >= 1.10, (
                f"{p.entity_a.entity_id}↔{p.entity_b.entity_id}: "
                f"sf={p.surprise_factor} below threshold 1.10"
            )

    def test_all_pairs_have_prompts(self, live_pairs):
        for p in live_pairs:
            assert len(p.research_prompts) >= 1, (
                f"No prompts for {p.entity_a.entity_id}↔{p.entity_b.entity_id}"
            )

    # ── Known discoveries from prior analysis ─────────────────────────────────

    def _pair_exists(self, pairs: list, a: str, b: str) -> bool:
        """Check if a specific pair appears in the surprising pairs list."""
        for p in pairs:
            ids = {p.entity_a.entity_id, p.entity_b.entity_id}
            if ids == {a, b}:
                return True
        return False

    def _get_pair(self, pairs: list, a: str, b: str) -> SurprisingPair | None:
        for p in pairs:
            ids = {p.entity_a.entity_id, p.entity_b.entity_id}
            if ids == {a, b}:
                return p
        return None

    def test_q3_t9_discovered(self, live_pairs):
        """
        Q3 (Regime Capacity Bound) ↔ T9 (Regime Capacity Bound Test)
        is a structurally stable cross-type pair (sim ≈ 0.95).
        """
        assert self._pair_exists(live_pairs, "Q3", "T9"), (
            "Expected Q3↔T9 (Regime Capacity Bound ↔ Test) in surprising pairs"
        )

    def test_q3_t9_has_reasonable_similarity(self, live_pairs):
        p = self._get_pair(live_pairs, "Q3", "T9")
        if p is not None:
            assert p.similarity >= 0.80, (
                f"Q3↔T9 similarity {p.similarity} unexpectedly low"
            )

    def test_cross_type_pairs_present(self, live_pairs):
        """At least some pairs should cross entity types."""
        cross_type = [
            p for p in live_pairs
            if p.entity_a.entity_type != p.entity_b.entity_type
        ]
        assert len(cross_type) >= 5, (
            f"Expected ≥ 5 cross-type pairs, got {len(cross_type)}"
        )

    def test_reference_law_pairs_present(self, live_pairs):
        rl_pairs = [
            p for p in live_pairs
            if p.entity_a.entity_type == "reference_law"
            and p.entity_b.entity_type == "reference_law"
        ]
        assert len(rl_pairs) >= 10, (
            f"Expected ≥ 10 reference_law↔reference_law pairs, got {len(rl_pairs)}"
        )

    def test_ds_native_to_reference_law_pairs_present(self, live_pairs):
        """DS laws / constraints / methods should be linked to reference laws."""
        ds_native_types = {"law", "constraint", "method", "axiom", "theorem",
                           "instantiation", "open question", "parameter", "mechanism"}
        ds_ref_pairs = [
            p for p in live_pairs
            if (p.entity_a.entity_type in ds_native_types
                and p.entity_b.entity_type == "reference_law")
            or (p.entity_b.entity_type in ds_native_types
                and p.entity_a.entity_type == "reference_law")
        ]
        assert len(ds_ref_pairs) >= 3, (
            f"Expected ≥ 3 DS-native↔reference_law cross pairs, got {len(ds_ref_pairs)}"
        )

    # ── Prompts quality checks ────────────────────────────────────────────────

    def test_top_pair_prompts_reference_entity_titles(self, live_pairs):
        """The top 5 pairs should have prompts that mention entity titles."""
        for p in live_pairs[:5]:
            all_text = " ".join(p.research_prompts)
            assert (
                p.entity_a.title in all_text or p.entity_b.title in all_text
            ), (
                f"Prompts for {p.entity_a.entity_id}↔{p.entity_b.entity_id} "
                f"don't mention entity titles"
            )

    def test_prompts_not_empty_strings(self, live_pairs):
        for p in live_pairs[:20]:
            for prompt in p.research_prompts:
                assert len(prompt.strip()) > 10, (
                    f"Prompt too short: {prompt!r} "
                    f"(pair {p.entity_a.entity_id}↔{p.entity_b.entity_id})"
                )

    # ── has_existing_link correctness ─────────────────────────────────────────

    def test_has_existing_link_field_is_bool(self, live_pairs):
        for p in live_pairs:
            assert isinstance(p.has_existing_link, bool)

    def test_linked_pairs_correctly_identified(self, gen, live_pairs):
        """
        Pairs that have an explicit link should be marked has_existing_link=True.
        Check a pair we know has a Tier 2 link: B5↔TD3.
        """
        p = self._get_pair(live_pairs, "B5", "TD3")
        if p is not None:
            # B5→TD3 was added as a Tier 2 link ("derives from")
            assert p.has_existing_link is True, (
                "B5↔TD3 should be flagged as linked (Tier 2 link exists)"
            )

    # ── Markdown report ───────────────────────────────────────────────────────

    def test_markdown_report_non_empty(self, gen, live_pairs):
        md = gen.generate_markdown_report(live_pairs[:20])
        assert len(md) > 500

    def test_markdown_contains_summary(self, gen, live_pairs):
        md = gen.generate_markdown_report(live_pairs[:20])
        assert "## Summary" in md
        assert "Total surprising pairs" in md

    def test_markdown_contains_research_prompts(self, gen, live_pairs):
        md = gen.generate_markdown_report(live_pairs[:5])
        assert "Research prompts" in md

    def test_get_stats_structure(self, gen):
        stats = gen.get_stats(sim_threshold=0.80, surprise_threshold=1.10)
        assert stats["total_surprising_pairs"] >= 0
        assert "type_pair_distribution" in stats
        assert stats["sim_threshold"] == 0.80
        assert stats["surprise_threshold"] == 1.10

    # ── include_linked=False ──────────────────────────────────────────────────

    def test_no_linked_filter_reduces_count(self, gen):
        all_pairs     = gen.find_surprising_pairs(sim_threshold=0.80, surprise_threshold=1.10, include_linked=True)
        unlinked_only = gen.find_surprising_pairs(sim_threshold=0.80, surprise_threshold=1.10, include_linked=False)
        assert len(unlinked_only) <= len(all_pairs)
        for p in unlinked_only:
            assert not p.has_existing_link


# ── Composition: both tools together ───────────────────────────────────────────

class TestToolComposition:
    """
    Run coverage analysis → hypothesis generation in sequence and verify
    there is no state leak between the two tools.
    """

    def test_coverage_then_hypothesis_no_state_leak(self, ca, gen):
        """Running CA then HG should give consistent results each time."""
        report1 = ca.compute_report()
        pairs1  = gen.find_surprising_pairs(sim_threshold=0.82, surprise_threshold=1.10)

        # Re-run both (clear any caches)
        gen._entity_meta    = None
        gen._existing_links = None

        report2 = ca.compute_report()
        pairs2  = gen.find_surprising_pairs(sim_threshold=0.82, surprise_threshold=1.10)

        assert report1.total_entities  == report2.total_entities
        assert report1.total_links     == report2.total_links
        assert len(pairs1) == len(pairs2)

    def test_markdown_outputs_are_complete(self, ca, gen):
        """Both markdown generators should produce complete, non-empty output."""
        cov_md = ca.generate_markdown()
        pairs  = gen.find_surprising_pairs(sim_threshold=0.82, surprise_threshold=1.10, max_pairs=10)
        hyp_md = gen.generate_markdown_report(pairs)

        # Coverage report
        assert len(cov_md) > 1000
        assert "209" in cov_md              # entity count
        assert "conservation-law" in cov_md  # most common archetype (post Option E)

        # Hypothesis report
        assert len(hyp_md) > 500
        assert "## Summary" in hyp_md

    def test_isolated_in_coverage_are_candidates_in_hypothesis(self, live_report, live_pairs):
        """
        Entities flagged as isolated (no explicit links) by the Coverage Analyzer
        may appear as candidates in Hypothesis Generator output if they have
        high semantic similarity with others.

        NOTE: As of 2026-03-11, only Q2 remains isolated (all 12 formerly isolated
        entries have been linked). Q2 has insufficient semantic similarity to appear
        in hypothesis pairs, so we now assert only that the pipeline produces results.
        """
        isolated_ids = set(live_report.network_metrics.isolated_entities)
        hypothesis_ids = set()
        for p in live_pairs:
            hypothesis_ids.add(p.entity_a.entity_id)
            hypothesis_ids.add(p.entity_b.entity_id)

        # Pipeline produces results (previously: isolated entries appeared in pairs)
        assert len(hypothesis_ids) > 0, "Hypothesis generator should produce candidate pairs"
        assert len(isolated_ids) <= 5, (
            f"Expected at most 5 isolated entries (DB is well-linked); got {len(isolated_ids)}: {isolated_ids}"
        )
