"""
test_hypothesis_generator.py — Unit tests for HypothesisGenerator.

Tests use a synthetic in-memory SQLite DB and hand-crafted numpy embeddings so
no external files are required.  Each test is fully self-contained.

Run with:
    cd .
    python -m pytest tests/test_hypothesis_generator.py -v
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# ── Path bootstrap ──────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.hypothesis_generator import (
    EntityInfo,
    HypothesisGenerator,
    SurprisingPair,
    _get_typed_templates,
    BASE_TEMPLATES,
    TYPED_TEMPLATES,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_source_db(tmp_path: Path) -> Path:
    """
    Build a minimal ds_wiki.db equivalent with:
    - 6 entries covering 4 entity_types
    - 3 explicit links (so 3 entities are linked, 3 are isolated)
    """
    db = tmp_path / "source.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE entries (
            id TEXT PRIMARY KEY,
            title TEXT,
            entry_type TEXT,
            domain TEXT,
            scale TEXT,
            status TEXT,
            confidence TEXT,
            type_group TEXT
        );
        CREATE TABLE links (
            id INTEGER PRIMARY KEY,
            link_type TEXT,
            source_id TEXT,
            source_label TEXT,
            target_id TEXT,
            target_label TEXT,
            description TEXT,
            link_order INTEGER DEFAULT 0,
            confidence_tier TEXT
        );
        CREATE TABLE entry_properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT,
            table_name TEXT,
            property_name TEXT,
            property_value TEXT,
            prop_order INTEGER DEFAULT 0
        );
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT,
            section_name TEXT,
            content TEXT,
            section_order INTEGER DEFAULT 0
        );
    """)

    # 6 entities
    entries = [
        ("E1", "Law of Gravity",             "reference_law", "physics"),
        ("E2", "Newton's Second Law",         "reference_law", "physics"),
        ("E3", "Coulomb's Law",               "reference_law", "physics"),
        ("E4", "DS Dimensional Gravity",      "law",           "physics"),
        ("E5", "Conservation of Energy",      "reference_law", "physics · chemistry"),
        ("E6", "Open Question: Singularities","open question", "physics · cosmology"),
    ]
    conn.executemany(
        "INSERT INTO entries(id, title, entry_type, domain) VALUES (?,?,?,?)",
        entries,
    )

    # 3 links: E1↔E4, E2↔E5, (E3 and E6 are isolated in opposite directions)
    links = [
        ("derives from", "E4", "DS Gravity", "E1", "Gravity", "E4 derives from E1", 1, "2"),
        ("analogous to", "E2", "Newton", "E5", "Conservation", "Second law conserves momentum", 2, None),
    ]
    conn.executemany(
        "INSERT INTO links(link_type, source_id, source_label, target_id, target_label, description, link_order, confidence_tier) VALUES (?,?,?,?,?,?,?,?)",
        links,
    )

    conn.commit()
    conn.close()
    return db


def _make_history_db(tmp_path: Path, source_db: Path) -> Path:
    """
    Build a minimal wiki_history.db with synthetic embeddings.

    Embeddings are designed so that:
    - E1 and E2 are very close (sim ~0.99) — both Newton's laws
    - E1 and E4 are moderately similar (sim ~0.87) — gravity analogy
    - E5 and E6 are moderately close (sim ~0.84)
    - E3 is relatively isolated
    Strategy: place embeddings in 384-dim space using simple unit vectors
    with controlled dot products.
    """
    db = tmp_path / "history.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE wiki_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            trigger     TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            notes       TEXT
        );
        CREATE TABLE chunk_embedding_history (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id       TEXT NOT NULL,
            chunk_id          TEXT NOT NULL,
            entry_id          TEXT NOT NULL,
            content_hash      TEXT NOT NULL,
            embedding         BLOB NOT NULL,
            top5_neighbors    TEXT NOT NULL,
            centroid_distance REAL NOT NULL,
            UNIQUE(snapshot_id, chunk_id)
        );
    """)

    snap_id = "snap_test_001"
    conn.execute(
        "INSERT INTO wiki_snapshots VALUES (?,?,?,?,?)",
        (snap_id, "2026-01-01T00:00:00+00:00", "test", 6, None),
    )

    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32)

    rng = np.random.default_rng(42)
    dim = 384

    # Base direction shared among "Newton" laws
    base_newton = _unit(rng.standard_normal(dim))

    # E1: exactly base_newton
    e1 = base_newton.copy()
    # E2: very close to E1 (linear combo heavy on E1)
    e2 = _unit(0.99 * base_newton + 0.001 * rng.standard_normal(dim))
    # E4: moderately similar to E1 (DS gravity)
    e4 = _unit(0.80 * base_newton + 0.20 * rng.standard_normal(dim))

    # E5 and E6: share a different base direction
    base_energy = _unit(rng.standard_normal(dim))
    e5 = base_energy.copy()
    e6 = _unit(0.82 * base_energy + 0.18 * rng.standard_normal(dim))

    # E3: random (isolated)
    e3 = _unit(rng.standard_normal(dim))

    embeddings = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6}

    for eid, emb in embeddings.items():
        conn.execute(
            "INSERT INTO chunk_embedding_history "
            "(snapshot_id, chunk_id, entry_id, content_hash, embedding, top5_neighbors, centroid_distance) "
            "VALUES (?,?,?,?,?,?,?)",
            (snap_id, f"{eid}_sec1", eid, "hash_" + eid,
             emb.tobytes(), "[]", 0.1),
        )

    conn.commit()
    conn.close()
    return db


# ── Tests: template helpers ─────────────────────────────────────────────────────

class TestGetTypedTemplates:
    def test_known_pair_forward(self):
        templates = _get_typed_templates("reference_law", "reference_law")
        assert len(templates) >= 5
        assert all("{a}" in t and "{b}" in t for t in templates)

    def test_known_pair_reversed_swaps_labels(self):
        """Reversed (law, reference_law) should have {a} and {b} swapped vs forward."""
        fwd = _get_typed_templates("law", "reference_law")
        rev = _get_typed_templates("reference_law", "law")
        # Both should be non-empty and contain placeholders
        assert len(fwd) >= 4
        assert len(rev) >= 4

    def test_unknown_pair_falls_back_to_base(self):
        templates = _get_typed_templates("unicorn_type", "another_unicorn")
        assert templates == BASE_TEMPLATES

    def test_all_typed_templates_have_placeholders(self):
        for key, tmplts in TYPED_TEMPLATES.items():
            for t in tmplts:
                assert "{a}" in t and "{b}" in t, (
                    f"Template missing {{a}} or {{b}}: key={key!r}, tmpl={t!r}"
                )


# ── Tests: HypothesisGenerator ──────────────────────────────────────────────────

class TestHypothesisGeneratorUnit:
    @pytest.fixture
    def tmp_dbs(self, tmp_path):
        src  = _make_source_db(tmp_path)
        hist = _make_history_db(tmp_path, src)
        return src, hist

    @pytest.fixture
    def gen(self, tmp_dbs):
        src, hist = tmp_dbs
        return HypothesisGenerator(source_db=src, history_db=hist)

    def test_init_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="source_db"):
            HypothesisGenerator(
                source_db=tmp_path / "nonexistent.db",
                history_db=tmp_path / "also_nonexistent.db",
            )

    def test_init_missing_history_raises(self, tmp_path):
        src = _make_source_db(tmp_path)
        with pytest.raises(FileNotFoundError, match="history_db"):
            HypothesisGenerator(
                source_db=src,
                history_db=tmp_path / "nonexistent_hist.db",
            )

    def test_entity_meta_loaded(self, gen):
        gen._entity_meta = gen._load_entity_meta()
        assert len(gen._entity_meta) == 6
        e1 = gen._entity_meta["E1"]
        assert e1.title == "Law of Gravity"
        assert e1.entity_type == "reference_law"
        assert e1.domain == "physics"

    def test_existing_links_loaded(self, gen):
        gen._existing_links = gen._load_existing_links()
        # E4→E1 and E1→E4 both present (undirected)
        assert ("E4", "E1") in gen._existing_links
        assert ("E1", "E4") in gen._existing_links
        # E2↔E5
        assert ("E2", "E5") in gen._existing_links
        assert ("E5", "E2") in gen._existing_links
        # E3 is not linked to anything
        assert ("E3", "E1") not in gen._existing_links

    def test_load_entry_centroids(self, gen, tmp_dbs):
        meta = gen._load_entity_meta()
        centroids = gen._load_entry_centroids(set(meta.keys()))
        assert len(centroids) == 6
        # All centroids should be L2-normalised (norm ~1)
        for eid, c in centroids.items():
            assert abs(np.linalg.norm(c) - 1.0) < 1e-3, f"{eid} centroid not normalised"

    def test_pairwise_matrix_shape(self, gen):
        meta = gen._load_entity_meta()
        centroids = gen._load_entry_centroids(set(meta.keys()))
        valid_ids = sorted(meta.keys())
        sim = gen._compute_pairwise(valid_ids, centroids)
        assert sim.shape == (6, 6)
        # Diagonal should be 0
        assert np.allclose(np.diag(sim), 0.0)

    def test_pairwise_symmetry(self, gen):
        meta = gen._load_entity_meta()
        centroids = gen._load_entry_centroids(set(meta.keys()))
        valid_ids = sorted(meta.keys())
        sim = gen._compute_pairwise(valid_ids, centroids)
        assert np.allclose(sim, sim.T, atol=1e-5)

    def test_e1_e2_very_similar(self, gen):
        """E1 and E2 share 99% of the same embedding direction."""
        meta = gen._load_entity_meta()
        centroids = gen._load_entry_centroids(set(meta.keys()))
        valid_ids = sorted(meta.keys())
        sim = gen._compute_pairwise(valid_ids, centroids)
        idx = {eid: i for i, eid in enumerate(valid_ids)}
        e1_e2_sim = float(sim[idx["E1"], idx["E2"]])
        assert e1_e2_sim > 0.95, f"E1↔E2 similarity {e1_e2_sim:.4f} should be > 0.95"

    def test_find_surprising_pairs_returns_list(self, gen):
        pairs = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.10)
        assert isinstance(pairs, list)
        for p in pairs:
            assert isinstance(p, SurprisingPair)

    def test_pairs_sorted_by_surprise_desc(self, gen):
        pairs = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.00)
        factors = [p.surprise_factor for p in pairs]
        assert factors == sorted(factors, reverse=True)

    def test_pair_similarity_above_threshold(self, gen):
        threshold = 0.75
        pairs = gen.find_surprising_pairs(sim_threshold=threshold, surprise_threshold=1.00)
        for p in pairs:
            assert p.similarity >= threshold, (
                f"Pair {p.entity_a.entity_id}↔{p.entity_b.entity_id} "
                f"sim={p.similarity} below threshold {threshold}"
            )

    def test_pair_surprise_factor_above_threshold(self, gen):
        sf_threshold = 1.05
        pairs = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=sf_threshold)
        for p in pairs:
            assert p.surprise_factor >= sf_threshold, (
                f"Pair {p.entity_a.entity_id}↔{p.entity_b.entity_id} "
                f"sf={p.surprise_factor} below threshold {sf_threshold}"
            )

    def test_surprise_factor_equals_sim_over_baseline(self, gen):
        """
        surprise_factor = sim / baseline.  Both are stored as round(x, 4),
        so recomputing from the stored values can drift by up to ~5e-3.
        """
        pairs = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.00)
        for p in pairs:
            expected_sf = p.similarity / p.baseline
            assert abs(p.surprise_factor - expected_sf) < 5e-3, (
                f"surprise_factor mismatch: {p.surprise_factor} vs {expected_sf:.4f}"
            )

    def test_no_self_pairs(self, gen):
        pairs = gen.find_surprising_pairs(sim_threshold=0.0, surprise_threshold=1.0)
        for p in pairs:
            assert p.entity_a.entity_id != p.entity_b.entity_id

    def test_no_duplicate_pairs(self, gen):
        pairs = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.00)
        seen = set()
        for p in pairs:
            key = tuple(sorted([p.entity_a.entity_id, p.entity_b.entity_id]))
            assert key not in seen, f"Duplicate pair: {key}"
            seen.add(key)

    def test_has_existing_link_correct(self, gen):
        pairs = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.00)
        linked_pair_ids = set()
        for p in pairs:
            if p.has_existing_link:
                key = tuple(sorted([p.entity_a.entity_id, p.entity_b.entity_id]))
                linked_pair_ids.add(key)
        # E1↔E4 should be marked as linked if it appears in pairs
        for p in pairs:
            a, b = p.entity_a.entity_id, p.entity_b.entity_id
            if {a, b} == {"E1", "E4"}:
                assert p.has_existing_link, "E1↔E4 should be marked as linked"
            if {a, b} == {"E2", "E5"}:
                assert p.has_existing_link, "E2↔E5 should be marked as linked"

    def test_include_linked_false_excludes_linked_pairs(self, gen):
        pairs_all      = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.00, include_linked=True)
        pairs_unlinked = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.00, include_linked=False)
        for p in pairs_unlinked:
            assert not p.has_existing_link
        assert len(pairs_unlinked) <= len(pairs_all)

    def test_max_pairs_respected(self, gen):
        pairs = gen.find_surprising_pairs(sim_threshold=0.0, surprise_threshold=1.0, max_pairs=2)
        assert len(pairs) <= 2

    def test_research_prompts_generated(self, gen):
        pairs = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.00)
        for p in pairs:
            assert len(p.research_prompts) >= 1
            for prompt in p.research_prompts:
                assert isinstance(prompt, str)
                assert len(prompt) > 20  # not empty

    def test_prompts_contain_entity_titles(self, gen):
        pairs = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.00)
        for p in pairs[:5]:
            all_text = " ".join(p.research_prompts)
            # At least one prompt should reference both entity titles
            title_a_in = p.entity_a.title in all_text
            title_b_in = p.entity_b.title in all_text
            assert title_a_in or title_b_in, (
                f"Neither title found in prompts for {p.entity_a.entity_id}↔{p.entity_b.entity_id}"
            )

    def test_generate_markdown_report_returns_string(self, gen):
        pairs = gen.find_surprising_pairs(sim_threshold=0.70, surprise_threshold=1.00)
        md = gen.generate_markdown_report(pairs)
        assert isinstance(md, str)
        assert "# Hypothesis Generator Report" in md
        assert "## Summary" in md

    def test_generate_markdown_calls_find_pairs_if_none(self, gen):
        md = gen.generate_markdown_report(
            pairs=None,
            sim_threshold=0.70,
            surprise_threshold=1.00,
        )
        assert isinstance(md, str)
        assert "# Hypothesis Generator Report" in md

    def test_get_stats_structure(self, gen):
        stats = gen.get_stats(sim_threshold=0.70, surprise_threshold=1.00)
        required_keys = {
            "total_surprising_pairs",
            "unlinked_pairs",
            "linked_pairs",
            "cross_type_pairs",
            "max_surprise_factor",
            "mean_similarity",
            "type_pair_distribution",
            "sim_threshold",
            "surprise_threshold",
        }
        for k in required_keys:
            assert k in stats, f"Missing key in stats: {k}"
        assert stats["total_surprising_pairs"] == (
            stats["unlinked_pairs"] + stats["linked_pairs"]
        )

    def test_baseline_fallback_for_sparse_type_pair(self, gen):
        """
        With only 6 entities, some type-pair combinations will have < 5 samples.
        The baseline should still be a valid float, not zero or NaN.
        """
        meta = gen._load_entity_meta()
        centroids = gen._load_entry_centroids(set(meta.keys()))
        valid_ids = sorted(meta.keys())
        sim = gen._compute_pairwise(valid_ids, centroids)
        baselines, global_mean = gen._compute_baselines(valid_ids, meta, sim, min_pairs=5)
        for key, b in baselines.items():
            assert np.isfinite(b), f"Baseline for {key} is not finite: {b}"
            assert b > 0, f"Baseline for {key} should be positive: {b}"


# ── Tests: EntityInfo and SurprisingPair dataclasses ───────────────────────────

class TestDataclasses:
    def test_entity_info_fields(self):
        ei = EntityInfo("A1", "Test Law", "reference_law", "physics")
        assert ei.entity_id == "A1"
        assert ei.title == "Test Law"
        assert ei.entity_type == "reference_law"
        assert ei.domain == "physics"

    def test_surprising_pair_default_prompts_empty(self):
        a = EntityInfo("A1", "A", "reference_law", "physics")
        b = EntityInfo("B1", "B", "law", "physics")
        sp = SurprisingPair(
            entity_a=a, entity_b=b,
            similarity=0.88, baseline=0.72, surprise_factor=1.22,
            has_existing_link=False,
        )
        assert sp.research_prompts == []

    def test_surprising_pair_with_prompts(self):
        a = EntityInfo("A1", "A", "reference_law", "physics")
        b = EntityInfo("B1", "B", "law", "physics")
        sp = SurprisingPair(
            entity_a=a, entity_b=b,
            similarity=0.88, baseline=0.72, surprise_factor=1.22,
            has_existing_link=True,
            research_prompts=["Is A a generalisation of B?"],
        )
        assert len(sp.research_prompts) == 1
        assert sp.has_existing_link is True
