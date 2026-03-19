"""
test_cross_encoder_rerank.py — Phase 3B cross-encoder reranking tests.

Tests:
    TestCrossEncoderConfig       — config constants exist
    TestCrossEncoderRerank       — reranking logic (mocked model)
    TestCrossEncoderDisabled     — gate respects CROSS_ENCODER_ENABLED=False
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ═══════════════════════════════════════════════════════════════════════════════
# TestCrossEncoderConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossEncoderConfig:
    def test_model_constant_exists(self):
        from config import CROSS_ENCODER_MODEL
        assert "cross-encoder" in CROSS_ENCODER_MODEL

    def test_enabled_constant_exists(self):
        from config import CROSS_ENCODER_ENABLED
        assert isinstance(CROSS_ENCODER_ENABLED, bool)

    def test_rerank_top_k_exists(self):
        from config import RERANK_TOP_K
        assert isinstance(RERANK_TOP_K, int)
        assert RERANK_TOP_K > 0

    def test_rerank_top_k_greater_than_default(self):
        from config import RERANK_TOP_K
        from ingestion.cross_universe_query import TOP_K_CHUNKS
        assert RERANK_TOP_K >= TOP_K_CHUNKS


# ═══════════════════════════════════════════════════════════════════════════════
# TestCrossEncoderRerank
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossEncoderRerank:
    @pytest.fixture
    def query_obj(self):
        """Create a CrossUniverseQuery with mocked internals."""
        from ingestion.cross_universe_query import CrossUniverseQuery
        cq = CrossUniverseQuery.__new__(CrossUniverseQuery)
        cq.bundle_db = Path("/tmp/fake.db")
        cq.chroma_dir = Path("/tmp/fake_chroma")
        cq.collection = "ds_wiki"
        cq.model_name = "fake-model"
        cq._model = None
        cq._coll = None
        cq._reranker = None
        return cq

    def test_rerank_preserves_bge_scores(self, query_obj):
        """Cross-encoder reranking preserves BGE cosine similarity values."""
        candidates = {
            "ENTRY_A": {
                "ds_entry_id": "ENTRY_A",
                "ds_entry_title": "Low BGE, High CE",
                "similarity": 0.70,
                "_chunk_text": "Some text about entry A",
            },
            "ENTRY_B": {
                "ds_entry_id": "ENTRY_B",
                "ds_entry_title": "High BGE, Low CE",
                "similarity": 0.90,
                "_chunk_text": "Some text about entry B",
            },
        }

        # Mock cross-encoder: A gets high score, B gets low
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [2.0, -2.0]
        query_obj._reranker = mock_reranker

        result = query_obj._rerank_candidates(candidates, "test query")

        # BGE scores should be PRESERVED (not replaced)
        assert result["ENTRY_A"]["similarity"] == pytest.approx(0.70)
        assert result["ENTRY_B"]["similarity"] == pytest.approx(0.90)
        # But cross-encoder rank should be stored internally
        assert result["ENTRY_A"]["_ce_rank"] == 0  # ranked first by CE
        assert result["ENTRY_B"]["_ce_rank"] == 1  # ranked second

    def test_rerank_stores_ce_raw_score(self, query_obj):
        """Cross-encoder raw scores are stored for diagnostics."""
        candidates = {
            "E1": {
                "ds_entry_id": "E1",
                "ds_entry_title": "Test",
                "similarity": 0.85,
                "_chunk_text": "content",
            },
        }
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [-5.0]
        query_obj._reranker = mock_reranker

        result = query_obj._rerank_candidates(candidates, "query")
        assert result["E1"]["_ce_raw_score"] == pytest.approx(-5.0)
        assert result["E1"]["similarity"] == pytest.approx(0.85)  # BGE preserved

    def test_rerank_uses_chunk_text(self, query_obj):
        """Cross-encoder should receive (rrp_text, chunk_text) pairs."""
        candidates = {
            "E1": {
                "ds_entry_id": "E1",
                "ds_entry_title": "Fallback Title",
                "similarity": 0.85,
                "_chunk_text": "The actual chunk content",
            },
        }
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [1.0]
        query_obj._reranker = mock_reranker

        query_obj._rerank_candidates(candidates, "my query text")

        # Verify the cross-encoder received the right pairs
        call_args = mock_reranker.predict.call_args
        pairs = call_args[0][0]
        assert pairs[0] == ("my query text", "The actual chunk content")

    def test_rerank_fallback_to_title_when_no_chunk(self, query_obj):
        """If _chunk_text is empty, fall back to ds_entry_title."""
        candidates = {
            "E1": {
                "ds_entry_id": "E1",
                "ds_entry_title": "My DS Wiki Title",
                "similarity": 0.85,
                "_chunk_text": "",
            },
        }
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [1.0]
        query_obj._reranker = mock_reranker

        query_obj._rerank_candidates(candidates, "query")

        pairs = mock_reranker.predict.call_args[0][0]
        assert pairs[0] == ("query", "My DS Wiki Title")

    def test_empty_candidates_no_crash(self, query_obj):
        """Empty candidate dict should not crash."""
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = []
        query_obj._reranker = mock_reranker

        result = query_obj._rerank_candidates({}, "query")
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════════
# TestCrossEncoderDisabled
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossEncoderDisabled:
    @pytest.fixture
    def query_obj(self):
        from ingestion.cross_universe_query import CrossUniverseQuery
        cq = CrossUniverseQuery.__new__(CrossUniverseQuery)
        cq.bundle_db = Path("/tmp/fake.db")
        cq.chroma_dir = Path("/tmp/fake_chroma")
        cq.collection = "ds_wiki"
        cq.model_name = "fake-model"
        cq._model = None
        cq._coll = None
        cq._reranker = None
        return cq

    @patch("ingestion.cross_universe_query.CROSS_ENCODER_ENABLED", False)
    def test_disabled_skips_reranking(self, query_obj):
        """When CROSS_ENCODER_ENABLED=False, _query_ds_wiki uses BGE scores only."""
        # Mock ChromaDB collection
        mock_coll = MagicMock()
        mock_coll.query.return_value = {
            "metadatas": [[
                {"entry_id": "E1", "title": "Entry 1"},
                {"entry_id": "E2", "title": "Entry 2"},
            ]],
            "distances": [[0.3, 0.5]],  # L2 distances
        }
        query_obj._coll = mock_coll

        embedding = np.zeros(1024, dtype=np.float32)
        results = query_obj._query_ds_wiki(embedding, rrp_text="some text")

        # Should NOT have called cross-encoder
        assert query_obj._reranker is None
        # Should have BGE cosine sim values
        assert len(results) == 2
        assert results[0]["similarity"] == pytest.approx(1.0 - 0.15)  # 1 - (0.3/2)

    @patch("ingestion.cross_universe_query.CROSS_ENCODER_ENABLED", True)
    @patch("ingestion.cross_universe_query.RERANK_TOP_K", 20)
    def test_enabled_uses_more_candidates(self, query_obj):
        """When enabled, retrieves RERANK_TOP_K candidates instead of TOP_K_CHUNKS."""
        mock_coll = MagicMock()
        mock_coll.query.return_value = {
            "metadatas": [[{"entry_id": "E1", "title": "Entry 1"}]],
            "distances": [[0.3]],
            "documents": [["chunk text"]],
        }
        query_obj._coll = mock_coll

        # Mock cross-encoder
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [1.0]
        query_obj._reranker = mock_reranker

        embedding = np.zeros(1024, dtype=np.float32)
        query_obj._query_ds_wiki(embedding, rrp_text="query")

        # Verify ChromaDB was queried with RERANK_TOP_K
        call_args = mock_coll.query.call_args
        assert call_args[1]["n_results"] == 20
        assert "documents" in call_args[1]["include"]

    def test_no_rrp_text_skips_reranking(self, query_obj):
        """When rrp_text is None, skip reranking even if enabled."""
        mock_coll = MagicMock()
        mock_coll.query.return_value = {
            "metadatas": [[{"entry_id": "E1", "title": "Entry 1"}]],
            "distances": [[0.3]],
        }
        query_obj._coll = mock_coll

        embedding = np.zeros(1024, dtype=np.float32)
        results = query_obj._query_ds_wiki(embedding)  # no rrp_text

        assert query_obj._reranker is None
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestChunkTextStripping
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkTextStripping:
    def test_internal_fields_stripped_from_output(self):
        """_chunk_text should not appear in final output."""
        from ingestion.cross_universe_query import CrossUniverseQuery
        cq = CrossUniverseQuery.__new__(CrossUniverseQuery)
        cq._reranker = None
        cq._coll = MagicMock()
        cq._coll.query.return_value = {
            "metadatas": [[{"entry_id": "E1", "title": "T1"}]],
            "distances": [[0.3]],
        }

        embedding = np.zeros(1024, dtype=np.float32)
        results = cq._query_ds_wiki(embedding)

        for r in results:
            assert "_chunk_text" not in r
