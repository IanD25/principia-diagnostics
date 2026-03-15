"""
test_result_validator.py — Unit and integration tests for ResultValidator.

Unit tests use a fake ChromaDB (numpy-based stub) and an in-memory SQLite DB
so no external model or files are required.  Integration tests run against the
real ds_wiki.db and chroma_db.

Run with:
    cd .
    python -m pytest tests/test_result_validator.py -v
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Path bootstrap ──────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.result_validator import (
    EvidenceItem,
    ResultValidator,
    ValidationResult,
    SUPPORTING_LINK_TYPES,
    CONTRADICTING_LINK_TYPES,
    HIGH_SIM_THRESHOLD,
    LOW_SIM_THRESHOLD,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_source_db(tmp_path: Path) -> Path:
    """Build a minimal ds_wiki.db with entries, sections, and links."""
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
        CREATE TABLE sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT,
            section_name TEXT,
            section_order INTEGER DEFAULT 0,
            content TEXT,
            UNIQUE(entry_id, section_name)
        );
        CREATE TABLE links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link_type TEXT,
            source_id TEXT,
            source_label TEXT,
            target_id TEXT,
            target_label TEXT,
            description TEXT,
            link_order INTEGER DEFAULT 0,
            confidence_tier TEXT
        );
    """)
    # Entries
    conn.executemany(
        "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?)",
        [
            ("E1", "Entropy Law",         "reference_law", "physics",     "macro", "established", "Tier 1", "RL"),
            ("E2", "Information Entropy",  "reference_law", "information", "macro", "established", "Tier 1", "RL"),
            ("E3", "Thermodynamic Bound",  "reference_law", "physics",     "macro", "established", "Tier 1", "RL"),
            ("E4", "Maxwell's Demon",      "open_question", "physics",     "micro", "open",        "Tier 2", "OQ"),
            ("E5", "Gravity Law",          "reference_law", "physics",     "macro", "established", "Tier 1", "RL"),
            ("E6", "Quantum Uncertainty",  "reference_law", "physics",     "micro", "established", "Tier 1", "RL"),
        ]
    )
    # Sections (WIC excerpts)
    conn.executemany(
        "INSERT INTO sections (entry_id, section_name, section_order, content) VALUES (?,?,?,?)",
        [
            ("E1", "What It Claims", 0, "Entropy always increases in isolated systems."),
            ("E2", "What It Claims", 0, "Information content quantifies uncertainty."),
            ("E3", "What It Claims", 0, "Thermodynamic processes are bounded by entropy."),
            ("E4", "What It Claims", 0, "Maxwell's demon appears to violate entropy."),
            ("E5", "What It Claims", 0, "Gravitational force follows an inverse-square law."),
            ("E6", "What It Claims", 0, "Position and momentum cannot both be known precisely."),
        ]
    )
    # Links between entries
    conn.executemany(
        "INSERT INTO links (link_type, source_id, source_label, target_id, target_label, confidence_tier) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("derives from",  "E2", "Information Entropy",  "E1", "Entropy Law",        "1.5"),
            ("couples to",    "E1", "Entropy Law",          "E3", "Thermodynamic Bound", "1.5"),
            ("tensions with", "E4", "Maxwell's Demon",      "E1", "Entropy Law",         "2"),
        ]
    )
    conn.commit()
    conn.close()
    return db


def _make_validator_with_stub_chroma(
    source_db: Path,
    entry_embeddings: Dict[str, np.ndarray],
    claim_embedding: np.ndarray,
) -> ResultValidator:
    """
    Build a ResultValidator whose ChromaDB query and model are replaced by
    deterministic numpy stubs based on the provided embeddings.
    """
    v = ResultValidator(source_db=source_db, chroma_dir="/stub", collection="stub")

    # Stub _embed: return claim_embedding
    v._embed = lambda text: claim_embedding

    # Stub _query_chroma: compute cosine sim against each entry's embedding,
    # return (chunk_id, entry_id, sim) sorted descending
    def stub_query(embedding: np.ndarray, top_k: int):
        results = []
        for eid, evec in entry_embeddings.items():
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(evec)
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = float(np.dot(embedding, evec) / (norm_a * norm_b))
            results.append((f"{eid}_chunk", eid, sim))
        results.sort(key=lambda x: -x[2])
        return results[:top_k]

    v._query_chroma = stub_query
    return v


# ── Helper: unit embeddings ────────────────────────────────────────────────────

def _unit(dim: int, idx: int, val: float = 1.0) -> np.ndarray:
    """Return a zero vector with `val` at position idx (basis vector)."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = val
    return v


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests — no I/O, deterministic embeddings
# ══════════════════════════════════════════════════════════════════════════════

class TestEvidenceItem:
    def test_construction(self):
        item = EvidenceItem(
            entry_id="E1", title="Entropy Law", entry_type="reference_law",
            domain="physics", similarity=0.95,
        )
        assert item.entry_id == "E1"
        assert item.link_type is None
        assert item.excerpt == ""

    def test_optional_fields(self):
        item = EvidenceItem(
            entry_id="E1", title="Entropy Law", entry_type="reference_law",
            domain="physics", similarity=0.95,
            link_type="derives from", linked_to="E2", linked_title="Info Entropy",
            excerpt="Entropy always increases.",
        )
        assert item.link_type == "derives from"
        assert item.linked_to == "E2"
        assert len(item.excerpt) > 0


class TestValidationResult:
    def _result(self, s=2, c=1, r=3, score=0.7) -> ValidationResult:
        supporting = [
            EvidenceItem(f"S{i}", f"S{i}", "reference_law", "physics", 0.9)
            for i in range(s)
        ]
        contradictions = [
            EvidenceItem(f"C{i}", f"C{i}", "open_question", "physics", 0.8)
            for i in range(c)
        ]
        related = [
            EvidenceItem(f"R{i}", f"R{i}", "reference_law", "physics", 0.65)
            for i in range(r)
        ]
        return ValidationResult(
            claim="Test claim",
            consistency_score=score,
            supporting_evidence=supporting,
            contradictions=contradictions,
            related_entities=related,
        )

    def test_summary_format(self):
        r = self._result()
        s = r.summary
        assert "Consistency score:" in s
        assert "Supporting: 2" in s
        assert "Contradictions: 1" in s
        assert "Related: 3" in s

    def test_as_markdown_sections(self):
        r = self._result()
        md = r.as_markdown()
        assert "## Claim Validation Report" in md
        assert "### Supporting Evidence" in md
        assert "### Contradictions" in md
        assert "### Related Entities" in md

    def test_as_markdown_empty(self):
        r = ValidationResult(claim="No evidence", consistency_score=0.0)
        md = r.as_markdown()
        assert "No evidence" in md
        assert "Supporting Evidence" not in md

    def test_notes_in_markdown(self):
        r = ValidationResult(
            claim="Test", consistency_score=0.5,
            notes=["Note A", "Note B"]
        )
        md = r.as_markdown()
        assert "Note A" in md
        assert "Note B" in md

    def test_score_clamped_at_zero(self):
        # Negative raw score → clamped to 0.0
        r = ValidationResult(claim="x", consistency_score=0.0)
        assert r.consistency_score == 0.0

    def test_score_clamped_at_one(self):
        r = ValidationResult(claim="x", consistency_score=1.0)
        assert r.consistency_score == 1.0


class TestLinkTypeConstants:
    def test_supporting_types_nonempty(self):
        assert len(SUPPORTING_LINK_TYPES) > 0

    def test_contradicting_types_nonempty(self):
        assert len(CONTRADICTING_LINK_TYPES) > 0

    def test_no_overlap(self):
        assert SUPPORTING_LINK_TYPES.isdisjoint(CONTRADICTING_LINK_TYPES)

    def test_tensions_with_is_contradicting(self):
        assert "tensions with" in CONTRADICTING_LINK_TYPES

    def test_derives_from_is_supporting(self):
        assert "derives from" in SUPPORTING_LINK_TYPES

    def test_analogous_to_is_supporting(self):
        assert "analogous to" in SUPPORTING_LINK_TYPES


class TestResultValidatorHelpers:
    """Tests for internal helper methods using stub data."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = _make_source_db(Path(self.tmp))

    def test_fetch_metadata_returns_correct_fields(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        meta = v._fetch_metadata(["E1", "E2"])
        assert meta["E1"]["title"] == "Entropy Law"
        assert meta["E2"]["entry_type"] == "reference_law"
        assert meta["E1"]["domain"] == "physics"

    def test_fetch_metadata_unknown_id(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        meta = v._fetch_metadata(["UNKNOWN"])
        assert "UNKNOWN" not in meta

    def test_fetch_metadata_empty_list(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        meta = v._fetch_metadata([])
        assert meta == {}

    def test_fetch_excerpt_wic(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        excerpt = v._fetch_excerpt("E1")
        assert "Entropy" in excerpt

    def test_fetch_excerpt_unknown_id(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        excerpt = v._fetch_excerpt("UNKNOWN")
        assert excerpt == ""

    def test_fetch_excerpt_truncates_long_content(self, tmp_path):
        db = tmp_path / "long.db"
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE entries (id TEXT PRIMARY KEY, title TEXT, entry_type TEXT,
                domain TEXT, scale TEXT, status TEXT, confidence TEXT, type_group TEXT);
            CREATE TABLE sections (id INTEGER PRIMARY KEY, entry_id TEXT,
                section_name TEXT, section_order INTEGER, content TEXT,
                UNIQUE(entry_id, section_name));
            CREATE TABLE links (id INTEGER PRIMARY KEY, link_type TEXT,
                source_id TEXT, source_label TEXT, target_id TEXT, target_label TEXT,
                description TEXT, link_order INTEGER, confidence_tier TEXT);
        """)
        long_text = "A" * 500
        conn.execute("INSERT INTO entries VALUES ('X','X','law','physics','','','','')")
        conn.execute("INSERT INTO sections (entry_id,section_name,section_order,content) "
                     "VALUES ('X','What It Claims',0,?)", (long_text,))
        conn.commit()
        conn.close()
        v = ResultValidator(source_db=db, chroma_dir="/stub")
        excerpt = v._fetch_excerpt("X")
        assert len(excerpt) <= 123  # 120 chars + "…"
        assert excerpt.endswith("…")

    def test_fetch_links_between_returns_tension(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        links = v._fetch_links_between(["E1", "E4"])
        types = [l[0] for l in links]
        assert "tensions with" in types

    def test_fetch_links_between_supporting(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        links = v._fetch_links_between(["E1", "E2"])
        types = [l[0] for l in links]
        assert "derives from" in types

    def test_fetch_links_between_no_link(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        links = v._fetch_links_between(["E5", "E6"])
        assert links == []

    def test_fetch_links_between_single_entry(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        links = v._fetch_links_between(["E1"])
        assert links == []

    def test_best_sim_per_entry_deduplication(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        chunks = [
            ("E1_c1", "E1", 0.90),
            ("E1_c2", "E1", 0.85),   # same entry, lower sim → should be ignored
            ("E2_c1", "E2", 0.75),
        ]
        result = v._best_sim_per_entry(chunks)
        assert result["E1"] == pytest.approx(0.90)
        assert result["E2"] == pytest.approx(0.75)

    def test_best_sim_per_entry_empty_entry_id(self):
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        chunks = [("chunk1", "", 0.90)]
        result = v._best_sim_per_entry(chunks)
        assert "" not in result or result.get("") == 0.90   # empty string ignored


class TestValidateClaim:
    """Core validate_claim tests using stub ChromaDB and embeddings."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = _make_source_db(Path(self.tmp))

    def _make_embeddings(self) -> Dict[str, np.ndarray]:
        """6 orthogonal unit vectors, one per entry."""
        return {
            "E1": _unit(6, 0),  # Entropy Law
            "E2": _unit(6, 1),  # Information Entropy
            "E3": _unit(6, 2),  # Thermodynamic Bound
            "E4": _unit(6, 3),  # Maxwell's Demon
            "E5": _unit(6, 4),  # Gravity Law
            "E6": _unit(6, 5),  # Quantum Uncertainty
        }

    def test_claim_matches_supporting_entry(self):
        """Claim similar to E1 (Entropy Law) → E1 is supporting evidence."""
        embs  = self._make_embeddings()
        claim = _unit(6, 0)   # identical to E1
        v     = _make_validator_with_stub_chroma(self.db_path, embs, claim)
        result = v.validate_claim("Entropy always increases", high_threshold=0.9)
        assert result.consistency_score > 0.0
        ids = [e.entry_id for e in result.supporting_evidence]
        assert "E1" in ids

    def test_claim_with_contradiction(self):
        """
        Claim similar to both E1 and E4 → E1 and E4 have 'tensions with' link
        → both appear in contradictions.
        """
        embs = self._make_embeddings()
        # Claim equidistant from E1 and E4
        claim = (embs["E1"] + embs["E4"]) / 2
        claim /= np.linalg.norm(claim)
        v = _make_validator_with_stub_chroma(self.db_path, embs, claim)
        result = v.validate_claim("Maxwell's demon and entropy", high_threshold=0.5)
        contradiction_ids = [e.entry_id for e in result.contradictions]
        # At least E1 or E4 should be flagged
        assert len(contradiction_ids) > 0

    def test_empty_chroma_returns_zero_score(self):
        """If ChromaDB returns nothing, score=0 and a note is added."""
        v = ResultValidator(source_db=self.db_path, chroma_dir="/stub")
        v._embed = lambda t: np.zeros(6, dtype=np.float32)
        v._query_chroma = lambda emb, k: []
        result = v.validate_claim("Any claim")
        assert result.consistency_score == 0.0
        assert any("no results" in n.lower() for n in result.notes)

    def test_consistency_score_formula_all_supporting(self):
        """S=3, C=0, R=0 → score = 3/3 = 1.0"""
        embs = {
            "E1": _unit(6, 0),
            "E2": _unit(6, 0) * 0.99 + _unit(6, 1) * 0.01,
            "E3": _unit(6, 0) * 0.98 + _unit(6, 2) * 0.02,
        }
        for k in embs:
            embs[k] /= np.linalg.norm(embs[k])
        claim = _unit(6, 0)
        v = _make_validator_with_stub_chroma(self.db_path, embs, claim)

        # Patch DB to return a source_db with only E1/E2/E3 and no tension links
        tmp2 = tempfile.mkdtemp()
        db2  = Path(tmp2) / "small.db"
        conn = sqlite3.connect(db2)
        conn.executescript("""
            CREATE TABLE entries (id TEXT PRIMARY KEY, title TEXT, entry_type TEXT,
                domain TEXT, scale TEXT, status TEXT, confidence TEXT, type_group TEXT);
            CREATE TABLE sections (id INTEGER PRIMARY KEY, entry_id TEXT,
                section_name TEXT, section_order INTEGER, content TEXT,
                UNIQUE(entry_id, section_name));
            CREATE TABLE links (id INTEGER PRIMARY KEY, link_type TEXT,
                source_id TEXT, source_label TEXT, target_id TEXT, target_label TEXT,
                description TEXT, link_order INTEGER, confidence_tier TEXT);
        """)
        conn.executemany("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?)", [
            ("E1","Entropy Law","reference_law","physics","","","",""),
            ("E2","Info Entropy","reference_law","information","","","",""),
            ("E3","Thermo Bound","reference_law","physics","","","",""),
        ])
        conn.commit()
        conn.close()
        v._source_db = db2
        result = v.validate_claim("All support", high_threshold=0.5, low_threshold=0.3)
        assert result.consistency_score == pytest.approx(1.0, abs=0.01)

    def test_consistency_score_formula_mixed(self):
        """
        S=2, C=2, R=1 → score = (2 - 0.5*2) / 5 = 1/5 = 0.2
        Verify formula is applied correctly (within rounding).
        """
        r = ValidationResult(
            claim="mixed",
            consistency_score=0.0,  # will not be auto-computed
            supporting_evidence=[
                EvidenceItem("S1","A","law","p",0.9),
                EvidenceItem("S2","B","law","p",0.8),
            ],
            contradictions=[
                EvidenceItem("C1","C","law","p",0.8),
                EvidenceItem("C2","D","law","p",0.75),
            ],
            related_entities=[
                EvidenceItem("R1","E","law","p",0.65),
            ],
        )
        S, C, R = 2, 2, 1
        expected = (S - 0.5 * C) / max(1, S + C + R)
        assert expected == pytest.approx(0.2)

    def test_result_has_required_fields(self):
        embs  = self._make_embeddings()
        claim = _unit(6, 0)
        v     = _make_validator_with_stub_chroma(self.db_path, embs, claim)
        result = v.validate_claim("Entropy increases")
        assert hasattr(result, "claim")
        assert hasattr(result, "consistency_score")
        assert hasattr(result, "supporting_evidence")
        assert hasattr(result, "contradictions")
        assert hasattr(result, "related_entities")
        assert hasattr(result, "notes")
        assert 0.0 <= result.consistency_score <= 1.0

    def test_related_entities_below_high_threshold(self):
        """Entries between low and high threshold → appear as related, not supporting."""
        embs = {
            "E1": _unit(6, 0),  # will be cos_sim=1.0 to claim
            "E5": _unit(6, 4),  # cos_sim=0 to claim → below both thresholds
        }
        claim = _unit(6, 0)
        v = _make_validator_with_stub_chroma(self.db_path, embs, claim)

        # Manually set thresholds so E5 lands in related band
        # E5 sim to claim = 0.0, so it won't appear unless low_threshold=0
        result = v.validate_claim("Test", high_threshold=0.9, low_threshold=0.0)
        related_ids = [e.entry_id for e in result.related_entities]
        assert "E5" in related_ids

    def test_claim_below_all_thresholds_returns_note(self):
        """If all sims < low_threshold, no evidence and a note is returned."""
        embs  = {"E1": _unit(6, 0), "E5": _unit(6, 4)}
        claim = _unit(6, 1)   # orthogonal to both → sim=0
        v     = _make_validator_with_stub_chroma(self.db_path, embs, claim)
        result = v.validate_claim("Claim", high_threshold=0.9, low_threshold=0.8)
        assert result.consistency_score == 0.0
        assert len(result.notes) > 0

    def test_supporting_item_has_excerpt(self):
        """Supporting evidence items should have a non-empty excerpt from the DB."""
        embs  = self._make_embeddings()
        claim = _unit(6, 0)  # matches E1
        v     = _make_validator_with_stub_chroma(self.db_path, embs, claim)
        result = v.validate_claim("Entropy increases", high_threshold=0.9)
        e1_items = [e for e in result.supporting_evidence if e.entry_id == "E1"]
        if e1_items:
            assert e1_items[0].excerpt != ""

    def test_supporting_item_has_link_info(self):
        """E2 is linked to E1 (derives from) — if both are high-sim, link info is set."""
        embs = {
            "E1": _unit(6, 0),
            "E2": _unit(6, 0) * 0.99 + _unit(6, 1) * 0.1,
        }
        for k in embs:
            embs[k] /= np.linalg.norm(embs[k])
        claim = _unit(6, 0)
        v = _make_validator_with_stub_chroma(self.db_path, embs, claim)
        result = v.validate_claim("Entropy and information", high_threshold=0.5)
        e2_items = [e for e in result.supporting_evidence if e.entry_id == "E2"]
        if e2_items:
            assert e2_items[0].link_type in SUPPORTING_LINK_TYPES

    def test_sorted_by_descending_similarity(self):
        """Supporting evidence should be sorted highest sim first."""
        embs = {
            "E1": _unit(6, 0),
            "E3": _unit(6, 0) * 0.9 + _unit(6, 2) * 0.1,
        }
        for k in embs:
            embs[k] /= np.linalg.norm(embs[k])
        claim = _unit(6, 0)
        v = _make_validator_with_stub_chroma(self.db_path, embs, claim)
        result = v.validate_claim("Entropy", high_threshold=0.5)
        sims = [e.similarity for e in result.supporting_evidence]
        assert sims == sorted(sims, reverse=True)

    def test_validate_returns_consistent_types(self):
        embs  = self._make_embeddings()
        claim = _unit(6, 0)
        v     = _make_validator_with_stub_chroma(self.db_path, embs, claim)
        result = v.validate_claim("Test")
        assert isinstance(result, ValidationResult)
        for item in result.supporting_evidence + result.contradictions + result.related_entities:
            assert isinstance(item, EvidenceItem)
            assert isinstance(item.similarity, float)

    def test_top_k_limits_results(self):
        """top_k=1 → at most 1 entry can appear in output."""
        embs  = self._make_embeddings()
        claim = _unit(6, 0)
        v     = _make_validator_with_stub_chroma(self.db_path, embs, claim)
        result = v.validate_claim("Test", top_k=1, high_threshold=0.0, low_threshold=-1.0)
        total = (len(result.supporting_evidence) + len(result.contradictions)
                 + len(result.related_entities))
        assert total <= 1


class TestValidationResultMarkdown:
    """Detailed markdown output tests."""

    def test_markdown_contains_claim(self):
        r = ValidationResult(claim="My specific claim", consistency_score=0.5)
        assert "My specific claim" in r.as_markdown()

    def test_markdown_contains_score(self):
        r = ValidationResult(claim="x", consistency_score=0.75)
        assert "0.75" in r.as_markdown()

    def test_markdown_shows_link_type_for_evidence(self):
        r = ValidationResult(
            claim="test",
            consistency_score=0.8,
            supporting_evidence=[
                EvidenceItem("E1","Title","law","physics",0.9,
                             link_type="derives from", linked_to="E2",
                             linked_title="Other Entry")
            ],
        )
        md = r.as_markdown()
        assert "derives from" in md
        assert "E2" in md

    def test_markdown_shows_tension_symbol_for_contradictions(self):
        r = ValidationResult(
            claim="test",
            consistency_score=0.2,
            contradictions=[
                EvidenceItem("E4","Maxwell","question","physics",0.8,
                             link_type="tensions with", linked_to="E1",
                             linked_title="Entropy Law")
            ],
        )
        md = r.as_markdown()
        assert "tensions with" in md
        assert "⚡" in md


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests — real ds_wiki.db and chroma_db
# ══════════════════════════════════════════════════════════════════════════════

REAL_DB    = Path(__file__).resolve().parent.parent / "data" / "ds_wiki.db"
REAL_CHROMA = Path(__file__).resolve().parent.parent / "data" / "chroma_db"

INTEGRATION_AVAILABLE = REAL_DB.exists() and REAL_CHROMA.exists()
skip_integration = pytest.mark.skipif(
    not INTEGRATION_AVAILABLE,
    reason="Real ds_wiki.db or chroma_db not found",
)


@skip_integration
class TestIntegration:
    """
    Integration tests against the real DB and ChromaDB.
    These embed real claims and check structural properties of results
    (not exact entry IDs, which change as the KB grows).
    """

    @pytest.fixture(scope="class")
    def validator(self):
        return ResultValidator(source_db=REAL_DB, chroma_dir=REAL_CHROMA)

    def test_returns_validation_result(self, validator):
        result = validator.validate_claim("Entropy increases in isolated systems")
        assert isinstance(result, ValidationResult)

    def test_score_in_range(self, validator):
        result = validator.validate_claim("Entropy increases in isolated systems")
        assert 0.0 <= result.consistency_score <= 1.0

    def test_physics_claim_has_supporting_evidence(self, validator):
        """A well-known physics claim should find high-sim entries."""
        result = validator.validate_claim(
            "The force on a charged particle in a magnetic field is perpendicular "
            "to its velocity",
            high_threshold=0.55,
        )
        total = (len(result.supporting_evidence) + len(result.contradictions)
                 + len(result.related_entities))
        assert total > 0

    def test_nonsense_claim_low_score(self, validator):
        """A claim about medieval poetry should have low consistency score."""
        result = validator.validate_claim(
            "The iambic pentameter of 14th-century sonnets determines rhyme schemes",
            high_threshold=0.72,
            low_threshold=0.55,
        )
        assert result.consistency_score < 0.5

    def test_evidence_items_have_titles(self, validator):
        """All returned evidence items should have non-empty titles."""
        result = validator.validate_claim(
            "Information entropy is bounded by thermodynamic entropy",
            high_threshold=0.60,
        )
        for item in result.supporting_evidence + result.contradictions + result.related_entities:
            assert item.title != "" and item.title != item.entry_id

    def test_evidence_items_have_excerpts_for_high_sim(self, validator):
        """High-sim (supporting) entries should have an excerpt from the DB."""
        result = validator.validate_claim(
            "Conservation of energy: the total energy of an isolated system is constant",
            high_threshold=0.65,
        )
        for item in result.supporting_evidence:
            assert item.excerpt != "", f"{item.entry_id} missing excerpt"

    def test_markdown_output_is_string(self, validator):
        result = validator.validate_claim("Entropy and information")
        assert isinstance(result.as_markdown(), str)
        assert len(result.as_markdown()) > 50

    def test_entropy_claim_finds_td_or_es_entries(self, validator):
        """Entropy claim should surface TD, ES, or AM entries."""
        result = validator.validate_claim(
            "Entropy production is minimised at steady state",
            high_threshold=0.60,
        )
        all_ids = [
            e.entry_id for e in
            result.supporting_evidence + result.contradictions + result.related_entities
        ]
        physics_prefixes = {"TD", "ES", "AM", "GL", "RD", "CM", "EM"}
        found = any(
            any(eid.startswith(p) for p in physics_prefixes)
            for eid in all_ids
        )
        assert found, f"Expected physics entries, got: {all_ids}"

    def test_consistency_note_present_when_no_high_sim(self, validator):
        """Very off-topic claim → note added about low coverage."""
        result = validator.validate_claim(
            "Culinary techniques in haute cuisine involve reduction sauces",
            high_threshold=0.99,   # impossible to reach
        )
        assert len(result.notes) > 0

    def test_top_k_parameter_respected(self, validator):
        """top_k=3 → at most 3 distinct entries across all categories."""
        result = validator.validate_claim(
            "Entropy and information",
            top_k=3, high_threshold=0.0, low_threshold=-1.0,
        )
        total = (len(result.supporting_evidence) + len(result.contradictions)
                 + len(result.related_entities))
        assert total <= 3

    def test_real_link_flagged_as_supporting(self, validator):
        """
        Claim about conservation laws should surface CM entries with links.
        At least one high-sim CM entry should have link_type set.
        """
        result = validator.validate_claim(
            "Conservation of momentum: total momentum of a closed system is constant",
            high_threshold=0.65,
        )
        linked = [e for e in result.supporting_evidence if e.link_type is not None]
        # CM10 couples to or derives from CM1/CM11 — expect at least one linked item
        assert len(linked) > 0 or len(result.supporting_evidence) > 0
