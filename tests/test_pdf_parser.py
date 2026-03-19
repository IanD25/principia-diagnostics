"""
Tests for ingestion.parsers.pdf_parser — text extraction and section detection.

PDF extraction tests that require PyMuPDF are marked with pytest.mark.skipif
and will skip gracefully if pymupdf is not installed. The section segmentation
logic is tested independently against plain text (no PDF dependency).
"""

import json
import pytest
from pathlib import Path

# Import the text-processing functions (no PyMuPDF needed)
from ingestion.parsers.pdf_parser import (
    segment_into_sections,
    _KNOWN_SECTIONS,
    detect_math_regions,
    format_math_flags_for_review,
    MathFlag,
    _split_paragraphs,
    assemble_llm_sections,
    _detect_headers_footers,
    _strip_headers_footers,
    PDFPage,
)


# ── Section segmentation tests (no PDF needed) ──────────────────────────────


class TestSegmentIntoSections:
    """Test section detection from plain text."""

    def test_empty_text(self):
        assert segment_into_sections("") == {}
        assert segment_into_sections("   ") == {}

    def test_no_headings(self):
        text = "This is a paragraph with no section headings at all."
        result = segment_into_sections(text)
        assert "full_text" in result
        assert result["full_text"] == text

    def test_numbered_sections(self):
        text = (
            "Some preamble text that comes before everything.\n\n"
            "1. Introduction\n"
            "This is the introduction section with some content.\n\n"
            "2. Methods\n"
            "This describes the methodology used.\n\n"
            "3. Results\n"
            "Here are the results of the study.\n"
        )
        result = segment_into_sections(text)
        assert "Introduction" in result
        assert "Methods" in result
        assert "Results" in result
        assert "introduction" in result["Introduction"].lower()
        assert "methodology" in result["Methods"].lower()

    def test_numbered_subsections(self):
        text = (
            "1. Introduction\n"
            "Intro text.\n\n"
            "1.1 Background\n"
            "Background information.\n\n"
            "1.2 Motivation\n"
            "Why we did this.\n\n"
            "2. Methods\n"
            "Methods text.\n"
        )
        result = segment_into_sections(text)
        assert "Introduction" in result
        assert "Background" in result
        assert "Motivation" in result
        assert "Methods" in result

    def test_roman_numeral_sections(self):
        text = (
            "I. Introduction\n"
            "This paper presents our work.\n\n"
            "II. Experimental Setup\n"
            "We constructed the following apparatus.\n\n"
            "III. Results\n"
            "Our measurements show the following.\n"
        )
        result = segment_into_sections(text)
        assert "Introduction" in result
        assert "Experimental Setup" in result
        assert "Results" in result

    def test_allcaps_sections(self):
        text = (
            "ABSTRACT\n"
            "We present a measurement of something important.\n\n"
            "INTRODUCTION\n"
            "The study of this phenomenon has a long history.\n\n"
            "RESULTS\n"
            "The data shows clear evidence.\n"
        )
        result = segment_into_sections(text)
        assert "ABSTRACT" in result
        assert "INTRODUCTION" in result
        assert "RESULTS" in result

    def test_preamble_captured(self):
        text = (
            "Title of the Paper\n"
            "Author Name, Institution\n"
            "email@example.com\n\n"
            "1. Introduction\n"
            "This is the introduction.\n"
        )
        result = segment_into_sections(text)
        assert "preamble" in result
        assert "Title" in result["preamble"]

    def test_short_preamble_skipped(self):
        text = (
            "Hi\n\n"
            "1. Introduction\n"
            "This is the introduction.\n"
        )
        result = segment_into_sections(text)
        # Preamble too short (< 50 chars) should be skipped
        assert "preamble" not in result

    def test_figure_captions_skipped(self):
        text = (
            "1. Introduction\n"
            "This is the introduction.\n\n"
            "Fig. 1: A diagram of the system\n"
            "More text after the figure.\n\n"
            "2. Methods\n"
            "The methods section.\n"
        )
        result = segment_into_sections(text)
        # "Fig. 1..." should not create its own section
        assert not any("Fig" in k for k in result)

    def test_duplicate_section_names(self):
        text = (
            "1. Results\n"
            "First results section.\n\n"
            "2. Results\n"
            "Second results section.\n"
        )
        result = segment_into_sections(text)
        assert "Results" in result
        assert "Results (2)" in result

    def test_mixed_heading_styles(self):
        text = (
            "ABSTRACT\n"
            "We measure neutrino velocity.\n\n"
            "1. Introduction\n"
            "Background on the measurement.\n\n"
            "2. Experimental Setup\n"
            "The OPERA detector.\n\n"
            "CONCLUSIONS\n"
            "Our findings are significant.\n"
        )
        result = segment_into_sections(text)
        assert len(result) >= 4

    def test_section_content_correct(self):
        text = (
            "1. Introduction\n"
            "Line one of intro.\n"
            "Line two of intro.\n\n"
            "2. Methods\n"
            "Line one of methods.\n"
        )
        result = segment_into_sections(text)
        assert "Line one of intro" in result["Introduction"]
        assert "Line two of intro" in result["Introduction"]
        assert "Line one of methods" in result["Methods"]
        # Methods content should NOT appear in Introduction
        assert "methods" not in result["Introduction"].lower()


class TestKnownSections:
    """Verify the known sections set is populated."""

    def test_common_sections_present(self):
        assert "abstract" in _KNOWN_SECTIONS
        assert "introduction" in _KNOWN_SECTIONS
        assert "methods" in _KNOWN_SECTIONS
        assert "results" in _KNOWN_SECTIONS
        assert "conclusion" in _KNOWN_SECTIONS
        assert "references" in _KNOWN_SECTIONS
        assert "discussion" in _KNOWN_SECTIONS

    def test_known_sections_are_lowercase(self):
        for s in _KNOWN_SECTIONS:
            assert s == s.lower(), f"Section '{s}' should be lowercase"


# ── PyMuPDF-dependent tests ─────────────────────────────────────────────────

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFExtraction:
    """Tests that require PyMuPDF (run only if installed)."""

    def test_import_extract_functions(self):
        from ingestion.parsers.pdf_parser import extract_text, extract_pages, extract_sections
        assert callable(extract_text)
        assert callable(extract_pages)
        assert callable(extract_sections)

    def test_file_not_found(self):
        from ingestion.parsers.pdf_parser import extract_text
        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/file.pdf")

    def test_not_a_pdf(self, tmp_path):
        from ingestion.parsers.pdf_parser import extract_text
        fake = tmp_path / "fake.txt"
        fake.write_text("not a pdf")
        with pytest.raises(ValueError, match="Not a PDF"):
            extract_text(fake)

    def test_extract_from_synthetic_pdf(self, tmp_path):
        """Create a minimal PDF in memory and extract text."""
        from ingestion.parsers.pdf_parser import extract_text, extract_pages

        # Create a simple PDF with PyMuPDF
        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "1. Introduction\nThis is the intro.", fontsize=11)
        page.insert_text((72, 120), "2. Methods\nWe used these methods.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        # Test extract_text
        text = extract_text(pdf_path)
        assert "Introduction" in text
        assert "Methods" in text

        # Test extract_pages
        pages = extract_pages(pdf_path)
        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert pages[0].char_count > 0

    def test_extract_sections_from_synthetic_pdf(self, tmp_path):
        """Section detection on a synthetic PDF."""
        from ingestion.parsers.pdf_parser import extract_sections

        pdf_path = tmp_path / "test_sections.pdf"
        doc = fitz.open()
        page = doc.new_page()
        y = 72
        for line in [
            "1. Introduction",
            "We present a study of particle physics.",
            "",
            "2. Experimental Setup",
            "The detector was configured as follows.",
            "",
            "3. Results",
            "Our measurements confirm the hypothesis.",
        ]:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
        doc.save(str(pdf_path))
        doc.close()

        sections = extract_sections(pdf_path)
        assert len(sections) >= 3

    def test_multipage_pdf(self, tmp_path):
        """Multi-page extraction."""
        from ingestion.parsers.pdf_parser import extract_pages

        pdf_path = tmp_path / "multipage.pdf"
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i+1} content here.", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

        pages = extract_pages(pdf_path)
        assert len(pages) == 3
        assert pages[0].page_number == 1
        assert pages[2].page_number == 3
        assert "Page 1" in pages[0].text
        assert "Page 3" in pages[2].text


@pytest.mark.skipif(not HAS_PYMUPDF, reason="PyMuPDF not installed")
class TestPDFClaimIntegration:
    """Test end-to-end PDF → claims pipeline."""

    def test_extract_to_claims(self, tmp_path):
        from ingestion.parsers.pdf_parser import extract_to_claims

        pdf_path = tmp_path / "claims_test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        y = 72
        lines = [
            "ABSTRACT",
            "We find that the neutrino velocity is consistent with the speed of light.",
            "Our results confirm special relativity predictions.",
            "",
            "RESULTS",
            "The measured arrival time shows a significant deviation from expectations.",
            "This implies that systematic errors were underestimated.",
        ]
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 16
        doc.save(str(pdf_path))
        doc.close()

        claims = extract_to_claims(pdf_path)
        # Should extract at least one claim
        assert len(claims) >= 1
        # All claims should have human_approved=False
        assert all(not c.human_approved for c in claims)


# ── Math detection tests (no PDF needed) ───────────────────────────────────


class TestDetectMathRegions:
    """Test formula ghost and survivor detection."""

    def test_empty_text(self):
        assert detect_math_regions("") == []

    def test_no_math(self):
        text = "This is a regular paragraph about biology with no formulas."
        assert detect_math_regions(text) == []

    def test_ghost_orphaned_where(self):
        """'where' at end of a line signals a stripped formula."""
        # The pattern requires "where" at end of line (after period or start)
        text = "The current is defined.\nwhere\nthe field varies."
        flags = detect_math_regions(text)
        ghosts = [f for f in flags if f.detection_type == "ghost"]
        assert len(ghosts) >= 1

    def test_ghost_empty_parentheses(self):
        """Empty parens '( )' signal stripped inline formula."""
        text = "The piezoelectric coefficient ( ) of BTO is a function of thickness."
        flags = detect_math_regions(text)
        ghosts = [f for f in flags if f.detection_type == "ghost"]
        assert len(ghosts) >= 1

    def test_ghost_equation_reference(self):
        """'Eq. 3' reference signals a formula that should be present."""
        text = "As shown in Eq. 3, the energy is conserved."
        flags = detect_math_regions(text)
        ghosts = [f for f in flags if f.detection_type == "ghost"]
        assert len(ghosts) >= 1

    def test_ghost_sentence_ending_is(self):
        """Sentence ending in 'is' signals stripped formula after it."""
        text = "The total energy of the system is\nconserved under rotation."
        flags = detect_math_regions(text)
        ghosts = [f for f in flags if f.detection_type == "ghost"]
        assert len(ghosts) >= 1

    def test_survivor_greek_letters(self):
        """Greek letters indicate surviving math symbols."""
        text = "The phase angle θ determines the interference pattern."
        flags = detect_math_regions(text)
        survivors = [f for f in flags if f.detection_type == "survivor"]
        assert len(survivors) >= 1
        assert any("θ" in f.pattern_matched for f in survivors)

    def test_survivor_math_operators(self):
        """Math operators like ± ≈ ≤ indicate surviving formulas."""
        text = "The measured value was 99.98% ± 0.01% fidelity."
        flags = detect_math_regions(text)
        survivors = [f for f in flags if f.detection_type == "survivor"]
        assert len(survivors) >= 1

    def test_mixed_ghost_and_survivor(self):
        """Text with both ghosts and survivors on distant lines."""
        text = (
            "The coefficient ( ) is defined.\n"
            "Some filler text here to add distance between detections.\n"
            "More filler text to ensure they are far enough apart.\n"
            "Yet more filler text for spacing.\n"
            "The angle θ exceeds the critical value."
        )
        flags = detect_math_regions(text)
        types = {f.detection_type for f in flags}
        assert "ghost" in types
        assert "survivor" in types

    def test_deduplication_within_3_lines(self):
        """Flags within 3 lines of each other should be deduplicated."""
        text = "The value is\nequal to where\nθ varies."
        flags = detect_math_regions(text)
        # Should be deduplicated — ghost preferred over survivor
        assert len(flags) <= 2

    def test_flag_has_context(self):
        """Each flag should include surrounding text context."""
        text = "The displacement current.\nwhere\nD is the field."
        flags = detect_math_regions(text)
        assert len(flags) >= 1
        assert len(flags[0].context) > 10


class TestFormatMathFlagsForReview:
    """Test the human-readable math review format."""

    def test_no_flags(self):
        result = format_math_flags_for_review([])
        assert "No formula regions" in result

    def test_ghost_flags_formatted(self):
        flags = [MathFlag(
            line_number=42,
            context="...the coefficient ( ) of BTO...",
            detection_type="ghost",
            pattern_matched="( )",
        )]
        result = format_math_flags_for_review(flags)
        assert "Missing/Garbled" in result
        assert "MUST annotate" in result
        assert "line 42" in result

    def test_survivor_flags_formatted(self):
        flags = [MathFlag(
            line_number=10,
            context="...value ± 0.01%...",
            detection_type="survivor",
            pattern_matched="symbols: ±",
        )]
        result = format_math_flags_for_review(flags)
        assert "Partially Extracted" in result
        assert "SHOULD annotate" in result

    def test_annotation_instructions_present(self):
        flags = [MathFlag(
            line_number=1,
            context="test",
            detection_type="ghost",
            pattern_matched="test",
        )]
        result = format_math_flags_for_review(flags)
        assert "formula" in result
        assert "plain_english" in result
        assert "what_it_constrains" in result
        assert "ds_wiki_anchor" in result


class TestLLMSegmentation:
    """Test the LLM-assisted paragraph splitting and assembly."""

    def test_split_paragraphs_double_newline(self):
        """Double newlines split into separate paragraphs (each > 30 chars)."""
        text = (
            "First paragraph with enough text to exceed the merge threshold easily.\n\n"
            "Second paragraph also with enough text to stand on its own as a unit."
        )
        paras = _split_paragraphs(text)
        assert len(paras) == 2
        assert "First" in paras[0]
        assert "Second" in paras[1]

    def test_split_paragraphs_reflowed(self):
        """Single newlines within a page chunk are unwrapped to spaces."""
        text = (
            "This is a sentence that wraps across\n"
            "two lines in the PDF extraction output.\n\n"
            "Next paragraph also has enough text to be a standalone unit here."
        )
        paras = _split_paragraphs(text)
        assert len(paras) == 2
        assert "wraps across two lines" in paras[0]

    def test_split_paragraphs_merges_short(self):
        """Fragments under 30 chars merge with predecessor."""
        text = "A real paragraph of sufficient length here.\n\nHi\n\nAnother paragraph."
        paras = _split_paragraphs(text)
        # "Hi" should merge, not be standalone
        assert not any(p.strip() == "Hi" for p in paras)

    def test_assemble_llm_sections_basic(self):
        paras = ["Title text", "Abstract content", "Intro line 1", "Intro line 2"]
        groupings = json.dumps([
            {"label": "Title", "paragraphs": [0]},
            {"label": "Abstract", "paragraphs": [1]},
            {"label": "Introduction", "paragraphs": [2, 3]},
        ])
        sections = assemble_llm_sections(paras, groupings)
        assert "Title" in sections
        assert "Abstract" in sections
        assert "Introduction" in sections
        assert "Intro line 1" in sections["Introduction"]
        assert "Intro line 2" in sections["Introduction"]

    def test_assemble_llm_sections_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            assemble_llm_sections(["p1"], "not json")

    def test_assemble_llm_sections_duplicate_index(self):
        paras = ["p1", "p2"]
        groupings = json.dumps([
            {"label": "A", "paragraphs": [0, 1]},
            {"label": "B", "paragraphs": [1]},
        ])
        with pytest.raises(ValueError, match="multiple groups"):
            assemble_llm_sections(paras, groupings)

    def test_assemble_llm_sections_out_of_range(self):
        paras = ["p1"]
        groupings = json.dumps([{"label": "A", "paragraphs": [0, 5]}])
        with pytest.raises(ValueError, match="Invalid paragraph index"):
            assemble_llm_sections(paras, groupings)

    def test_assemble_llm_sections_unassigned(self):
        """Unassigned paragraphs go into 'Unassigned' section."""
        paras = ["p1", "p2", "p3"]
        groupings = json.dumps([{"label": "A", "paragraphs": [0]}])
        sections = assemble_llm_sections(paras, groupings)
        assert "Unassigned" in sections
        assert "p2" in sections["Unassigned"]

    def test_assemble_preserves_verbatim(self):
        """Text must be preserved exactly — no summarization."""
        original = "The coefficient (d₃₃) of BTO is 290–470 pC/N for Zr-doped films."
        paras = [original]
        groupings = json.dumps([{"label": "Materials", "paragraphs": [0]}])
        sections = assemble_llm_sections(paras, groupings)
        assert sections["Materials"] == original


# ── Math interpretation in RRP bundle ──────────────────────────────────────


class TestMathInterpretationSchema:
    """Test the math_interpretations table in RRP bundles."""

    def test_fresh_bundle_has_table(self, tmp_path):
        from ingestion.rrp_bundle import create_rrp_bundle

        db = tmp_path / "test.db"
        conn = create_rrp_bundle(str(db), "test", "test", "test")
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "math_interpretations" in tables
        conn.close()

    def test_insert_and_query(self, tmp_path):
        from ingestion.rrp_bundle import create_rrp_bundle

        db = tmp_path / "test.db"
        conn = create_rrp_bundle(str(db), "test", "test", "test")
        conn.execute(
            "INSERT INTO entries (id, title, entry_type) VALUES (?, ?, ?)",
            ("E1", "Test Entry", "reference_law"),
        )
        conn.execute(
            """INSERT INTO math_interpretations
               (entry_id, formula, plain_english, what_it_constrains,
                ds_wiki_anchor, source_section, human_verified)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("E1", "E = mc²", "Energy equals mass times speed of light squared",
             "Mass-energy equivalence", "SR1", "Theory", 1),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT * FROM math_interpretations WHERE entry_id = ?", ("E1",)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["formula"] == "E = mc²"
        assert rows[0]["human_verified"] == 1
        conn.close()

    def test_migration_adds_table(self, tmp_path):
        """Opening an old bundle (no math table) should add it via migration."""
        import sqlite3
        db = tmp_path / "old.db"
        conn = sqlite3.connect(str(db))
        # Create minimal old schema without math_interpretations
        conn.executescript("""
            CREATE TABLE entries (id TEXT PRIMARY KEY, title TEXT, entry_type TEXT);
            CREATE TABLE sections (id INTEGER PRIMARY KEY, entry_id TEXT, section_name TEXT, content TEXT);
            CREATE TABLE links (id INTEGER PRIMARY KEY, link_type TEXT, source_id TEXT, target_id TEXT,
                               stoichiometry_coef REAL);
            CREATE TABLE entry_properties (id INTEGER PRIMARY KEY, entry_id TEXT, property_name TEXT, property_value TEXT);
            CREATE TABLE rrp_meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE cross_universe_bridges (id INTEGER PRIMARY KEY, rrp_entry_id TEXT,
                ds_entry_id TEXT, similarity REAL);
        """)
        conn.commit()
        conn.close()

        from ingestion.rrp_bundle import open_rrp_bundle
        conn = open_rrp_bundle(str(db))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "math_interpretations" in tables
        conn.close()


# ── Header/footer detection tests ────────────────────────────────────────────


class TestHeaderFooterDetection:
    """Test running header/footer detection and stripping."""

    def _make_pages(self, texts: list[str]) -> list[PDFPage]:
        return [PDFPage(page_number=i + 1, text=t, char_count=len(t)) for i, t in enumerate(texts)]

    def test_detects_repeated_header(self):
        pages = self._make_pages([
            "Journal of Science Vol 1\nActual content page 1\nPage 1",
            "Journal of Science Vol 1\nActual content page 2\nPage 2",
            "Journal of Science Vol 1\nActual content page 3\nPage 3",
        ])
        hf = _detect_headers_footers(pages, threshold=3)
        assert "Journal of Science Vol 1" in hf

    def test_does_not_flag_unique_lines(self):
        pages = self._make_pages([
            "Journal of Science Vol 1\nUnique line A\nPage 1",
            "Journal of Science Vol 1\nUnique line B\nPage 2",
            "Journal of Science Vol 1\nUnique line C\nPage 3",
        ])
        hf = _detect_headers_footers(pages, threshold=3)
        assert "Unique line A" not in hf
        assert "Unique line B" not in hf

    def test_skips_short_lines(self):
        pages = self._make_pages([
            "Hi\nDifferent content on page one\nBye",
            "Hi\nSomething else on page two here\nBye",
            "Hi\nYet another thing on page three\nBye",
        ])
        hf = _detect_headers_footers(pages, threshold=3)
        # "Hi" and "Bye" are < 5 chars, should be skipped
        assert "Hi" not in hf
        assert "Bye" not in hf

    def test_strips_detected_lines(self):
        text = "Running Header Here\nActual paragraph content\nRunning Header Here"
        result = _strip_headers_footers(text, {"Running Header Here"})
        assert "Running Header Here" not in result
        assert "Actual paragraph content" in result

    def test_strip_empty_set_passthrough(self):
        text = "Some text\nMore text"
        result = _strip_headers_footers(text, set())
        assert result == text

    def test_alternating_headers(self):
        """Detect alternating odd/even page headers."""
        pages = self._make_pages([
            "OPEN ACCESS Perspective\nPage 1 content\nFooter",
            "Perspective OPEN ACCESS\nPage 2 content\nFooter",
            "OPEN ACCESS Perspective\nPage 3 content\nFooter",
            "Perspective OPEN ACCESS\nPage 4 content\nFooter",
        ])
        hf = _detect_headers_footers(pages, threshold=2)
        assert "OPEN ACCESS Perspective" in hf
        assert "Perspective OPEN ACCESS" in hf

    def test_footer_detected(self):
        """Lines at the end of pages are also checked."""
        pages = self._make_pages([
            "Content\nCopyright 2026 Elsevier",
            "Content\nCopyright 2026 Elsevier",
            "Content\nCopyright 2026 Elsevier",
        ])
        hf = _detect_headers_footers(pages, threshold=3)
        assert "Copyright 2026 Elsevier" in hf
