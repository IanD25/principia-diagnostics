"""
Tests for ingestion.parsers.pdf_parser — text extraction and section detection.

PDF extraction tests that require PyMuPDF are marked with pytest.mark.skipif
and will skip gracefully if pymupdf is not installed. The section segmentation
logic is tested independently against plain text (no PDF dependency).
"""

import pytest
from pathlib import Path

# Import the text-processing functions (no PyMuPDF needed)
from ingestion.parsers.pdf_parser import (
    segment_into_sections,
    _KNOWN_SECTIONS,
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
