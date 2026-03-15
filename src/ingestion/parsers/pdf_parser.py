"""
pdf_parser.py — Extract text and sections from scientific paper PDFs.

Provides a structured text extraction layer that feeds into the existing
claim extractor (Phase 3A) and can bootstrap paper-based RRP bundles.

Dependencies:
  PyMuPDF (fitz) — pip install pymupdf

Three modes of use:

  1. extract_text(pdf_path) → plain text string
     Simplest: all text concatenated, page breaks preserved.

  2. extract_sections(pdf_path) → dict[section_name, section_text]
     Heuristic section detection via heading patterns common in scientific
     papers (numbered sections, ALL-CAPS headings, bolded headings).
     Feeds directly into ClaimExtractor.extract_from_sections().

  3. extract_to_claims(pdf_path) → list[Claim]
     End-to-end: PDF → sections → claim extraction.
     Claims start with human_approved=False (mandatory gate).

Encoding: UTF-8 throughout. PyMuPDF handles internal PDF encoding.

CLI usage:
    python -m ingestion.parsers.pdf_parser paper.pdf
    python -m ingestion.parsers.pdf_parser paper.pdf --sections
    python -m ingestion.parsers.pdf_parser paper.pdf --claims
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Section heading detection ────────────────────────────────────────────────

# Patterns for detecting section headings in scientific papers.
# Ordered by specificity — first match wins.
_SECTION_PATTERNS = [
    # Numbered sections: "1. Introduction", "2.1 Methods", "III. Results"
    re.compile(
        r"^(?:"
        r"\d+(?:\.\d+)*\.?\s+"           # Arabic: 1. or 1.2 or 1.2.3
        r"|[IVXLC]+\.\s+"               # Roman: I. or II. or III. (period required)
        r")"
        r"([A-Z][A-Za-z ,&\-:]+)",       # Heading text (no \s to avoid matching \n)
        re.MULTILINE,
    ),
    # ALL-CAPS headings: "ABSTRACT", "INTRODUCTION", "RESULTS AND DISCUSSION"
    re.compile(
        r"^([A-Z]{2}[A-Z ,&\-:]{1,50})$",
        re.MULTILINE,
    ),
]

# Common section names in scientific papers (for fuzzy matching)
_KNOWN_SECTIONS = {
    "abstract", "introduction", "background", "related work",
    "methods", "methodology", "materials and methods", "experimental",
    "experimental setup", "apparatus", "procedure", "data",
    "results", "results and discussion", "discussion",
    "analysis", "findings", "observations",
    "conclusion", "conclusions", "summary",
    "acknowledgments", "acknowledgements",
    "references", "bibliography",
    "appendix", "supplementary",
}


@dataclass
class PDFPage:
    """Extracted content from a single PDF page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int


# ── Core extraction ──────────────────────────────────────────────────────────

def _get_fitz():
    """Import fitz (PyMuPDF) with a helpful error message if not installed."""
    try:
        import fitz
        return fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF parsing.\n"
            "Install it with: pip install pymupdf\n"
            "Or install PFD with PDF support: pip install -e '.[pdf]'"
        )


def extract_pages(pdf_path: str | Path) -> list[PDFPage]:
    """
    Extract text from each page of a PDF.

    Parameters
    ----------
    pdf_path : Path to the PDF file.

    Returns
    -------
    List of PDFPage objects, one per page.
    """
    fitz = _get_fitz()
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    pages = []
    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(PDFPage(
                page_number=i + 1,
                text=text,
                char_count=len(text),
            ))
    finally:
        doc.close()

    return pages


def extract_text(pdf_path: str | Path) -> str:
    """
    Extract all text from a PDF, concatenated with page breaks.

    Parameters
    ----------
    pdf_path : Path to the PDF file.

    Returns
    -------
    Full text content of the PDF.
    """
    pages = extract_pages(pdf_path)
    return "\n\n".join(p.text for p in pages)


def extract_sections(pdf_path: str | Path) -> dict[str, str]:
    """
    Extract text organized by detected section headings.

    Uses heuristic heading detection for scientific papers:
    - Numbered sections (1. Introduction, 2.1 Methods)
    - Roman-numeral sections (I. Introduction, II. Methods)
    - ALL-CAPS headings (ABSTRACT, RESULTS)

    Parameters
    ----------
    pdf_path : Path to the PDF file.

    Returns
    -------
    Ordered dict mapping section_name → section_text.
    If no sections detected, returns {"full_text": <all text>}.
    """
    full_text = extract_text(pdf_path)
    return segment_into_sections(full_text)


def segment_into_sections(text: str) -> dict[str, str]:
    """
    Segment text into sections by detecting heading patterns.

    This is the core logic, separated from PDF extraction so it can
    be tested independently and used on non-PDF text.

    Parameters
    ----------
    text : Full paper text.

    Returns
    -------
    Ordered dict mapping section_name → section_text.
    """
    if not text or not text.strip():
        return {}

    # Find all heading positions
    headings: list[tuple[int, str]] = []

    for pat_idx, pattern in enumerate(_SECTION_PATTERNS):
        for match in pattern.finditer(text):
            heading_text = match.group(0).strip()
            # For numbered/roman patterns (index 0), use the captured group
            # For ALL-CAPS patterns (index 1), use the full match as-is
            if pat_idx == 0 and match.lastindex:
                clean = match.group(1).strip()
            else:
                clean = heading_text
            if not clean:
                continue

            # Validate: skip very short or very long "headings" (likely false positives)
            if len(clean) < 3 or len(clean) > 60:
                continue

            # Skip if it looks like a reference or figure caption
            lower = clean.lower()
            if lower.startswith(("fig", "table", "eq.", "equation")):
                continue

            headings.append((match.start(), clean))

    if not headings:
        return {"full_text": text.strip()}

    # Sort by position and deduplicate nearby headings
    headings.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str]] = []
    for pos, name in headings:
        if deduped and pos - deduped[-1][0] < 20:
            # Too close to previous heading — skip (likely duplicate detection)
            continue
        deduped.append((pos, name))
    headings = deduped

    # Extract text between headings
    sections: dict[str, str] = {}

    # Text before first heading → "preamble"
    preamble = text[: headings[0][0]].strip()
    if preamble and len(preamble) > 50:
        sections["preamble"] = preamble

    for i, (pos, name) in enumerate(headings):
        # Find the end of this section (start of next heading, or end of text)
        if i + 1 < len(headings):
            end = headings[i + 1][0]
        else:
            end = len(text)

        # Extract section content (skip the heading line itself)
        heading_end = text.find("\n", pos)
        if heading_end == -1:
            heading_end = pos + len(name)
        content = text[heading_end:end].strip()

        if content:
            # Handle duplicate section names
            key = name
            counter = 2
            while key in sections:
                key = f"{name} ({counter})"
                counter += 1
            sections[key] = content

    return sections


# ── Claim extraction integration ─────────────────────────────────────────────

def extract_to_claims(
    pdf_path: str | Path,
    min_confidence: str = "low",
) -> list:
    """
    End-to-end: PDF → sections → claim extraction.

    Parameters
    ----------
    pdf_path       : Path to the PDF file.
    min_confidence : "low", "medium", or "high".

    Returns
    -------
    List of Claim objects (human_approved=False).
    """
    from analysis.claim_extractor import ClaimExtractor, ClaimConfidence

    conf_map = {
        "low": ClaimConfidence.LOW,
        "medium": ClaimConfidence.MEDIUM,
        "high": ClaimConfidence.HIGH,
    }
    min_conf = conf_map.get(min_confidence, ClaimConfidence.LOW)

    sections = extract_sections(pdf_path)
    extractor = ClaimExtractor()
    return extractor.extract_from_sections(sections, min_confidence=min_conf)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text and claims from scientific paper PDFs"
    )
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument(
        "--sections", action="store_true",
        help="Show detected sections (instead of raw text)"
    )
    parser.add_argument(
        "--claims", action="store_true",
        help="Extract claims from PDF (end-to-end)"
    )
    parser.add_argument(
        "--min-confidence", choices=["low", "medium", "high"],
        default="low", help="Minimum claim confidence (with --claims)"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    if args.claims:
        claims = extract_to_claims(pdf_path, args.min_confidence)
        from analysis.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor()
        print(extractor.format_for_human_gate(claims))
        print(f"\n[{len(claims)} claims extracted from {pdf_path.name}]")

    elif args.sections:
        sections = extract_sections(pdf_path)
        for name, content in sections.items():
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            preview = content[:500] + ("..." if len(content) > 500 else "")
            print(preview)
        print(f"\n[{len(sections)} sections detected in {pdf_path.name}]")

    else:
        text = extract_text(pdf_path)
        print(text)
        print(f"\n[{len(text)} characters extracted from {pdf_path.name}]")
