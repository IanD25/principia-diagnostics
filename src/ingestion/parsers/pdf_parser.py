"""
pdf_parser.py — Extract text and sections from scientific paper PDFs.

Provides a structured text extraction layer that feeds into the existing
claim extractor (Phase 3A) and can bootstrap paper-based RRP bundles.

Dependencies:
  PyMuPDF (fitz) — pip install pymupdf

Four modes of use:

  1. extract_text(pdf_path) → plain text string
     Simplest: all text concatenated, page breaks preserved.

  2. extract_sections(pdf_path) → dict[section_name, section_text]
     Heuristic section detection via heading patterns common in scientific
     papers (numbered sections, ALL-CAPS headings, bolded headings).
     Feeds directly into ClaimExtractor.extract_from_sections().

  3. extract_to_claims(pdf_path) → list[Claim]
     End-to-end: PDF → sections → claim extraction.
     Claims start with human_approved=False (mandatory gate).

  4. LLM-assisted segmentation (two-step, Claude-in-the-loop):
     a) prepare_for_llm_segmentation(pdf_path) → numbered paragraphs
        Returns indexed paragraphs for Claude to classify into idea groups.
     b) assemble_llm_sections(paragraphs, groupings) → dict[str, str]
        Takes Claude's grouping response and reassembles verbatim text.
     This avoids regex false positives (table labels, figure numbers) by
     letting the LLM understand semantic structure.

Encoding: UTF-8 throughout. PyMuPDF handles internal PDF encoding.

CLI usage:
    python -m ingestion.parsers.pdf_parser paper.pdf
    python -m ingestion.parsers.pdf_parser paper.pdf --sections
    python -m ingestion.parsers.pdf_parser paper.pdf --claims
    python -m ingestion.parsers.pdf_parser paper.pdf --llm-prep
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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

    Uses sort=True for proper multi-column reading order — PyMuPDF
    reorders text blocks by position (top-to-bottom, left-to-right)
    so two-column layouts read correctly instead of interleaving.

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
            text = page.get_text("text", sort=True)
            pages.append(PDFPage(
                page_number=i + 1,
                text=text,
                char_count=len(text),
            ))
    finally:
        doc.close()

    return pages


def _detect_headers_footers(pages: list[PDFPage], threshold: int = 3) -> set[str]:
    """
    Detect repeated header/footer lines across pages.

    A line appearing on >= `threshold` pages (normalized, stripped) is
    likely a running header or footer. Returns the set of such lines
    for removal.

    Skips very short lines (<5 chars) and very long lines (>120 chars)
    since those are unlikely to be headers/footers.
    """
    from collections import Counter
    line_counts: Counter[str] = Counter()

    for page in pages:
        # Only check first 3 and last 3 lines per page
        lines = page.text.split("\n")
        check_lines = lines[:3] + lines[-3:] if len(lines) > 6 else lines
        seen_on_page: set[str] = set()
        for line in check_lines:
            normalized = line.strip()
            if 5 <= len(normalized) <= 120 and normalized not in seen_on_page:
                seen_on_page.add(normalized)
                line_counts[normalized] += 1

    # Lines appearing on >= threshold pages are headers/footers
    n_pages = len(pages)
    min_occurrences = min(threshold, max(2, n_pages // 3))
    return {line for line, count in line_counts.items() if count >= min_occurrences}


def _strip_headers_footers(text: str, hf_lines: set[str]) -> str:
    """Remove detected header/footer lines from text."""
    if not hf_lines:
        return text
    lines = text.split("\n")
    cleaned = [line for line in lines if line.strip() not in hf_lines]
    return "\n".join(cleaned)


def extract_text(pdf_path: str | Path) -> str:
    """
    Extract all text from a PDF, concatenated with page breaks.

    Applies multi-column reading order (sort=True) and strips
    detected running headers/footers.

    Parameters
    ----------
    pdf_path : Path to the PDF file.

    Returns
    -------
    Full text content of the PDF.
    """
    pages = extract_pages(pdf_path)
    hf_lines = _detect_headers_footers(pages)
    cleaned_pages = [_strip_headers_footers(p.text, hf_lines) for p in pages]
    return "\n\n".join(cleaned_pages)


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


# ── Math formula detection ───────────────────────────────────────────────────

# Indicators that a formula was present but got stripped by PDF extraction.
# PyMuPDF loses LaTeX-rendered equations, leaving orphaned connective text.
_MATH_GHOST_PATTERNS = [
    # Orphaned connective: "where" or "such that" followed by very little
    re.compile(r"(?:^|\.\s+)(?:where|such that|given that|letting|for)\s*$", re.MULTILINE),
    # Lonely equals/inequality on a short line (equation remnant)
    re.compile(r"^\s*[=<>≤≥≈∝±]+\s*$", re.MULTILINE),
    # Parenthetical with nothing inside: "( )" or "(  )"
    re.compile(r"\(\s*\)"),
    # "Eq. N" or "Equation N" reference but no equation visible nearby
    re.compile(r"(?:Eq(?:uation)?\.?\s*\(?(\d+)\)?)", re.IGNORECASE),
    # Sentence ending with "is" or "becomes" or "yields" then whitespace/nothing
    re.compile(r"(?:is|becomes|yields|equals|gives|reads)\s*[,.]?\s*$", re.MULTILINE),
    # Inline unit expressions that lost their number: "dB m−1", "μm", "keV"
    re.compile(r"^\s*(?:dB|μm|nm|keV|MeV|GeV|eV|Hz|GHz|MHz|cm|mm|W|mW|nW)\s*$", re.MULTILINE),
]

# Symbols that indicate math survived extraction (not a ghost)
_MATH_SURVIVOR_CHARS = set("αβγδεζηθλμνξπρσφψωΔΩΣ∫∑∏√≈≤≥±∝∞ℏħ∂∇")


@dataclass
class MathFlag:
    """A detected location where a formula may be garbled or missing."""
    line_number: int
    context: str            # surrounding text (up to ~200 chars)
    detection_type: str     # "ghost" (formula stripped) or "survivor" (symbols present but may be garbled)
    pattern_matched: str    # which pattern or indicator triggered this


def detect_math_regions(text: str) -> list[MathFlag]:
    """
    Scan extracted PDF text for locations where formulas are present or missing.

    Returns a list of MathFlag objects marking each location. These feed into
    the human gate — the reviewer annotates "what the math says" for each flag.

    Two detection modes:
    - "ghost": connective text ("where", "Eq. 3") suggests a formula was here
      but PDF extraction lost it. These MUST be annotated.
    - "survivor": math symbols survived extraction but may be garbled or
      context-free. These SHOULD be annotated for embedding quality.
    """
    flags: list[MathFlag] = []
    lines = text.split("\n")

    # Ghost detection: regex patterns
    for pat in _MATH_GHOST_PATTERNS:
        for match in pat.finditer(text):
            # Find line number
            line_num = text[:match.start()].count("\n") + 1
            # Extract context window
            ctx_start = max(0, match.start() - 80)
            ctx_end = min(len(text), match.end() + 80)
            context = text[ctx_start:ctx_end].replace("\n", " ").strip()

            flags.append(MathFlag(
                line_number=line_num,
                context=context,
                detection_type="ghost",
                pattern_matched=match.group().strip() or pat.pattern[:50],
            ))

    # Survivor detection: lines with math symbols
    for i, line in enumerate(lines):
        survivor_chars = [c for c in line if c in _MATH_SURVIVOR_CHARS]
        if survivor_chars:
            ctx_start = max(0, i - 1)
            ctx_end = min(len(lines), i + 2)
            context = " ".join(lines[ctx_start:ctx_end]).strip()[:200]

            flags.append(MathFlag(
                line_number=i + 1,
                context=context,
                detection_type="survivor",
                pattern_matched=f"symbols: {''.join(set(survivor_chars))}",
            ))

    # Deduplicate flags that are within 3 lines of each other
    flags.sort(key=lambda f: f.line_number)
    deduped: list[MathFlag] = []
    for flag in flags:
        if deduped and abs(flag.line_number - deduped[-1].line_number) < 3:
            # Keep the ghost over survivor (ghost = more urgent)
            if flag.detection_type == "ghost" and deduped[-1].detection_type == "survivor":
                deduped[-1] = flag
            continue
        deduped.append(flag)

    return deduped


def format_math_flags_for_review(flags: list[MathFlag]) -> str:
    """
    Format math flags for human review as part of the mandatory gate.

    Returns markdown showing each detected formula location with its context,
    ready for the reviewer to annotate "what the math says".
    """
    if not flags:
        return "No formula regions detected.\n"

    ghosts = [f for f in flags if f.detection_type == "ghost"]
    survivors = [f for f in flags if f.detection_type == "survivor"]

    lines = [
        "# Math Formula Review — Annotations Required",
        "",
        f"**{len(flags)} formula regions detected** "
        f"({len(ghosts)} missing/garbled, {len(survivors)} partially extracted).",
        "",
        "For each region, provide:",
        "- **formula**: symbolic form (e.g. `J_d = ∂D/∂t`)",
        "- **plain_english**: what the math says in words",
        "- **what_it_constrains**: what physical/logical constraint this imposes",
        "- **ds_wiki_anchor**: DS Wiki entry ID if applicable (e.g. `EM3`)",
        "",
    ]

    if ghosts:
        lines.append("## Missing/Garbled Formulas (MUST annotate)")
        lines.append("")
        for i, flag in enumerate(ghosts, 1):
            lines.append(f"### Math Region {i} (line {flag.line_number})")
            lines.append(f"**Detected:** `{flag.pattern_matched}`")
            lines.append(f"**Context:** ...{flag.context}...")
            lines.append("")

    if survivors:
        lines.append("## Partially Extracted (SHOULD annotate)")
        lines.append("")
        for i, flag in enumerate(survivors, 1):
            lines.append(f"### Math Region {len(ghosts) + i} (line {flag.line_number})")
            lines.append(f"**Symbols found:** `{flag.pattern_matched}`")
            lines.append(f"**Context:** ...{flag.context}...")
            lines.append("")

    lines.append("---")
    lines.append("*Math annotations are stored in the RRP `math_interpretations` table.*")
    lines.append("*Plain-English interpretations are embedded for bridge detection (BGE similarity).*")
    return "\n".join(lines)


# ── LLM-assisted segmentation (Claude-in-the-loop) ──────────────────────────

def _split_paragraphs(text: str) -> list[str]:
    """
    Split reflowed PDF text into semantic paragraphs.

    PyMuPDF uses single \\n for line wrapping and \\n\\n only between pages.
    This splitter detects real paragraph boundaries by looking for:
      - Double newlines (explicit breaks)
      - Sentence-ending punctuation followed by a line starting with uppercase
      - Known section headings on their own line

    Merges very short fragments (< 30 chars) with their predecessor to
    avoid noise from page headers, lone numbers, or figure labels.
    """
    # First split on double-newlines (page boundaries)
    page_chunks = re.split(r"\n\s*\n", text)

    paragraphs: list[str] = []
    for chunk in page_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Within each page chunk, find paragraph breaks:
        # A line ending in sentence-terminal punctuation, followed by a line
        # starting with an uppercase letter (new paragraph or heading)
        sub_paras = re.split(
            r"(?<=[.?!:»\u201d])\n(?=[A-Z\[\(])",
            chunk,
        )
        for sp in sub_paras:
            cleaned = sp.strip()
            if not cleaned:
                continue
            # Unwrap remaining single newlines into spaces (reflowed lines)
            cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
            # Collapse multiple spaces from unwrapping
            cleaned = re.sub(r"  +", " ", cleaned)
            paragraphs.append(cleaned)

    # Merge very short fragments with predecessor
    merged: list[str] = []
    for para in paragraphs:
        if len(para) < 30 and merged:
            merged[-1] += "\n" + para
        else:
            merged.append(para)

    return merged


@dataclass
class LLMSegmentationRequest:
    """Prepared data for Claude to classify into idea groups."""
    paragraphs: list[str]
    preview_text: str       # numbered previews for the LLM prompt
    prompt: str             # full prompt to send to Claude
    total_chars: int
    total_paragraphs: int


def prepare_for_llm_segmentation(
    pdf_path: str | Path,
    preview_chars: int = 300,
) -> LLMSegmentationRequest:
    """
    Prepare numbered paragraphs from a PDF for LLM-based idea grouping.

    Step 1 of the two-step LLM segmentation flow. Extracts text, splits
    into paragraphs, and builds a prompt for Claude to assign each
    paragraph to a semantic idea group.

    Parameters
    ----------
    pdf_path      : Path to PDF file.
    preview_chars : Max chars per paragraph preview in the prompt.

    Returns
    -------
    LLMSegmentationRequest with paragraphs, preview text, and prompt.
    """
    full_text = extract_text(pdf_path)
    paragraphs = _split_paragraphs(full_text)

    # Build numbered previews
    lines = []
    for i, para in enumerate(paragraphs):
        preview = para[:preview_chars].replace("\n", " ")
        if len(para) > preview_chars:
            preview += "..."
        lines.append(f"[{i}] {preview}")
    preview_text = "\n".join(lines)

    prompt = f"""You are parsing a scientific paper into semantic idea groups.

Below are {len(paragraphs)} numbered paragraphs extracted from a PDF.
Each shows the first ~{preview_chars} characters.

YOUR TASK: Group these paragraphs by semantic idea — not by formatting
or headings, but by what the text is actually about. Give each group a
short, descriptive label.

Return ONLY valid JSON — an array of objects:
[
  {{"label": "Title and Author Information", "paragraphs": [0, 1, 2]}},
  {{"label": "Abstract", "paragraphs": [3]}},
  {{"label": "Experimental Setup", "paragraphs": [4, 5, 6, 7]}},
  ...
]

Rules:
- Every paragraph index (0 to {len(paragraphs) - 1}) must appear exactly once
- Paragraphs in each group must be contiguous (no gaps)
- Use descriptive labels that capture the idea, not generic "Section 1"
- Tables, figures, and references can be their own groups
- Return ONLY the JSON array, no other text

PARAGRAPHS:
{preview_text}"""

    return LLMSegmentationRequest(
        paragraphs=paragraphs,
        preview_text=preview_text,
        prompt=prompt,
        total_chars=len(full_text),
        total_paragraphs=len(paragraphs),
    )


def assemble_llm_sections(
    paragraphs: list[str],
    groupings_json: str,
) -> dict[str, str]:
    """
    Reassemble verbatim text from LLM-provided idea groupings.

    Step 2 of the two-step LLM segmentation flow. Takes the original
    paragraphs and Claude's JSON response, returns sections dict.

    Parameters
    ----------
    paragraphs     : Original paragraph list from prepare_for_llm_segmentation.
    groupings_json : JSON string from Claude, array of {label, paragraphs}.

    Returns
    -------
    Ordered dict mapping group_label → verbatim concatenated paragraph text.

    Raises
    ------
    ValueError : If JSON is malformed or paragraph indices are invalid.
    """
    try:
        groupings = json.loads(groupings_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}")

    if not isinstance(groupings, list):
        raise ValueError("Expected a JSON array of grouping objects")

    seen: set[int] = set()
    sections: dict[str, str] = {}

    for group in groupings:
        if not isinstance(group, dict) or "label" not in group or "paragraphs" not in group:
            raise ValueError(f"Each group must have 'label' and 'paragraphs' keys: {group}")

        label = str(group["label"])
        indices = group["paragraphs"]

        if not isinstance(indices, list):
            raise ValueError(f"'paragraphs' must be a list of ints for group '{label}'")

        for idx in indices:
            if not isinstance(idx, int) or idx < 0 or idx >= len(paragraphs):
                raise ValueError(f"Invalid paragraph index {idx} for group '{label}' (valid: 0-{len(paragraphs)-1})")
            if idx in seen:
                raise ValueError(f"Paragraph {idx} assigned to multiple groups")
            seen.add(idx)

        text = "\n\n".join(paragraphs[i] for i in indices)
        if text.strip():
            # Handle duplicate labels
            key = label
            counter = 2
            while key in sections:
                key = f"{label} ({counter})"
                counter += 1
            sections[key] = text.strip()

    # Warn about unassigned paragraphs (non-fatal — append as "Unassigned")
    missing = set(range(len(paragraphs))) - seen
    if missing:
        leftover = "\n\n".join(paragraphs[i] for i in sorted(missing))
        if leftover.strip():
            sections["Unassigned"] = leftover.strip()

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
        "--llm-prep", action="store_true",
        help="Prepare numbered paragraphs for LLM-assisted segmentation"
    )
    parser.add_argument(
        "--min-confidence", choices=["low", "medium", "high"],
        default="low", help="Minimum claim confidence (with --claims)"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    if args.llm_prep:
        req = prepare_for_llm_segmentation(pdf_path)
        print(req.prompt)
        print(f"\n[{req.total_paragraphs} paragraphs, {req.total_chars} chars from {pdf_path.name}]")

    elif args.claims:
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
