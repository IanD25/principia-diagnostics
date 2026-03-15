"""
claim_extractor.py — Extract structured claims from scientific paper prose.

Phase 3A component: parses paper text into structured Claim objects with
subject-relationship-object triples, confidence levels, and polarity hints.

This module does NOT validate claims against the knowledge base — that is
the job of result_validator.py. It only extracts and structures them.

Polarity hints are inferred from linguistic cues in the claim text:
  - Negative markers: "contradicts", "violates", "exceeds", "inconsistent",
    "fails", "disproves", "refutes", "rules out", "incompatible"
  - Positive markers: "confirms", "consistent", "agrees", "validates",
    "supports", "corroborates", "predicts", "reproduces"
  - Neutral: no strong marker detected

These are HEURISTIC — they flag claims for human review, not classify them.

MANDATORY HUMAN GATE: Extracted claims MUST be presented to the user for
confirmation before downstream processing (Foundational Plan §3.1).

Entry points
------------
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(paper_text)
    for claim in claims:
        print(claim)

    # With section awareness:
    claims = extractor.extract_from_sections({
        "abstract": "We measure neutrino velocity...",
        "results": "The neutrino arrival time exceeds...",
    })
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PolarityHint(Enum):
    """Heuristic polarity inferred from linguistic cues."""
    POSITIVE = "positive"       # claim supports/confirms a principle
    NEGATIVE = "negative"       # claim contests/violates a principle
    NEUTRAL = "neutral"         # no strong signal


class ClaimConfidence(Enum):
    """Confidence of the extraction (how clearly structured the source text is)."""
    HIGH = "high"               # explicit claim statement with clear S-R-O
    MEDIUM = "medium"           # inferable claim but requires interpretation
    LOW = "low"                 # weak/implicit claim, may be background context


# ── Polarity markers ─────────────────────────────────────────────────────────

NEGATIVE_MARKERS: list[str] = [
    r"\bcontradicts?\b",
    r"\bviolates?\b",
    r"\bexceeds?\b",
    r"\binconsistent\b",
    r"\bfails?\b",
    r"\bdisproves?\b",
    r"\brefutes?\b",
    r"\brules?\s+out\b",
    r"\bincompatible\b",
    r"\bdisagrees?\b",
    r"\banomal(?:y|ous|ies)\b",
    r"\bdeviat(?:es?|ion)\b",
    r"\btension\b",
    r"\bdiscrepan(?:cy|cies|t)\b",
    r"\bchallenges?\b",
    r"\brejected?\b",
    r"\brunaway\b",
]

POSITIVE_MARKERS: list[str] = [
    r"\bconfirms?\b",
    r"\bconsistent\s+with\b",
    r"\bagrees?\s+with\b",
    r"\bvalidates?\b",
    r"\bsupports?\b",
    r"\bcorroborates?\b",
    r"\bpredicts?\b",
    r"\breproduces?\b",
    r"\bverif(?:y|ies|ied)\b",
    r"\bin\s+(?:good\s+)?agreement\b",
    r"\bcompatible\s+with\b",
    r"\bconverges?\b",
    r"\brecovers?\b",
    r"\bindependently\b",
]

_NEG_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_MARKERS]
_POS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in POSITIVE_MARKERS]


def detect_polarity(text: str) -> tuple[PolarityHint, list[str]]:
    """
    Detect polarity hint from linguistic cues in claim text.

    Returns
    -------
    (PolarityHint, list of matched marker strings)
    """
    neg_matches = []
    pos_matches = []

    for pat in _NEG_PATTERNS:
        m = pat.search(text)
        if m:
            neg_matches.append(m.group())

    for pat in _POS_PATTERNS:
        m = pat.search(text)
        if m:
            pos_matches.append(m.group())

    if neg_matches and not pos_matches:
        return PolarityHint.NEGATIVE, neg_matches
    if pos_matches and not neg_matches:
        return PolarityHint.POSITIVE, pos_matches
    if neg_matches and pos_matches:
        # Both present — flag as negative (more cautious) with all markers
        return PolarityHint.NEGATIVE, neg_matches + pos_matches

    return PolarityHint.NEUTRAL, []


# ── Claim dataclass ──────────────────────────────────────────────────────────

@dataclass
class Claim:
    """A structured claim extracted from paper prose."""
    text: str                                          # original claim text
    subject: str                                       # what the claim is about
    relationship: str                                  # relationship verb/phrase
    object: str                                        # what it relates to
    confidence: ClaimConfidence = ClaimConfidence.MEDIUM
    polarity_hint: PolarityHint = PolarityHint.NEUTRAL
    polarity_markers: list[str] = field(default_factory=list)
    source_section: str = ""                           # which paper section
    human_approved: bool = False                       # GATE: must be True before downstream

    def __str__(self) -> str:
        pol = f" [{self.polarity_hint.value}]" if self.polarity_hint != PolarityHint.NEUTRAL else ""
        gate = " [APPROVED]" if self.human_approved else " [PENDING APPROVAL]"
        return (
            f"Claim({self.confidence.value}{pol}{gate}): "
            f"{self.subject} —[{self.relationship}]→ {self.object}"
        )

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "subject": self.subject,
            "relationship": self.relationship,
            "object": self.object,
            "confidence": self.confidence.value,
            "polarity_hint": self.polarity_hint.value,
            "polarity_markers": self.polarity_markers,
            "source_section": self.source_section,
            "human_approved": self.human_approved,
        }


# ── Sentence splitting ──────────────────────────────────────────────────────

# Abbreviations that should NOT trigger sentence splits
_ABBREVS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "al",
    "fig", "eq", "ref", "vol", "no", "approx", "ca", "cf", "e.g",
    "i.e", "et", "phys", "rev", "lett", "apj", "mnras",
}

_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting common abbreviations."""
    # Simple regex split then recombine if false positive
    raw = _SENTENCE_SPLIT.split(text)
    sentences = []
    buffer = ""
    for chunk in raw:
        if buffer:
            # Check if previous chunk ended with an abbreviation
            last_word = buffer.rstrip(".!?").rsplit(None, 1)[-1].lower() if buffer.strip() else ""
            if last_word in _ABBREVS:
                buffer = buffer + " " + chunk
                continue
            else:
                sentences.append(buffer.strip())
                buffer = chunk
        else:
            buffer = chunk
    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences


# ── Claim sentence detection ────────────────────────────────────────────────

# Patterns that suggest a sentence contains a claim (not background/method)
_CLAIM_INDICATORS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bwe\s+(?:find|show|demonstrate|measure|observe|conclude|report|determine)\b",
        r"\bour\s+(?:results?|measurements?|analysis|data|findings?)\b",
        r"\bthis\s+(?:shows?|demonstrates?|implies?|suggests?|confirms?|indicates?)\b",
        r"\bthe\s+(?:results?|data|measurements?|observations?)\s+(?:show|indicate|suggest|confirm|demonstrate)\b",
        r"\bis\s+consistent\s+with\b",
        r"\bwe\s+(?:can|cannot)\s+(?:rule|exclude|reject)\b",
        r"\bimplies?\s+that\b",
        r"\bprovides?\s+evidence\b",
        r"\bsignificant(?:ly)?\b",
        r"\bat\s+\d+(?:\.\d+)?[σ%]\b",            # statistical significance
        r"\bp\s*[<>=]\s*0\.\d+\b",                  # p-value
        r"\bconfidence\s+(?:level|interval)\b",
        r"\b(?:the\s+)?model\s+(?:recovers?|predicts?|reproduces?|confirms?|validates?)\b",
        r"\b(?:accurately|correctly|successfully)\s+(?:recovers?|predicts?|reproduces?)\b",
    ]
]

# Patterns that suggest background/context (not a claim)
_BACKGROUND_INDICATORS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bit\s+is\s+well\s+known\b",
        r"\bprevious(?:ly)?\s+(?:shown|demonstrated|established)\b",
        r"\baccording\s+to\b",
        r"\b(?:in|from)\s+(?:the\s+)?(?:literature|previous\s+work|earlier\s+studies)\b",
        r"\bstandard\s+model\b",
        r"\bclassical(?:ly)?\b",
        r"\bconventional(?:ly)?\b",
    ]
]


def is_claim_sentence(sentence: str) -> tuple[bool, ClaimConfidence]:
    """
    Heuristic: does this sentence contain a scientific claim?

    Returns (is_claim, confidence).
    """
    claim_hits = sum(1 for p in _CLAIM_INDICATORS if p.search(sentence))
    bg_hits = sum(1 for p in _BACKGROUND_INDICATORS if p.search(sentence))

    if claim_hits >= 2:
        return True, ClaimConfidence.HIGH
    if claim_hits == 1 and bg_hits == 0:
        return True, ClaimConfidence.MEDIUM
    if claim_hits == 1 and bg_hits >= 1:
        # Mixed signals — might be a claim referencing background
        return True, ClaimConfidence.LOW
    return False, ClaimConfidence.LOW


# ── Subject-Relationship-Object extraction ──────────────────────────────────

# Simple pattern-based SRO extraction
_SRO_PATTERNS = [
    # "X is consistent with Y"
    re.compile(
        r"(?P<subject>.+?)\s+(?:is|are)\s+(?P<relationship>consistent\s+with|"
        r"incompatible\s+with|in\s+agreement\s+with|in\s+tension\s+with)\s+(?P<object>.+)",
        re.IGNORECASE,
    ),
    # "X confirms/validates/contradicts Y"
    re.compile(
        r"(?P<subject>.+?)\s+(?P<relationship>confirms?|validates?|contradicts?|"
        r"supports?|refutes?|challenges?|reproduces?|predicts?|violates?|"
        r"exceeds?|rules?\s+out|corroborates?|recovers?|agrees?\s+with|"
        r"disagrees?\s+with)\s+(?P<object>.+)",
        re.IGNORECASE,
    ),
    # "We find/show/demonstrate that X"
    re.compile(
        r"[Ww]e\s+(?:find|show|demonstrate|measure|observe|conclude|report)\s+(?:that\s+)?(?P<subject>.+?)\s+"
        r"(?P<relationship>is|are|equals?|has|have|can|cannot|does|do|shows?)\s+(?P<object>.+)",
        re.IGNORECASE,
    ),
    # "X implies Y"
    re.compile(
        r"(?P<subject>.+?)\s+(?P<relationship>implies?|suggests?|indicates?|"
        r"demonstrates?|provides?\s+evidence\s+for)\s+(?:that\s+)?(?P<object>.+)",
        re.IGNORECASE,
    ),
]


def extract_sro(sentence: str) -> tuple[str, str, str] | None:
    """
    Attempt to extract (subject, relationship, object) from a claim sentence.

    Returns None if no pattern matches.
    """
    for pat in _SRO_PATTERNS:
        m = pat.search(sentence)
        if m:
            groups = m.groupdict()
            subj = groups.get("subject", "").strip().rstrip(",;:")
            rel = groups.get("relationship", "").strip()
            obj = groups.get("object", "").strip().rstrip(".,;:")
            # Sanity: skip if subject or object is too short or too long
            if len(subj) < 3 or len(obj) < 3:
                continue
            if len(subj) > 200 or len(obj) > 200:
                # Truncate long extractions
                subj = subj[:200]
                obj = obj[:200]
            return subj, rel, obj
    return None


# ── ClaimExtractor ──────────────────────────────────────────────────────────

class ClaimExtractor:
    """
    Extract structured claims from scientific paper prose.

    This is a pattern-based extractor (no LLM required). It uses:
    1. Sentence splitting
    2. Claim sentence detection (claim vs. background heuristic)
    3. Subject-Relationship-Object extraction
    4. Polarity hint detection

    For LLM-enhanced extraction, use extract_claims_with_context()
    which formats claims for LLM review.
    """

    def extract_claims(
        self,
        text: str,
        source_section: str = "",
        min_confidence: ClaimConfidence = ClaimConfidence.LOW,
    ) -> list[Claim]:
        """
        Extract claims from a block of text.

        Parameters
        ----------
        text            : Paper prose to extract claims from.
        source_section  : Section label (e.g. "abstract", "results").
        min_confidence  : Minimum extraction confidence to include.

        Returns
        -------
        List of Claim objects (human_approved=False — MUST be approved before use).
        """
        if not text or not text.strip():
            return []

        sentences = split_sentences(text)
        claims = []

        confidence_order = {
            ClaimConfidence.HIGH: 2,
            ClaimConfidence.MEDIUM: 1,
            ClaimConfidence.LOW: 0,
        }
        min_conf_val = confidence_order[min_confidence]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # too short to be a claim
                continue

            is_claim, conf = is_claim_sentence(sentence)
            if not is_claim:
                continue
            if confidence_order[conf] < min_conf_val:
                continue

            # Try SRO extraction
            sro = extract_sro(sentence)
            if sro is None:
                # Still a detected claim sentence but no structured SRO
                # Include as unstructured claim
                polarity, markers = detect_polarity(sentence)
                claims.append(Claim(
                    text=sentence,
                    subject=sentence[:80],
                    relationship="claims",
                    object="(unstructured — needs human review)",
                    confidence=ClaimConfidence.LOW,
                    polarity_hint=polarity,
                    polarity_markers=markers,
                    source_section=source_section,
                    human_approved=False,
                ))
                continue

            subj, rel, obj = sro
            polarity, markers = detect_polarity(sentence)

            claims.append(Claim(
                text=sentence,
                subject=subj,
                relationship=rel,
                object=obj,
                confidence=conf,
                polarity_hint=polarity,
                polarity_markers=markers,
                source_section=source_section,
                human_approved=False,
            ))

        return claims

    def extract_from_sections(
        self,
        sections: dict[str, str],
        min_confidence: ClaimConfidence = ClaimConfidence.LOW,
    ) -> list[Claim]:
        """
        Extract claims from multiple paper sections.

        Parameters
        ----------
        sections : dict mapping section_name → section_text
        min_confidence : Minimum extraction confidence.

        Returns
        -------
        List of Claim objects across all sections.
        """
        all_claims = []
        for section_name, section_text in sections.items():
            claims = self.extract_claims(
                section_text,
                source_section=section_name,
                min_confidence=min_confidence,
            )
            all_claims.extend(claims)
        return all_claims

    def format_for_human_gate(self, claims: list[Claim]) -> str:
        """
        Format extracted claims for human review (mandatory gate).

        Returns a markdown string listing all claims with polarity hints
        and extraction confidence, ready for user approval/rejection.
        """
        if not claims:
            return "No claims extracted.\n"

        lines = [
            "# Extracted Claims — Human Review Required",
            "",
            f"**{len(claims)} claims extracted.** Review each and mark approved/rejected.",
            "",
        ]

        for i, claim in enumerate(claims, 1):
            status = "APPROVED" if claim.human_approved else "PENDING"
            pol_icon = {
                PolarityHint.POSITIVE: "+",
                PolarityHint.NEGATIVE: "!",
                PolarityHint.NEUTRAL: "~",
            }[claim.polarity_hint]

            lines.append(f"## Claim {i} [{status}] [{pol_icon} {claim.polarity_hint.value}]")
            lines.append(f"**Section:** {claim.source_section or '(unknown)'}")
            lines.append(f"**Confidence:** {claim.confidence.value}")
            lines.append(f"**Text:** {claim.text}")
            lines.append(f"**Subject:** {claim.subject}")
            lines.append(f"**Relationship:** {claim.relationship}")
            lines.append(f"**Object:** {claim.object}")
            if claim.polarity_markers:
                lines.append(f"**Polarity markers:** {', '.join(claim.polarity_markers)}")
            lines.append("")

        lines.append("---")
        lines.append("*Claims must be approved before downstream processing (PFD Foundational Plan §3.1).*")
        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract structured claims from paper text"
    )
    parser.add_argument("text", nargs="?", help="Text to extract claims from (or use --file)")
    parser.add_argument("--file", "-f", help="Read text from file")
    parser.add_argument(
        "--min-confidence",
        choices=["low", "medium", "high"],
        default="low",
        help="Minimum claim confidence to include",
    )
    args = parser.parse_args()

    if args.file:
        from pathlib import Path
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text as argument or use --file")

    conf_map = {"low": ClaimConfidence.LOW, "medium": ClaimConfidence.MEDIUM, "high": ClaimConfidence.HIGH}
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(text, min_confidence=conf_map[args.min_confidence])
    print(extractor.format_for_human_gate(claims))
