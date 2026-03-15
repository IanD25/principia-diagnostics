"""
Tests for claim_extractor.py — Phase 3A claim extraction.
"""

import sys
from pathlib import Path

# Path bootstrap
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from analysis.claim_extractor import (
    Claim,
    ClaimConfidence,
    ClaimExtractor,
    PolarityHint,
    detect_polarity,
    extract_sro,
    is_claim_sentence,
    split_sentences,
)


# ── Polarity detection ──────────────────────────────────────────────────────

class TestDetectPolarity:
    def test_negative_contradicts(self):
        hint, markers = detect_polarity("This result contradicts the standard model")
        assert hint == PolarityHint.NEGATIVE
        assert any("contradicts" in m for m in markers)

    def test_negative_violates(self):
        hint, _ = detect_polarity("The measurement violates conservation of energy")
        assert hint == PolarityHint.NEGATIVE

    def test_negative_exceeds(self):
        hint, _ = detect_polarity("The neutrino velocity exceeds c by 6 sigma")
        assert hint == PolarityHint.NEGATIVE

    def test_negative_tension(self):
        hint, _ = detect_polarity("There is tension between these observations")
        assert hint == PolarityHint.NEGATIVE

    def test_negative_anomaly(self):
        hint, _ = detect_polarity("We observe an anomalous signal in the data")
        assert hint == PolarityHint.NEGATIVE

    def test_positive_confirms(self):
        hint, markers = detect_polarity("This confirms the theoretical prediction")
        assert hint == PolarityHint.POSITIVE
        assert any("confirms" in m for m in markers)

    def test_positive_consistent(self):
        hint, _ = detect_polarity("The result is consistent with general relativity")
        assert hint == PolarityHint.POSITIVE

    def test_positive_agreement(self):
        hint, _ = detect_polarity("The data is in good agreement with the model")
        assert hint == PolarityHint.POSITIVE

    def test_positive_validates(self):
        hint, _ = detect_polarity("This independently validates the prediction")
        assert hint == PolarityHint.POSITIVE

    def test_positive_recovers(self):
        hint, _ = detect_polarity("The model recovers the standard cosmological expansion")
        assert hint == PolarityHint.POSITIVE

    def test_neutral_no_markers(self):
        hint, markers = detect_polarity("The temperature was 300 Kelvin")
        assert hint == PolarityHint.NEUTRAL
        assert markers == []

    def test_mixed_defaults_negative(self):
        hint, markers = detect_polarity(
            "This confirms the model but contradicts earlier results"
        )
        assert hint == PolarityHint.NEGATIVE
        assert len(markers) >= 2


# ── Sentence splitting ──────────────────────────────────────────────────────

class TestSplitSentences:
    def test_basic_split(self):
        result = split_sentences("First sentence. Second sentence.")
        assert len(result) == 2

    def test_abbreviation_not_split(self):
        result = split_sentences("Dr. Smith found the result. It was significant.")
        assert len(result) == 2
        assert "Dr." in result[0]

    def test_single_sentence(self):
        result = split_sentences("Just one sentence here")
        assert len(result) == 1

    def test_empty(self):
        result = split_sentences("")
        assert result == [] or result == [""]

    def test_et_al_not_split(self):
        result = split_sentences("Farrah et al. measured k. The value was 3.")
        assert len(result) == 2


# ── Claim detection ─────────────────────────────────────────────────────────

class TestIsClaimSentence:
    def test_we_find(self):
        is_claim, conf = is_claim_sentence("We find that the temperature increases with pressure")
        assert is_claim is True

    def test_our_results(self):
        is_claim, conf = is_claim_sentence("Our results show a significant correlation at 3σ")
        assert is_claim is True
        assert conf == ClaimConfidence.HIGH  # two indicators

    def test_background_only(self):
        is_claim, _ = is_claim_sentence("It is well known that water boils at 100C")
        assert is_claim is False

    def test_neutral_statement(self):
        is_claim, _ = is_claim_sentence("The sample was prepared at room temperature")
        assert is_claim is False

    def test_statistical_significance(self):
        is_claim, _ = is_claim_sentence("The detection is significant at 5σ confidence level")
        assert is_claim is True


# ── SRO extraction ──────────────────────────────────────────────────────────

class TestExtractSRO:
    def test_consistent_with(self):
        sro = extract_sro("The measurement is consistent with the standard model prediction")
        assert sro is not None
        subj, rel, obj = sro
        assert "consistent with" in rel
        assert len(subj) > 0
        assert len(obj) > 0

    def test_contradicts(self):
        sro = extract_sro("Our data contradicts the null hypothesis")
        assert sro is not None
        _, rel, _ = sro
        assert "contradicts" in rel

    def test_we_find(self):
        sro = extract_sro("We find that the neutrino velocity is equal to c within uncertainties")
        assert sro is not None

    def test_no_match(self):
        sro = extract_sro("The temperature was 300 Kelvin")
        assert sro is None

    def test_implies(self):
        sro = extract_sro("This implies that dark energy dominates the expansion")
        assert sro is not None
        _, rel, _ = sro
        assert "implies" in rel


# ── ClaimExtractor ──────────────────────────────────────────────────────────

class TestClaimExtractor:
    def setup_method(self):
        self.extractor = ClaimExtractor()

    def test_extract_simple_claim(self):
        text = "We find that the BH mass is consistent with the vacuum energy model."
        claims = self.extractor.extract_claims(text, source_section="results")
        assert len(claims) >= 1
        assert claims[0].source_section == "results"
        assert claims[0].human_approved is False

    def test_extract_negative_claim(self):
        text = "We show that the neutrino velocity exceeds the speed of light by 6σ."
        claims = self.extractor.extract_claims(text)
        assert len(claims) >= 1
        assert claims[0].polarity_hint == PolarityHint.NEGATIVE

    def test_extract_positive_claim(self):
        text = "Our results confirm the cosmological coupling prediction of k = 3."
        claims = self.extractor.extract_claims(text)
        assert len(claims) >= 1
        assert claims[0].polarity_hint == PolarityHint.POSITIVE

    def test_no_claims_from_background(self):
        text = "It is well known that the universe is expanding."
        claims = self.extractor.extract_claims(text, min_confidence=ClaimConfidence.MEDIUM)
        assert len(claims) == 0

    def test_multiple_claims(self):
        text = (
            "We find that k = 3.11 at 90% confidence. "
            "Our results confirm the CCBH model. "
            "This implies that BHs are cosmologically coupled."
        )
        claims = self.extractor.extract_claims(text)
        assert len(claims) >= 2

    def test_human_gate_always_false(self):
        text = "We demonstrate that entropy increases monotonically."
        claims = self.extractor.extract_claims(text)
        for claim in claims:
            assert claim.human_approved is False

    def test_extract_from_sections(self):
        sections = {
            "abstract": "We find that the coupling constant k equals 3.",
            "results": "Our measurements confirm the theoretical prediction.",
        }
        claims = self.extractor.extract_from_sections(sections)
        sections_found = {c.source_section for c in claims}
        assert len(claims) >= 2
        assert "abstract" in sections_found or "results" in sections_found

    def test_format_for_human_gate(self):
        text = "We show that dark energy is consistent with vacuum energy."
        claims = self.extractor.extract_claims(text)
        md = self.extractor.format_for_human_gate(claims)
        assert "Human Review Required" in md
        assert "PENDING" in md

    def test_empty_text(self):
        claims = self.extractor.extract_claims("")
        assert claims == []

    def test_claim_as_dict(self):
        text = "We find that temperature is consistent with the model prediction."
        claims = self.extractor.extract_claims(text)
        if claims:
            d = claims[0].as_dict()
            assert "text" in d
            assert "polarity_hint" in d
            assert "human_approved" in d
            assert d["human_approved"] is False

    def test_claim_str(self):
        text = "We demonstrate that entropy supports the second law."
        claims = self.extractor.extract_claims(text)
        if claims:
            s = str(claims[0])
            assert "PENDING APPROVAL" in s


# ── OPERA test cases (SCF validation) ───────────────────────────────────────

class TestOPERAClaims:
    """Test claims from the OPERA experiment — the canonical Phase 3 test case."""

    def setup_method(self):
        self.extractor = ClaimExtractor()

    def test_original_ftl_claim(self):
        text = "We observe that the neutrino velocity exceeds the speed of light by 6.1σ."
        claims = self.extractor.extract_claims(text, source_section="results_2011")
        assert len(claims) >= 1
        assert claims[0].polarity_hint == PolarityHint.NEGATIVE

    def test_corrected_claim(self):
        text = "We find that the neutrino velocity is consistent with c within uncertainties."
        claims = self.extractor.extract_claims(text, source_section="results_2012")
        assert len(claims) >= 1
        assert claims[0].polarity_hint == PolarityHint.POSITIVE


# ── CCBH test cases ─────────────────────────────────────────────────────────

class TestCCBHClaims:
    """Test claims from the CCBH cluster papers."""

    def setup_method(self):
        self.extractor = ClaimExtractor()

    def test_farrah_k_measurement(self):
        text = "We measure the cosmological coupling constant k = 3.11 at 90% confidence level."
        claims = self.extractor.extract_claims(text, source_section="results")
        assert len(claims) >= 1

    def test_hubble_tension_reduction(self):
        text = "Our results show that the Hubble tension is reduced from 4.2σ to 2.7σ."
        claims = self.extractor.extract_claims(text, source_section="results")
        assert len(claims) >= 1

    def test_desi_recovery(self):
        text = "The CCBH model recovers the standard cosmological expansion history."
        claims = self.extractor.extract_claims(text)
        assert len(claims) >= 1
        assert claims[0].polarity_hint == PolarityHint.POSITIVE
