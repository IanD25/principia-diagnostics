"""
analysis/ — Phase 1 Diagnostic Toolkit

Two tools implemented:
  1. HypothesisGenerator — surfaces surprising high-similarity entity pairs
     and generates natural-language research prompts for each.
  2. CoverageAnalyzer    — measures property coverage, link network density,
     archetype distributions, and produces a markdown coverage report.

Usage
-----
    from analysis import HypothesisGenerator, CoverageAnalyzer

Both classes accept the paths exported from config.py (SOURCE_DB, HISTORY_DB)
so they integrate cleanly into the existing sync pipeline.
"""

from analysis.hypothesis_generator import HypothesisGenerator, SurprisingPair, EntityInfo
from analysis.coverage_analyzer import CoverageAnalyzer, CoverageReport, NetworkMetrics

__all__ = [
    "HypothesisGenerator",
    "SurprisingPair",
    "EntityInfo",
    "CoverageAnalyzer",
    "CoverageReport",
    "NetworkMetrics",
]
