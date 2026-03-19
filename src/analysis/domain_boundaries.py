"""
domain_boundaries.py — Domain boundary validation for cross-universe bridges.

Phase 4.2: Validates whether cross-domain bridge link types are appropriate
for the domain pair involved. Some link types (analogous_to, couples_to) freely
cross domains; others (derives_from, generalizes) should stay within the same
domain or adjacent domains.

Domain adjacency is defined by a curated matrix of domain pairs that share
sufficient formal structure to support strict logical relationships.

Entry points:
    validate_bridge_domain(...)    → DomainValidation  (single bridge)
    check_domain_boundaries(...)   → DomainBoundaryReport  (all bridges in RRP)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ── Domain crossing rules ────────────────────────────────────────────────────

class DomainStatus(Enum):
    VALID = "valid"
    WARNING = "warning"
    VIOLATION = "violation"


# Link types that freely cross domains (structural analogy, shared variables)
CROSS_DOMAIN_VALID: set[str] = {
    "analogous to",
    "couples to",
    "tests",
}

# Link types restricted to same-domain or adjacent-domain
SAME_DOMAIN_PREFERRED: set[str] = {
    "derives from",
    "generalizes",
    "implements",
    "constrains",
}

# Tension types — always valid cross-domain (detecting tension IS the point)
TENSION_TYPES: set[str] = {
    "tensions with",
    "predicts for",
}


# ── Domain adjacency matrix ─────────────────────────────────────────────────
# Each pair is bidirectional. Domains listed here share enough formal structure
# for "derives from" / "generalizes" links to be meaningful.

_ADJACENT_PAIRS: set[frozenset[str]] = {
    frozenset({"physics", "chemistry"}),
    frozenset({"physics", "mathematics"}),
    frozenset({"physics", "cosmology"}),
    frozenset({"physics", "information"}),
    frozenset({"physics", "geometry"}),
    frozenset({"chemistry", "biology"}),
    frozenset({"chemistry", "earth sciences"}),
    frozenset({"mathematics", "computer science"}),
    frozenset({"mathematics", "information"}),
    frozenset({"mathematics", "formal logic"}),
    frozenset({"mathematics", "geometry"}),
    frozenset({"formal logic", "computer science"}),
    frozenset({"formal logic", "information"}),
    frozenset({"computer science", "information"}),
    frozenset({"biology", "networks"}),
    frozenset({"physics", "networks"}),
    frozenset({"earth sciences", "physics"}),
    frozenset({"earth sciences", "biology"}),
    frozenset({"earth sciences", "astronomy"}),
}


def _extract_primary_domain(domain_str: str) -> str:
    """Extract primary domain from a multi-domain string like 'physics · chemistry'.
    Returns the first domain component."""
    if not domain_str:
        return "unknown"
    # Handle both ' · ' and ' ' separators
    parts = domain_str.replace(" · ", " ").replace(",", " ").split()
    return parts[0].lower().strip() if parts else "unknown"


def _extract_all_domains(domain_str: str) -> set[str]:
    """Extract all domains from a multi-domain string."""
    if not domain_str:
        return {"unknown"}
    parts = domain_str.replace(",", " · ").split(" · ")
    return {p.strip().lower() for p in parts if p.strip()}


def _domains_adjacent(domain_a: str, domain_b: str) -> bool:
    """Check if two domains are adjacent (including same domain)."""
    da = domain_a.lower().strip()
    db = domain_b.lower().strip()
    if da == db:
        return True
    return frozenset({da, db}) in _ADJACENT_PAIRS


def _any_domain_adjacent(domains_a: set[str], domains_b: set[str]) -> bool:
    """Check if any pair of domains between two sets is adjacent."""
    for da in domains_a:
        for db in domains_b:
            if _domains_adjacent(da, db):
                return True
    return False


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DomainValidation:
    """Result of validating a single bridge's domain crossing."""
    source_id: str
    target_id: str
    link_type: str
    source_domain: str
    target_domain: str
    status: DomainStatus
    reason: str

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "link_type": self.link_type,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass
class DomainBoundaryReport:
    """Summary of domain boundary validation for all bridges in an RRP."""
    rrp_db: str
    total_bridges: int
    valid_count: int = 0
    warning_count: int = 0
    violation_count: int = 0
    validations: list[DomainValidation] = field(default_factory=list)
    domain_coverage: list[str] = field(default_factory=list)

    @property
    def violation_rate(self) -> float:
        return self.violation_count / max(self.total_bridges, 1)

    @property
    def violations(self) -> list[DomainValidation]:
        return [v for v in self.validations if v.status == DomainStatus.VIOLATION]

    @property
    def warnings(self) -> list[DomainValidation]:
        return [v for v in self.validations if v.status == DomainStatus.WARNING]


# ── Core functions ───────────────────────────────────────────────────────────

def validate_bridge_domain(
    source_id: str,
    target_id: str,
    link_type: str,
    source_domain: str,
    target_domain: str,
) -> DomainValidation:
    """
    Validate whether a bridge's link type is appropriate for its domain pair.

    Rules:
    - CROSS_DOMAIN_VALID types: always valid across any domains
    - TENSION_TYPES: always valid (detecting tension is the point)
    - SAME_DOMAIN_PREFERRED types: valid if same domain or adjacent domains;
      warning if non-adjacent
    - Unknown link types: warning (unclassified)
    """
    source_domains = _extract_all_domains(source_domain)
    target_domains = _extract_all_domains(target_domain)

    # Tension types — always valid
    if link_type in TENSION_TYPES:
        return DomainValidation(
            source_id=source_id, target_id=target_id,
            link_type=link_type,
            source_domain=source_domain, target_domain=target_domain,
            status=DomainStatus.VALID,
            reason="tension type — cross-domain is expected",
        )

    # Cross-domain valid types
    if link_type in CROSS_DOMAIN_VALID:
        return DomainValidation(
            source_id=source_id, target_id=target_id,
            link_type=link_type,
            source_domain=source_domain, target_domain=target_domain,
            status=DomainStatus.VALID,
            reason=f"'{link_type}' freely crosses domain boundaries",
        )

    # Same-domain preferred types
    if link_type in SAME_DOMAIN_PREFERRED:
        if _any_domain_adjacent(source_domains, target_domains):
            return DomainValidation(
                source_id=source_id, target_id=target_id,
                link_type=link_type,
                source_domain=source_domain, target_domain=target_domain,
                status=DomainStatus.VALID,
                reason=f"'{link_type}' within same or adjacent domains",
            )
        else:
            return DomainValidation(
                source_id=source_id, target_id=target_id,
                link_type=link_type,
                source_domain=source_domain, target_domain=target_domain,
                status=DomainStatus.WARNING,
                reason=(
                    f"'{link_type}' crosses non-adjacent domains: "
                    f"{source_domain} → {target_domain}"
                ),
            )

    # Unknown link type
    return DomainValidation(
        source_id=source_id, target_id=target_id,
        link_type=link_type,
        source_domain=source_domain, target_domain=target_domain,
        status=DomainStatus.WARNING,
        reason=f"unclassified link type '{link_type}'",
    )


def check_domain_boundaries(
    rrp_db: str | Path,
    wiki_db: str | Path,
) -> DomainBoundaryReport:
    """
    Scan all cross-universe bridges in an RRP and validate domain boundaries.

    Loads domain info from both the RRP entries table and the DS Wiki entries
    table, then checks each bridge against the domain crossing rules.
    """
    rrp_db = Path(rrp_db)
    wiki_db = Path(wiki_db)

    # Load RRP entry domains
    rrp_conn = sqlite3.connect(rrp_db)
    rrp_cols = {row[1] for row in rrp_conn.execute("PRAGMA table_info(entries)")}
    if "domain" in rrp_cols:
        rrp_domains = {
            row[0]: row[1] or "unknown"
            for row in rrp_conn.execute("SELECT id, domain FROM entries")
        }
    else:
        rrp_domains = {}

    # Detect bridge column name
    bridge_cols = {row[1] for row in rrp_conn.execute("PRAGMA table_info(cross_universe_bridges)")}
    bridge_col = "rrp_entry_id" if "rrp_entry_id" in bridge_cols else "source_entry_id"

    # Load bridges with proposed link types
    bridges = rrp_conn.execute(
        f"SELECT {bridge_col}, ds_entry_id, proposed_link_type FROM cross_universe_bridges"
    ).fetchall()
    rrp_conn.close()

    # Load DS Wiki entry domains
    wiki_conn = sqlite3.connect(wiki_db)
    wiki_domains = {
        row[0]: row[1] or "unknown"
        for row in wiki_conn.execute("SELECT id, domain FROM entries")
    }
    wiki_conn.close()

    # Validate each bridge
    report = DomainBoundaryReport(
        rrp_db=str(rrp_db),
        total_bridges=len(bridges),
    )

    ds_domains_seen: set[str] = set()

    for rrp_id, ds_id, link_type in bridges:
        source_domain = rrp_domains.get(rrp_id, "unknown")
        target_domain = wiki_domains.get(ds_id, "unknown")
        link_type = link_type or "couples to"

        ds_domains_seen.update(_extract_all_domains(target_domain))

        validation = validate_bridge_domain(
            source_id=rrp_id,
            target_id=ds_id,
            link_type=link_type,
            source_domain=source_domain,
            target_domain=target_domain,
        )

        report.validations.append(validation)
        if validation.status == DomainStatus.VALID:
            report.valid_count += 1
        elif validation.status == DomainStatus.WARNING:
            report.warning_count += 1
        else:
            report.violation_count += 1

    report.domain_coverage = sorted(ds_domains_seen - {"unknown"})
    return report
