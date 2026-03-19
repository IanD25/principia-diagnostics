"""
structural_alignment.py — Link-type weighted bridge scoring (SPT replacement)

Instead of using LLM framing variants to detect logical polarity (SPT),
this module reads the explicit link types in the RRP graph and propagates
polarity along bridge paths.

Core insight: the sign of a scientific claim's relationship to formal principles
is encoded in the RRP's link_type field, not recoverable from BGE embeddings alone.

  signed_score(E, W) = polarity(link_type) × sim(E→W)

Two-hop paths:
  E --[link_type P]--> M --[bridge sim]--> DS Wiki W
  → signed_score = polarity(P) × sim(M, W)
  → meaning: E contests/aligns with W by virtue of its relationship to M

One-hop direct:
  E --[bridge sim]--> DS Wiki W, polarity from E's own outgoing links

Output:
  - Per-entry: net polarity, top CONTESTS / ALIGNS bridges
  - Paper-level: mean polarity, contested count, aligned count
  - DS Wiki-level: which formal entries the paper net-contests vs net-aligns with
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Link polarity taxonomy ────────────────────────────────────────────────────

LINK_POLARITY: dict[str, float] = {
    # Strong tension — entry contests the target
    "would_have_violated":            -1.0,
    "contradicted":                   -1.0,
    "explains_anomaly_in":            -0.5,   # acknowledges anomaly in target
    "introduced_undetected_error_in": -0.4,   # caused methodological error in target

    # Strong alignment — entry supports the target
    "is_consistent_with":             +1.0,
    "independently_validates":        +0.8,
    "supports":                       +0.7,
    "supersedes":                     +0.5,   # replaces, implies prior was flawed but corrected
    "bounds_uncertainty_of":          +0.3,

    # Neutral / methodological — no polarity signal (default 0.0)
    # produces, provides_*, sends_*, measured_by, etc.
}


def link_polarity(link_type: str) -> float:
    return LINK_POLARITY.get(link_type, 0.0)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BridgeAlignment:
    """Signed alignment of one RRP entry against one DS Wiki entry."""
    rrp_entry_id: str
    rrp_entry_title: str
    ds_entry_id: str
    ds_entry_title: str
    raw_sim: float           # cosine similarity (unsigned)
    polarity: float          # [-1, +1] derived from link graph
    signed_score: float      # raw_sim × polarity × formality_weight
    path_description: str    # human-readable derivation
    hop: int                 # 1 = direct bridge, 2 = via intermediate entry
    formality_tier: int = 2  # DS Wiki anchor's formality tier (Phase 4.3)

    @property
    def alignment_label(self) -> str:
        if self.polarity > 0.3:
            return "ALIGNS"
        if self.polarity < -0.3:
            return "CONTESTS"
        return "NEUTRAL"


@dataclass
class EntryAlignment:
    """All bridge alignments for one RRP entry."""
    entry_id: str
    entry_title: str
    entry_type: str
    bridges: list[BridgeAlignment] = field(default_factory=list)

    @property
    def net_polarity(self) -> float:
        polarized = [b for b in self.bridges if abs(b.polarity) > 0.1]
        if not polarized:
            return 0.0
        return sum(b.polarity for b in polarized) / len(polarized)

    @property
    def top_contested(self) -> list[BridgeAlignment]:
        return sorted(
            [b for b in self.bridges if b.polarity < -0.3],
            key=lambda b: b.signed_score
        )[:3]

    @property
    def top_aligned(self) -> list[BridgeAlignment]:
        return sorted(
            [b for b in self.bridges if b.polarity > 0.3],
            key=lambda b: -b.signed_score
        )[:3]


@dataclass
class StructuralAlignmentResult:
    rrp_db: str
    entries: list[EntryAlignment] = field(default_factory=list)

    @property
    def contested_entries(self) -> list[EntryAlignment]:
        return [e for e in self.entries if e.net_polarity < -0.3]

    @property
    def aligned_entries(self) -> list[EntryAlignment]:
        return [e for e in self.entries if e.net_polarity > 0.3]

    @property
    def mean_polarity(self) -> float:
        polarized = [e for e in self.entries if abs(e.net_polarity) > 0.1]
        if not polarized:
            return 0.0
        return sum(e.net_polarity for e in polarized) / len(polarized)

    def ds_wiki_summary(self) -> dict[str, float]:
        """Aggregate signed scores per DS Wiki entry across all RRP entries."""
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for ea in self.entries:
            for b in ea.bridges:
                if b.polarity == 0.0:
                    continue
                totals[b.ds_entry_id] = totals.get(b.ds_entry_id, 0.0) + b.signed_score
                counts[b.ds_entry_id] = counts.get(b.ds_entry_id, 0) + 1
        return {k: totals[k] / counts[k] for k in totals}


# ── Core analysis ─────────────────────────────────────────────────────────────

def _get_bridge_col(conn: sqlite3.Connection) -> str:
    """Detect whether bridges table uses rrp_entry_id or source_entry_id."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(cross_universe_bridges)")}
    if "rrp_entry_id" in cols:
        return "rrp_entry_id"
    return "source_entry_id"


def _load_formality_tiers_sa(wiki_db: Optional[str | Path] = None) -> dict[str, int]:
    """Load formality_tier for DS Wiki entries. Returns {entry_id: tier}."""
    if wiki_db is None:
        return {}
    conn = sqlite3.connect(wiki_db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(entries)")}
    if "formality_tier" not in cols:
        conn.close()
        return {}
    tiers = {
        row[0]: row[1]
        for row in conn.execute("SELECT id, formality_tier FROM entries")
        if row[1] is not None
    }
    conn.close()
    return tiers


def _formality_weight_sa(tier: int) -> float:
    """Formality weight for signed score scaling (Phase 4.3)."""
    return {1: 1.0, 2: 0.85, 3: 0.70}.get(tier, 0.85)


def run_structural_alignment(
    rrp_db: str | Path,
    wiki_db: Optional[str | Path] = None,
) -> StructuralAlignmentResult:
    """
    Compute link-type weighted bridge alignments for all entries in an RRP.

    Phase 4.3: If wiki_db is provided, formality_tier weights are applied to
    signed scores — bridges to higher-formality DS Wiki entries carry more weight.

    Requires:
      - entries table with (id, title, entry_type)
      - links table with (source_id, target_id, link_type)
      - cross_universe_bridges table with bridges already populated (run Pass 2 first)
    """
    rrp_db = Path(rrp_db)
    formality_tiers = _load_formality_tiers_sa(wiki_db)
    conn = sqlite3.connect(rrp_db)

    bridge_col = _get_bridge_col(conn)

    entries_meta: dict[str, tuple[str, str]] = {
        row[0]: (row[1], row[2])
        for row in conn.execute("SELECT id, title, entry_type FROM entries")
    }

    links = conn.execute(
        "SELECT source_id, target_id, link_type FROM links"
    ).fetchall()

    bridges_by_entry: dict[str, list[tuple[str, str, float]]] = {}
    for row in conn.execute(
        f"SELECT {bridge_col}, ds_entry_id, ds_entry_title, similarity "
        f"FROM cross_universe_bridges"
    ):
        bridges_by_entry.setdefault(row[0], []).append((row[1], row[2], float(row[3])))

    conn.close()

    # Build outgoing link index: source → [(target, link_type)]
    outgoing: dict[str, list[tuple[str, str]]] = {}
    for src, tgt, ltype in links:
        outgoing.setdefault(src, []).append((tgt, ltype))

    result = StructuralAlignmentResult(rrp_db=str(rrp_db))

    for entry_id, (title, entry_type) in entries_meta.items():
        ea = EntryAlignment(
            entry_id=entry_id,
            entry_title=title[:60],
            entry_type=entry_type,
        )

        own_links = outgoing.get(entry_id, [])
        own_polar_vals = [link_polarity(lt) for _, lt in own_links if link_polarity(lt) != 0.0]
        own_polarity = sum(own_polar_vals) / len(own_polar_vals) if own_polar_vals else 0.0
        own_link_desc = (
            " + ".join(f"{lt}({link_polarity(lt):+.1f})" for _, lt in own_links if link_polarity(lt) != 0.0)
            or "no polarity links"
        )

        # -- One-hop: direct bridges with E's own polarity --
        direct_bridges: dict[str, BridgeAlignment] = {}
        for ds_id, ds_title, sim in bridges_by_entry.get(entry_id, []):
            ds_tier = formality_tiers.get(ds_id, 2)
            fw = _formality_weight_sa(ds_tier)
            ba = BridgeAlignment(
                rrp_entry_id=entry_id,
                rrp_entry_title=title[:50],
                ds_entry_id=ds_id,
                ds_entry_title=ds_title[:40],
                raw_sim=sim,
                polarity=own_polarity,
                signed_score=sim * own_polarity * fw,
                path_description=f"direct [{own_link_desc}]",
                hop=1,
                formality_tier=ds_tier,
            )
            direct_bridges[ds_id] = ba
            ea.bridges.append(ba)

        # -- Two-hop: E --[polarity link]--> M --[bridge]--> DS Wiki W --
        for target_m, ltype in own_links:
            pol = link_polarity(ltype)
            if pol == 0.0:
                continue
            target_title = entries_meta.get(target_m, ("unknown", ""))[0][:40]
            for ds_id, ds_title, sim in bridges_by_entry.get(target_m, []):
                ds_tier = formality_tiers.get(ds_id, 2)
                fw = _formality_weight_sa(ds_tier)
                if ds_id in direct_bridges:
                    # Strengthen the existing direct bridge if this path has higher |polarity|
                    existing = direct_bridges[ds_id]
                    if abs(pol) > abs(existing.polarity):
                        existing.polarity = pol
                        existing.signed_score = existing.raw_sim * pol * fw
                        existing.path_description = (
                            f"via {target_m} [{ltype}({pol:+.1f})]"
                        )
                    continue

                ea.bridges.append(BridgeAlignment(
                    rrp_entry_id=entry_id,
                    rrp_entry_title=title[:50],
                    ds_entry_id=ds_id,
                    ds_entry_title=ds_title[:40],
                    raw_sim=sim,
                    polarity=pol,
                    signed_score=sim * pol * fw,
                    path_description=f"via {target_m} [{ltype}({pol:+.1f})]",
                    hop=2,
                    formality_tier=ds_tier,
                ))

        result.entries.append(ea)

    # Sort: most contested first
    result.entries.sort(key=lambda e: e.net_polarity)
    return result


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_sa_report(result: StructuralAlignmentResult) -> None:
    print(f"\n{'='*72}")
    print(f"STRUCTURAL ALIGNMENT REPORT")
    print(f"RRP: {Path(result.rrp_db).name}")
    print(f"mean_polarity = {result.mean_polarity:+.3f}  "
          f"| {len(result.contested_entries)} CONTESTED  "
          f"| {len(result.aligned_entries)} ALIGNED")
    print(f"{'='*72}\n")

    for ea in result.entries:
        if not ea.bridges:
            continue
        polarity_str = f"{ea.net_polarity:+.3f}"
        status = (
            "CONTESTED" if ea.net_polarity < -0.3 else
            "ALIGNED  " if ea.net_polarity > 0.3 else
            "NEUTRAL  "
        )
        print(f"[{status}] {ea.entry_id:<42} polarity={polarity_str}")
        notable = sorted(
            [b for b in ea.bridges if abs(b.polarity) > 0.1],
            key=lambda b: b.polarity
        )[:5]
        for b in notable:
            sign = "↕ CONTESTS" if b.polarity < -0.3 else "↑ ALIGNS  " if b.polarity > 0.3 else "→ neutral "
            print(f"    {sign}  {b.ds_entry_id:<10}  "
                  f"sim={b.raw_sim:.4f}  score={b.signed_score:+.4f}  "
                  f"[{b.path_description}]")

    # DS Wiki summary — which formal entries does the paper net contest vs align?
    ds_summary = result.ds_wiki_summary()
    if ds_summary:
        sorted_ds = sorted(ds_summary.items(), key=lambda x: x[1])
        print(f"\n{'─'*72}")
        print("DS WIKI ENTRIES — net signed score across paper (negative = contested):")
        contested_ds = [(k, v) for k, v in sorted_ds if v < -0.1]
        aligned_ds = [(k, v) for k, v in sorted_ds if v > 0.1]
        if contested_ds:
            print("  CONTESTED:")
            for ds_id, score in contested_ds:
                print(f"    {ds_id:<12}  net={score:+.4f}")
        if aligned_ds:
            print("  ALIGNED:")
            for ds_id, score in reversed(aligned_ds):
                print(f"    {ds_id:<12}  net={score:+.4f}")

    print(f"\n{'─'*72}")
    print(f"CONTESTED  (net < -0.3) : {len(result.contested_entries)}")
    print(f"NEUTRAL    (-0.3–+0.3)  : "
          f"{len([e for e in result.entries if -0.3 <= e.net_polarity <= 0.3])}")
    print(f"ALIGNED    (net > +0.3) : {len(result.aligned_entries)}")
    print()
    print("signed_score = raw_sim × link_polarity")
    print("polarity taxonomy: would_have_violated=-1.0, contradicted=-1.0,")
    print("  explains_anomaly=-0.5, is_consistent_with=+1.0,")
    print("  independently_validates=+0.8, supports=+0.7, supersedes=+0.5")
    print(f"{'='*72}\n")
