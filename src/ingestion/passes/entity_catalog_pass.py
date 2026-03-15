"""
entity_catalog_pass.py — Pass 1.5: extract law-like patterns from entity catalogs.

Entity catalogs (periodic table, protein databases, materials catalogs) produce weak
cross-universe bridges when run through Pass 2 directly, because individual entities
connect to DS Wiki laws via *usage*, not structural analogy.

This pass extracts patterns ACROSS entities — trends, anomalies, family behaviours —
and inserts them as synthetic "derived_pattern" entries. These law-like pattern entries
are then embedded and queried in a Pass 2b re-run, surfacing connections that the
raw entity-level comparison misses.

Synthetic entry conventions:
  source_type      = "derived_pattern"
  entry_type       = "reference_law"   (law-like statements, embeds well)
  WIC section name = "What It Claims"  (matches EMBED_SECTIONS in cross_universe_query)
  confidence       = "Tier 2"          (derived, not directly observed)
  authoring_status = "pattern_extracted"

Pattern types generated:
  group_trend      — trends within a periodic group (vertical column)
  period_trend     — trends across a period (horizontal row)
  block_char       — block-level orbital character (s/p/d/f)
  category_char    — element family characterisation (alkali metals, noble gases, etc.)
  anomaly          — statistically anomalous (element, property) pairs (|z| > 2.5)
  notable_anomaly  — hardcoded entries for 5 scientifically famous anomalies
                     (gold, mercury, helium, carbon, hydrogen)

Usage:
    from ingestion.passes.entity_catalog_pass import EntityCatalogPass
    stats = EntityCatalogPass("data/rrp/periodic_table/rrp_periodic_table.db").run()
"""

import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v: Optional[str]) -> Optional[float]:
    """Parse a string property value to float, return None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _slugify(s: str) -> str:
    """Convert a string to a safe identifier fragment."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _trend_direction(slope: float, values: list[float]) -> str:
    """Return human-readable trend description based on slope."""
    span = max(values) - min(values)
    if abs(slope) < 0.001 or span < 0.05:
        return "remains roughly constant"
    elif slope > 0:
        return "increases steadily"
    else:
        return "decreases steadily"


def _ordinal(n: int) -> str:
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    return f"{n}{suffixes.get(n % 10 if n % 100 not in (11, 12, 13) else 0, 'th')}"


# ── Notable anomaly overrides (hardcoded rich prose) ─────────────────────────
# These 5 entries replace or supplement the algorithmic anomaly detection for
# scientifically famous cases that require domain-specific mechanism prose.

NOTABLE_ANOMALIES = {
    "Au": {
        "id": "pat_anomaly_Au_relativistic",
        "title": "Gold: Relativistic Contraction and Anomalous Properties",
        "wic": (
            "Gold (Au, Z=79) exhibits anomalous properties relative to its Group 11 "
            "neighbours copper and silver: it has an unusually high density "
            "(19.30 g/cm³ vs silver's 10.49 g/cm³), absorbs blue light producing its "
            "characteristic yellow colour, and forms noble-metal-like bonds far stronger "
            "than group trends predict. These anomalies arise from relativistic contraction: "
            "at Z=79, 6s electrons travel at ~58% of the speed of light, causing their "
            "orbitals to contract and stabilise. The contracted 6s shell raises the energy "
            "gap to visible-light frequencies and dramatically increases the effective "
            "nuclear charge experienced by valence electrons. This connects gold to "
            "special-relativistic mechanics (Lorentz contraction applied to atomic orbitals), "
            "quantum electrodynamics corrections, and scaling laws describing how relativistic "
            "effects grow as Z². Gold is the canonical test case for relativistic quantum "
            "chemistry and a bridge between condensed matter and high-energy physics."
        ),
        "notes": (
            "Relativistic effect magnitude scales roughly as (Z/137)²: negligible for "
            "light elements, significant at Z>50, dominant at Z>79. The same effect "
            "explains mercury's anomalously low melting point (liquid at STP) and "
            "lead's divergence from tin in Group 14."
        ),
        "domain": "chemistry · physics",
    },
    "Hg": {
        "id": "pat_anomaly_Hg_liquid_metal",
        "title": "Mercury: Relativistic Stabilisation and Anomalous Liquid State",
        "wic": (
            "Mercury (Hg, Z=80) is the only metallic element that is liquid at standard "
            "temperature and pressure (melting point 234.32 K, −38.83°C), an anomaly "
            "unique among metals. The Group 12 trend predicts a melting point near "
            "500 K based on zinc (692.7 K) and cadmium (594.2 K). The actual value "
            "(234.32 K) deviates by more than 3σ from the group mean. "
            "The mechanism is relativistic contraction of the 6s orbital: the filled "
            "6s² shell is so strongly contracted and stabilised that mercury behaves "
            "more like a noble gas than a metal in terms of interatomic bonding. "
            "Weak metallic bonding → low cohesive energy → liquid at room temperature. "
            "This is a direct experimental consequence of special relativity applied at "
            "atomic scales and connects to Landauer's principle (information about "
            "bonding state), thermodynamic melting laws, and quantum confinement effects."
        ),
        "notes": (
            "Mercury also has anomalously high vapour pressure for a metal "
            "(1.3×10⁻³ mmHg at 20°C), consistent with the weak bonding argument. "
            "This relativistic stabilisation is the same effect as gold's colour, "
            "differing only in which physical property it most dramatically affects."
        ),
        "domain": "chemistry · physics",
    },
    "He": {
        "id": "pat_anomaly_He_quantum_liquid",
        "title": "Helium: Zero-Point Energy and Anomalously Low Boiling Point",
        "wic": (
            "Helium (He, Z=2) has the lowest boiling point of any element (4.222 K, "
            "−268.928°C), and uniquely does not solidify at atmospheric pressure "
            "regardless of temperature — it remains liquid down to absolute zero unless "
            "pressurised above 25 atmospheres. Group 18 noble gases show a clear "
            "increasing trend in boiling points with atomic mass (Ne 27.1 K, Ar 87.3 K, "
            "Kr 119.9 K, Xe 165.1 K). Helium's 4.22 K is an outlier below all extrapolations. "
            "Two quantum effects govern this: (1) London dispersion forces scale with "
            "polarisability, which is minimal for helium's compact electron cloud; "
            "(2) quantum zero-point energy for helium's light nucleus is large enough "
            "to prevent the crystal lattice from forming — the ground state kinetic "
            "energy exceeds the lattice binding energy. This connects helium to quantum "
            "statistical mechanics (Bose-Einstein statistics for ⁴He), superfluidity "
            "phenomena, and the quantum harmonic oscillator's ground-state energy "
            "floor — a direct macroscopic manifestation of the uncertainty principle."
        ),
        "notes": (
            "⁴He becomes a superfluid below 2.17 K (lambda point), a state with zero "
            "viscosity and infinite thermal conductivity. This is a macroscopic quantum "
            "coherence phenomenon governed by Bose-Einstein condensation, connecting "
            "helium to critical phenomena, phase transitions, and quantum field theory."
        ),
        "domain": "chemistry · physics",
    },
    "C": {
        "id": "pat_anomaly_C_tetravalence",
        "title": "Carbon: Anomalous Tetravalence and Organic Chemistry Foundation",
        "wic": (
            "Carbon (C, Z=6) is anomalous within Group 14 in its extraordinary ability "
            "to form stable chains, rings, and three-dimensional networks with itself "
            "and other elements — the basis of all organic chemistry. While silicon "
            "also forms four bonds, silicon-silicon chains are unstable above a few "
            "atoms in air; carbon chains are stable to thousands of atoms. "
            "Carbon's anomalous chemical versatility arises from the near-degeneracy "
            "of its 2s and 2p orbitals, enabling sp, sp², and sp³ hybridisation at "
            "essentially zero energy cost. The 2s-2p gap in carbon (4.18 eV) is "
            "smaller than in nitrogen (5.78 eV) or oxygen (8.70 eV), permitting "
            "facile hybrid orbital formation. Combined with intermediate electronegativity "
            "(2.55 Pauling), carbon forms stable bonds with H, O, N, S, and itself. "
            "This connects to graph theory (molecular topology as graphs), Shannon "
            "information theory (molecular diversity as combinatorial explosion), "
            "power-law scaling (number of possible organic molecules grows exponentially "
            "with carbon count), and the free energy landscape of protein folding."
        ),
        "notes": (
            "The number of possible organic molecules with molecular formula CₙHₘOₚ "
            "grows super-exponentially with n. Carbon's role as the backbone of "
            "biochemistry is not chemically inevitable — it is an anomaly of "
            "second-row orbital energetics that has no clear periodic analogue."
        ),
        "domain": "chemistry · biology",
    },
    "H": {
        "id": "pat_anomaly_H_dual_nature",
        "title": "Hydrogen: Dual Group Membership and Unique Chemical Duality",
        "wic": (
            "Hydrogen (H, Z=1) is anomalous in occupying Group 1 by convention while "
            "sharing properties with both Group 1 (alkali metals) and Group 17 (halogens). "
            "Like alkali metals, hydrogen has one valence electron and commonly forms H⁺ "
            "(proton). Like halogens, hydrogen can gain one electron to form H⁻ (hydride) "
            "and exists as a diatomic molecule H₂. Quantitatively: hydrogen's "
            "electronegativity (2.20 Pauling) is far above alkali metals (Li=0.98, "
            "Na=0.93) but below fluorine (3.98). Its first ionisation energy (1312 "
            "kJ/mol) is dramatically higher than lithium (520 kJ/mol) and sodium "
            "(496 kJ/mol) but comparable to chlorine (1251 kJ/mol). "
            "This dual nature makes hydrogen a special case in the periodic law: it "
            "satisfies the 1-electron rule for Group 1 placement but violates nearly "
            "every chemical trend of that group. It connects to amphoteric behaviour "
            "in acid-base theory, the proton's role in thermodynamic pH equilibria, "
            "hydrogen bonding as a cross-domain interaction (chemistry, biology, "
            "materials science), and the quantum mechanics of the simplest atom — "
            "the only element with an analytically solvable Schrödinger equation."
        ),
        "notes": (
            "The hydrogen atom is the exact solution of the Schrödinger equation "
            "with a Coulomb potential. All of quantum chemistry's approximation "
            "methods (variational, perturbational, DFT) are ultimately calibrated "
            "against the hydrogen atom's analytically known eigenvalues."
        ),
        "domain": "chemistry · physics",
    },
}

# Properties the anomaly detector runs over (all numeric)
ANOMALY_PROPERTIES = [
    "electronegativity_pauling",
    "density_g_cm3",
    "melting_point_K",
    "boiling_point_K",
    "first_ionization_energy_kJ_mol",
    "electron_affinity_kJ_mol",
    "molar_heat_J_mol_K",
]

# Block mechanism prose (partially hardcoded — orbital physics is universal)
BLOCK_PROSE = {
    "s": {
        "wic": (
            "The s-block comprises Groups 1 and 2 plus helium: {n} elements whose "
            "distinguishing valence electrons occupy s orbitals. Chemically, s-block "
            "elements are characterised by low ionisation energies (IE₁ range: "
            "{ie_min:.0f}–{ie_max:.0f} kJ/mol), low electronegativities ({en_min:.2f}–{en_max:.2f} "
            "Pauling), and high reactivity toward water and oxygen. Alkali metals "
            "(Group 1) donate their single valence electron readily; alkaline earth "
            "metals (Group 2) donate two. The s orbital's spherical symmetry and "
            "penetrating character near the nucleus means s electrons experience "
            "the least shielding of any valence type — hence the very high reactivity. "
            "Across the s-block, density ({dens_min:.3f}–{dens_max:.3f} g/cm³), melting point, "
            "and ionisation energy all increase from Group 1 to Group 2, "
            "while atomic radius and metallic character decrease. "
            "Helium is anomalous in the s-block: its filled 1s² shell gives it "
            "noble-gas inertness, placing it behaviourally in Group 18."
        ),
        "notes": (
            "The s→p energy gap governs the transition from reactive metals (s-block) "
            "to more electronegative elements (p-block). The width of this gap "
            "determines the electronegativity jump at the block boundary."
        ),
    },
    "p": {
        "wic": (
            "The p-block spans Groups 13–18: {n} elements whose outermost electrons "
            "occupy p orbitals, filling from one to six p electrons across each period. "
            "The p-block contains the greatest chemical diversity of any block: metals "
            "(aluminium, tin, lead), metalloids (silicon, germanium, arsenic), nonmetals "
            "(carbon, nitrogen, oxygen), halogens (fluorine, chlorine), and noble gases. "
            "Electronegativity ranges widely ({en_min:.2f}–{en_max:.2f} Pauling), spanning from "
            "near-metallic aluminium (1.61) to the most electronegative element fluorine (3.98). "
            "First ionisation energy ({ie_min:.0f}–{ie_max:.0f} kJ/mol) increases across each "
            "period as nuclear charge rises with constant shielding. "
            "The three p-block sub-groups — metals, metalloids, nonmetals — are separated "
            "by the 'staircase' diagonal that tracks the metal-nonmetal transition. "
            "This transition correlates with band-gap energy in solid-state physics: "
            "metals have zero gap, metalloids have semiconducting gaps, nonmetals have "
            "insulating gaps. The p-block is therefore the primary source of "
            "semiconductor materials (Si, Ge, As, Sb) and the basis of solid-state electronics."
        ),
        "notes": (
            "The 2p elements (B through Ne) are anomalous relative to their heavier "
            "p-block congeners: smaller atomic radii and no d orbitals for hypervalence. "
            "This explains why carbon chains are stable but silicon chains are not, "
            "and why nitrogen is N₂ (triple bond) while phosphorus is P₄ (single bonds)."
        ),
    },
    "d": {
        "wic": (
            "The d-block — the transition metals — spans Groups 3–12: {n} elements "
            "filling d orbitals from d¹ to d¹⁰. They share high melting points "
            "(mean: {mp_mean:.0f} K, range {mp_min:.0f}–{mp_max:.0f} K for those with data), "
            "high densities ({dens_mean:.2f} g/cm³ mean), and the ability to form multiple "
            "oxidation states due to the energetic proximity of the (n-1)d and ns orbitals. "
            "This multi-valence character enables transition metals to act as catalysts "
            "(variable oxidation state = variable electron donor/acceptor), form "
            "coordination complexes (d orbitals interact with ligand electron pairs), "
            "and exhibit magnetic behaviour (unpaired d electrons → paramagnetism or "
            "ferromagnetism). The Wiedemann-Franz law (thermal/electrical conductivity "
            "ratio) holds particularly well for d-block metals, connecting their "
            "electronic structure to the free-electron model of condensed matter. "
            "The d-block contains most industrially important metals: iron, copper, "
            "nickel, chromium, titanium, and the platinum-group metals."
        ),
        "notes": (
            "Chromium (d⁵s¹) and copper (d¹⁰s¹) are anomalous within the d-block: "
            "their actual configurations deviate from the Aufbau filling rule because "
            "the half-filled and fully-filled d shells have extra exchange-energy stability. "
            "This exchange-energy correction is a quantum mechanical effect not captured "
            "by the simple (n-1)d < ns orbital energy argument."
        ),
    },
    "f": {
        "wic": (
            "The f-block — lanthanides and actinides — contains {n} elements filling "
            "f orbitals. Lanthanides (Z=57–71) share nearly identical ionic radii and "
            "chemical behaviour due to the 'lanthanide contraction': as the 4f subshell "
            "fills, nuclear charge increases while 4f electrons provide poor shielding, "
            "so the effective nuclear charge increases across the series, steadily "
            "shrinking atomic radii from lanthanum (187 pm) to lutetium (175 pm). "
            "This contraction makes fifth-period and sixth-period d-block elements "
            "(e.g., Zr vs Hf, Nb vs Ta) nearly indistinguishable in ionic radius — "
            "a major challenge for their separation. "
            "Actinides (Z=89–103) are all radioactive. The lighter actinides (Th–Pu) "
            "show d-block-like multi-valence behaviour because 5f and 6d orbitals are "
            "close in energy; heavier actinides (Am–Lr) behave more lanthanide-like "
            "as 5f stabilises. The f-block connects nuclear physics (radioactive decay), "
            "quantum chemistry (f-orbital angular momentum, spin-orbit coupling), "
            "and materials science (nuclear fuel, magnets, phosphors)."
        ),
        "notes": (
            "The lanthanide contraction is the key cause of the otherwise anomalous "
            "similarity between 4d and 5d transition metal radii — without it, fifth "
            "and sixth period homologues would differ as much as third and fourth period "
            "pairs (e.g., Ti vs Zr differ by ~20 pm; Zr vs Hf differ by only ~3 pm)."
        ),
    },
}


# ── Main class ────────────────────────────────────────────────────────────────

class EntityCatalogPass:
    """
    Pass 1.5: extract and insert law-like pattern entries from an entity catalog bundle.

    Reads existing entries and their numeric entry_properties, generates five classes
    of synthetic pattern entries (group_trend, period_trend, block_char, category_char,
    anomaly/notable_anomaly), and inserts them back into the same bundle DB.

    All inserts use INSERT OR IGNORE — safe to re-run.
    Synthetic entries are identified by source_type='derived_pattern'.
    """

    def __init__(self, bundle_db: str | Path):
        self.bundle_db = Path(bundle_db)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run the full pattern extraction pipeline.
        Returns stats dict with counts per pattern type.
        """
        conn = sqlite3.connect(str(self.bundle_db))
        conn.row_factory = sqlite3.Row

        # Remove any previously extracted patterns (idempotent re-run)
        conn.execute("DELETE FROM entries WHERE source_type = 'derived_pattern'")
        conn.execute(
            "DELETE FROM sections WHERE entry_id LIKE 'pat_%'"
        )
        conn.execute(
            "DELETE FROM entry_properties WHERE entry_id LIKE 'pat_%'"
        )
        conn.commit()

        elements = self._load_element_data(conn)
        print(f"  Loaded {len(elements)} entity entries")

        stats = {
            "group_trends":    0,
            "period_trends":   0,
            "block_entries":   0,
            "category_entries":0,
            "anomaly_entries": 0,
            "notable_anomaly_entries": 0,
            "total_synthetic": 0,
        }

        group_trends    = self._extract_group_trends(elements)
        period_trends   = self._extract_period_trends(elements)
        block_chars     = self._extract_block_characterizations(elements)
        category_chars  = self._extract_category_characterizations(elements)
        anomalies       = self._extract_statistical_anomalies(elements)
        notable         = self._extract_notable_anomalies(elements)

        for entry in group_trends:
            self._insert_synthetic_entry(conn, entry)
            stats["group_trends"] += 1

        for entry in period_trends:
            self._insert_synthetic_entry(conn, entry)
            stats["period_trends"] += 1

        for entry in block_chars:
            self._insert_synthetic_entry(conn, entry)
            stats["block_entries"] += 1

        for entry in category_chars:
            self._insert_synthetic_entry(conn, entry)
            stats["category_entries"] += 1

        for entry in anomalies:
            self._insert_synthetic_entry(conn, entry)
            stats["anomaly_entries"] += 1

        for entry in notable:
            self._insert_synthetic_entry(conn, entry)
            stats["notable_anomaly_entries"] += 1

        conn.commit()
        conn.close()

        stats["total_synthetic"] = sum(
            v for k, v in stats.items() if k != "total_synthetic"
        )
        return stats

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_element_data(self, conn: sqlite3.Connection) -> list[dict]:
        """
        Load all non-derived entries + their entry_properties into a flat list of dicts.
        Each dict has keys: id, title, source_type, + property_name→value pairs.
        Numeric properties are pre-parsed to float (None if missing/unparseable).
        """
        entries = conn.execute(
            "SELECT id, title, source_type FROM entries WHERE source_type != 'derived_pattern'"
        ).fetchall()

        props_all = conn.execute(
            "SELECT entry_id, property_name, property_value FROM entry_properties"
        ).fetchall()

        # Build property map: entry_id → {prop_name: raw_value}
        prop_map: dict[str, dict] = defaultdict(dict)
        for row in props_all:
            prop_map[row["entry_id"]][row["property_name"]] = row["property_value"]

        elements = []
        for e in entries:
            props = prop_map.get(e["id"], {})
            el = {
                "id":          e["id"],
                "title":       e["title"],
                "source_type": e["source_type"],
                # Grouping keys (int where possible)
                "period": int(props["period"]) if "period" in props else None,
                "group":  int(props["group"])  if "group"  in props else None,
                "block":  props.get("block"),
                "category": props.get("category", ""),
                "symbol": props.get("symbol", e["id"].replace("elem_", "")),
            }
            # Parse all numeric properties
            for p in ANOMALY_PROPERTIES + ["atomic_mass_u", "atomic_number"]:
                el[p] = _safe_float(props.get(p))
            elements.append(el)

        return elements

    # ── Pattern generator 1: Group trends ────────────────────────────────────

    def _extract_group_trends(self, elements: list[dict]) -> list[dict]:
        """One synthetic entry per periodic group with sufficient EN or IE data."""
        entries = []
        by_group: dict[int, list[dict]] = defaultdict(list)
        for el in elements:
            if el["group"] is not None:
                by_group[el["group"]].append(el)

        # Sort each group by period (atomic number within group)
        for g in by_group:
            by_group[g].sort(key=lambda e: (e["period"] or 99, e["id"]))

        for group_num in sorted(by_group.keys()):
            members = by_group[group_num]

            # Collect EN and IE values (filter None)
            en_pairs  = [(m["period"], m["electronegativity_pauling"])
                         for m in members if m["electronegativity_pauling"] is not None]
            ie_pairs  = [(m["period"], m["first_ionization_energy_kJ_mol"])
                         for m in members if m["first_ionization_energy_kJ_mol"] is not None]

            # Need at least 3 data points for a meaningful trend
            use_en = len(en_pairs) >= 3
            use_ie = len(ie_pairs) >= 3
            if not (use_en or use_ie):
                continue

            min_period = min(m["period"] for m in members if m["period"])
            max_period = max(m["period"] for m in members if m["period"])
            titles     = [m["title"] for m in members]

            # Compute EN trend
            en_desc = ""
            if use_en:
                periods_en, vals_en = zip(*en_pairs)
                slope_en = float(np.polyfit(periods_en, vals_en, 1)[0])
                en_min, en_max = min(vals_en), max(vals_en)
                en_desc = (
                    f"Electronegativity {_trend_direction(slope_en, list(vals_en))} "
                    f"from {en_min:.2f} to {en_max:.2f} (Pauling scale, range {en_max - en_min:.2f}). "
                )

            # Compute IE trend
            ie_desc = ""
            if use_ie:
                periods_ie, vals_ie = zip(*ie_pairs)
                slope_ie = float(np.polyfit(periods_ie, vals_ie, 1)[0])
                ie_min, ie_max = min(vals_ie), max(vals_ie)
                ie_desc = (
                    f"First ionisation energy {_trend_direction(slope_ie, list(vals_ie))} "
                    f"from {ie_min:.0f} to {ie_max:.0f} kJ/mol (range {ie_max - ie_min:.0f}). "
                )

            members_str = ", ".join(titles[:6])
            if len(titles) > 6:
                members_str += f" and {len(titles) - 6} others"

            wic = (
                f"Group {group_num} contains {len(members)} elements spanning "
                f"Periods {min_period} through {max_period}: {members_str}. "
                f"All share the same number of valence electrons ({group_num if group_num <= 18 else '?'}), "
                f"producing analogous chemical behaviour across periods despite "
                f"very different atomic masses and nuclear charges. "
                f"{en_desc}{ie_desc}"
                f"The trend within the group is driven by increasing nuclear shielding "
                f"as inner electron shells accumulate with each successive period, "
                f"reducing the effective nuclear charge experienced by valence electrons "
                f"and lowering ionisation energy and electronegativity down the group."
            )

            entries.append({
                "id":     f"pat_group_{group_num}_trend",
                "title":  f"Group {group_num} Element Trends ({len(members)} elements, Periods {min_period}–{max_period})",
                "wic":    wic,
                "notes":  (
                    f"Group {group_num} valence configuration: "
                    f"{'ns¹' if group_num == 1 else 'ns²' if group_num == 2 else f'(n-1)d{group_num - 2}ns² / similar'}"
                    f". The group trend is the clearest demonstration of the periodic law: "
                    f"periodic recurrence of chemical properties at each new period."
                ),
                "domain": "chemistry",
            })

        return entries

    # ── Pattern generator 2: Period trends ───────────────────────────────────

    def _extract_period_trends(self, elements: list[dict]) -> list[dict]:
        """One synthetic entry per period describing trends across the row."""
        entries = []
        by_period: dict[int, list[dict]] = defaultdict(list)
        for el in elements:
            if el["period"] is not None:
                by_period[el["period"]].append(el)

        for period_num in sorted(by_period.keys()):
            members = sorted(by_period[period_num],
                             key=lambda e: e.get("atomic_number") or 999)
            if len(members) < 2:
                continue

            # EN trend across period
            en_pairs = [(m.get("atomic_number"), m["electronegativity_pauling"])
                        for m in members
                        if m.get("atomic_number") and m["electronegativity_pauling"] is not None]

            # IE trend across period
            ie_pairs = [(m.get("atomic_number"), m["first_ionization_energy_kJ_mol"])
                        for m in members
                        if m.get("atomic_number") and m["first_ionization_energy_kJ_mol"] is not None]

            if len(en_pairs) < 2 and len(ie_pairs) < 2:
                continue

            first_el = members[0]["title"]
            last_el  = members[-1]["title"]
            n        = len(members)

            en_desc = ""
            if len(en_pairs) >= 2:
                zs, vals = zip(*en_pairs)
                slope = float(np.polyfit(zs, vals, 1)[0])
                en_min, en_max = min(vals), max(vals)
                en_desc = (
                    f"Electronegativity {_trend_direction(slope, list(vals))} from "
                    f"{en_min:.2f} ({members[0]['title']}) to "
                    f"{en_max:.2f} ({members[-1]['title']}), "
                    f"a {en_max - en_min:.2f}-unit span. "
                )

            ie_desc = ""
            if len(ie_pairs) >= 2:
                zs, vals = zip(*ie_pairs)
                slope = float(np.polyfit(zs, vals, 1)[0])
                ie_min, ie_max = min(vals), max(vals)
                ie_desc = (
                    f"First ionisation energy {_trend_direction(slope, list(vals))} from "
                    f"{ie_min:.0f} to {ie_max:.0f} kJ/mol across the period. "
                )

            wic = (
                f"Period {period_num} contains {n} elements from {first_el} to {last_el}. "
                f"Across this period, nuclear charge increases by {n - 1} protons while "
                f"electrons are added to the same principal quantum shell (n={period_num}), "
                f"holding shielding roughly constant. "
                f"{en_desc}{ie_desc}"
                f"Atomic radius decreases across the period as the increasing nuclear charge "
                f"draws electrons closer. Metallic character decreases from left "
                f"({first_el}, typically a metal) to right ({last_el}, noble gas or nonmetal). "
                f"These simultaneous trends — increasing EN and IE, decreasing atomic radius "
                f"and metallic character — are the operational signature of the periodic law "
                f"within a horizontal row: a direct consequence of Coulomb's law applied "
                f"to the increasing nuclear charge at constant principal quantum number."
            )

            entries.append({
                "id":     f"pat_period_{period_num}_trend",
                "title":  f"Periodic Trends Across Period {period_num} ({n} elements, {first_el}→{last_el})",
                "wic":    wic,
                "notes":  (
                    f"Period {period_num} spans {n} elements. "
                    f"The trend is driven by Coulomb attraction: "
                    f"Z_eff = Z − σ (screening constant), which increases across the period "
                    f"as shielding σ changes slowly while Z increases by 1 each step. "
                    f"This is a direct experimental test of the Schrödinger equation's "
                    f"prediction for multi-electron atoms: effective nuclear charge governs "
                    f"all bulk chemical properties."
                ),
                "domain": "chemistry",
            })

        return entries

    # ── Pattern generator 3: Block characterizations ─────────────────────────

    def _extract_block_characterizations(self, elements: list[dict]) -> list[dict]:
        """One entry per orbital block (s, p, d, f) summarising its character."""
        entries = []
        by_block: dict[str, list[dict]] = defaultdict(list)
        for el in elements:
            if el["block"]:
                by_block[el["block"]].append(el)

        for block_name, members in sorted(by_block.items()):
            if block_name not in BLOCK_PROSE:
                continue

            n = len(members)
            # Compute stats for template insertion
            en_vals  = [m["electronegativity_pauling"] for m in members
                        if m["electronegativity_pauling"] is not None]
            ie_vals  = [m["first_ionization_energy_kJ_mol"] for m in members
                        if m["first_ionization_energy_kJ_mol"] is not None]
            mp_vals  = [m["melting_point_K"] for m in members
                        if m["melting_point_K"] is not None]
            d_vals   = [m["density_g_cm3"] for m in members
                        if m["density_g_cm3"] is not None]

            fmt = dict(
                n       = n,
                en_min  = min(en_vals) if en_vals else 0,
                en_max  = max(en_vals) if en_vals else 0,
                ie_min  = min(ie_vals) if ie_vals else 0,
                ie_max  = max(ie_vals) if ie_vals else 0,
                mp_min  = min(mp_vals) if mp_vals else 0,
                mp_max  = max(mp_vals) if mp_vals else 0,
                mp_mean = float(np.mean(mp_vals)) if mp_vals else 0,
                dens_min  = min(d_vals) if d_vals else 0,
                dens_max  = max(d_vals) if d_vals else 0,
                dens_mean = float(np.mean(d_vals)) if d_vals else 0,
            )

            prose = BLOCK_PROSE[block_name]
            try:
                wic   = prose["wic"].format(**fmt)
                notes = prose["notes"]
            except KeyError:
                continue

            entries.append({
                "id":     f"pat_block_{block_name}_characterization",
                "title":  f"{block_name.upper()}-Block Elements: Orbital Character and Chemical Behaviour ({n} elements)",
                "wic":    wic,
                "notes":  notes,
                "domain": "chemistry · physics" if block_name == "f" else "chemistry",
            })

        return entries

    # ── Pattern generator 4: Category characterizations ──────────────────────

    def _extract_category_characterizations(self, elements: list[dict]) -> list[dict]:
        """One entry per element category with ≥ 3 members."""
        entries = []
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for el in elements:
            cat = el.get("category", "").strip()
            if cat and not cat.startswith("unknown"):
                by_cat[cat].append(el)

        for category, members in sorted(by_cat.items()):
            if len(members) < 3:
                continue

            n = len(members)
            en_vals = [m["electronegativity_pauling"] for m in members
                       if m["electronegativity_pauling"] is not None]
            ie_vals = [m["first_ionization_energy_kJ_mol"] for m in members
                       if m["first_ionization_energy_kJ_mol"] is not None]
            d_vals  = [m["density_g_cm3"] for m in members
                       if m["density_g_cm3"] is not None]
            mp_vals = [m["melting_point_K"] for m in members
                       if m["melting_point_K"] is not None]

            names = [m["title"] for m in sorted(members, key=lambda e: e.get("atomic_number") or 999)]
            names_str = ", ".join(names[:6])
            if len(names) > 6:
                names_str += f" … ({len(names) - 6} more)"

            # Phase distribution
            phases = [m.get("phase_at_stp", "") for m in members]
            phase_counts = defaultdict(int)
            for p in phases:
                if p:
                    phase_counts[p.lower()] += 1
            phase_str = "; ".join(
                f"{cnt} {ph}" for ph, cnt in sorted(phase_counts.items(), key=lambda x: -x[1])
            )

            stats_parts = []
            if en_vals:
                stats_parts.append(
                    f"electronegativity {min(en_vals):.2f}–{max(en_vals):.2f} (mean {np.mean(en_vals):.2f} Pauling)"
                )
            if ie_vals:
                stats_parts.append(
                    f"first ionisation energy {min(ie_vals):.0f}–{max(ie_vals):.0f} kJ/mol "
                    f"(mean {np.mean(ie_vals):.0f})"
                )
            if d_vals:
                stats_parts.append(
                    f"density {min(d_vals):.2f}–{max(d_vals):.2f} g/cm³ (mean {np.mean(d_vals):.2f})"
                )
            if mp_vals:
                stats_parts.append(
                    f"melting point {min(mp_vals):.0f}–{max(mp_vals):.0f} K (mean {np.mean(mp_vals):.0f})"
                )

            stats_str = "; ".join(stats_parts) if stats_parts else "quantitative data limited"

            wic = (
                f"The {category} family contains {n} elements: {names_str}. "
                f"Standard phases at STP: {phase_str}. "
                f"Quantitative property ranges: {stats_str}. "
                f"Members of this family share characteristic electron configuration patterns "
                f"that produce coherent chemical behaviour across periods. "
                f"The family-level regularity is a direct expression of the periodic law: "
                f"chemical properties recur with period because valence electron count recurs. "
                f"Deviation from family trends signals special structural effects "
                f"(relativistic, quantum, or hybridisation anomalies) worth investigating."
            )

            # Domain: physics-chemistry for noble gases and actinides
            domain = "chemistry"
            if "noble" in category or "actinide" in category:
                domain = "chemistry · physics"
            elif "lanthanide" in category:
                domain = "chemistry · physics"

            entries.append({
                "id":     f"pat_category_{_slugify(category)}_characterization",
                "title":  f"{category.title()} Family: Properties and Periodic Behaviour ({n} elements)",
                "wic":    wic,
                "notes":  (
                    f"The {category} family spans atomic numbers "
                    f"{min(m.get('atomic_number') or 0 for m in members):.0f}–"
                    f"{max(m.get('atomic_number') or 0 for m in members):.0f}. "
                    f"Family membership is defined by the outermost electron configuration "
                    f"and block position, not by physical similarity alone."
                ),
                "domain": domain,
            })

        return entries

    # ── Pattern generator 5: Statistical anomalies ───────────────────────────

    def _extract_statistical_anomalies(self, elements: list[dict]) -> list[dict]:
        """
        Detect (element, property) anomalies using z-score within each group.
        Groups with < 3 data points are skipped.
        Anomalies with |z| > 2.5 are flagged.
        If an element is anomalous across multiple properties, one combined entry is generated.
        Notable anomaly symbols (NOTABLE_ANOMALIES) are skipped (handled separately).
        """
        by_group: dict[int, list[dict]] = defaultdict(list)
        for el in elements:
            if el["group"] is not None:
                by_group[el["group"]].append(el)

        # Map: symbol → {property: (z_score, expected, actual)}
        anomaly_map: dict[str, dict] = defaultdict(dict)

        for group_num, members in by_group.items():
            for prop in ANOMALY_PROPERTIES:
                vals    = [(m, m[prop]) for m in members if m[prop] is not None]
                if len(vals) < 3:
                    continue
                arr     = np.array([v for _, v in vals])
                mean    = float(np.mean(arr))
                std     = float(np.std(arr))
                if std < 1e-9:
                    continue
                for el, val in vals:
                    z = (val - mean) / std
                    if abs(z) > 2.5:
                        sym = el.get("symbol", "")
                        anomaly_map[sym][prop] = {
                            "z":        round(z, 2),
                            "expected": round(mean, 4),
                            "actual":   round(val, 4),
                            "group":    group_num,
                            "title":    el["title"],
                        }

        entries = []
        # Skip notable anomalies — handled by _extract_notable_anomalies
        notable_syms = set(NOTABLE_ANOMALIES.keys())

        for sym, prop_anomalies in sorted(anomaly_map.items()):
            if sym in notable_syms:
                continue
            if not prop_anomalies:
                continue

            # Pick the most extreme anomaly for the title/summary
            worst_prop = max(prop_anomalies, key=lambda p: abs(prop_anomalies[p]["z"]))
            worst      = prop_anomalies[worst_prop]
            elem_title = worst["title"]
            group_num  = worst["group"]

            # Build description of all anomalous properties
            prop_lines = []
            for p, a in sorted(prop_anomalies.items(), key=lambda x: -abs(x[1]["z"])):
                direction = "above" if a["z"] > 0 else "below"
                prop_lines.append(
                    f"  {p.replace('_', ' ')}: {a['actual']} "
                    f"({abs(a['z']):.1f}σ {direction} Group {a['group']} mean of {a['expected']})"
                )
            prop_str = "\n".join(prop_lines)

            wic = (
                f"{elem_title} ({sym}) is a statistical outlier within its periodic group "
                f"across {len(prop_anomalies)} {'property' if len(prop_anomalies)==1 else 'properties'}:\n"
                f"{prop_str}\n"
                f"The {abs(worst['z']):.1f}σ deviation in {worst_prop.replace('_', ' ')} "
                f"({'above' if worst['z'] > 0 else 'below'} Group {group_num} mean) "
                f"indicates a structural or quantum-mechanical effect beyond the "
                f"standard periodic trend. Statistical anomalies at this magnitude "
                f"typically reflect filled or half-filled subshell stability, "
                f"relativistic orbital contraction, or anomalous bonding hybridisation."
            )

            entries.append({
                "id":     f"pat_anomaly_{sym}_statistical",
                "title":  f"{elem_title} ({sym}): Statistical Outlier in Group {group_num} ({len(prop_anomalies)} properties)",
                "wic":    wic,
                "notes":  (
                    f"Z-scores computed within Group {group_num} "
                    f"using mean ± std of all group members with available data. "
                    f"Threshold: |z| > 2.5."
                ),
                "domain": "chemistry · physics" if worst["group"] in (11, 12) else "chemistry",
            })

        return entries

    # ── Pattern generator 6: Notable anomaly overrides ───────────────────────

    def _extract_notable_anomalies(self, elements: list[dict]) -> list[dict]:
        """Insert the 5 hardcoded scientifically famous anomaly entries."""
        # Only insert if the corresponding element is actually in this bundle
        present_symbols = {el.get("symbol", "") for el in elements}
        entries = []
        for sym, data in NOTABLE_ANOMALIES.items():
            if sym in present_symbols:
                entries.append({
                    "id":     data["id"],
                    "title":  data["title"],
                    "wic":    data["wic"],
                    "notes":  data.get("notes", ""),
                    "domain": data.get("domain", "chemistry · physics"),
                })
        return entries

    # ── Entry insertion ───────────────────────────────────────────────────────

    def _insert_synthetic_entry(self, conn: sqlite3.Connection, entry: dict) -> None:
        eid    = entry["id"]
        title  = entry["title"]
        domain = entry.get("domain", "chemistry")
        wic    = entry.get("wic", "").strip()
        notes  = entry.get("notes", "").strip()

        conn.execute("""
            INSERT OR IGNORE INTO entries
              (id, title, entry_type, source_type, domain, status, confidence, authoring_status)
            VALUES (?, ?, 'reference_law', 'derived_pattern', ?, 'established', 'Tier 2',
                    'pattern_extracted')
        """, (eid, title, domain))

        order = 0
        if wic:
            conn.execute("""
                INSERT OR IGNORE INTO sections (entry_id, section_name, content, section_order)
                VALUES (?, 'What It Claims', ?, ?)
            """, (eid, wic, order))
            order += 1

        if notes:
            conn.execute("""
                INSERT OR IGNORE INTO sections (entry_id, section_name, content, section_order)
                VALUES (?, 'Notes', ?, ?)
            """, (eid, notes, order))

        # Entry property tagging
        conn.execute("""
            INSERT OR IGNORE INTO entry_properties (entry_id, property_name, property_value)
            VALUES (?, 'pattern_type', ?)
        """, (eid, eid.split("_")[1] if "_" in eid else "unknown"))


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_entity_catalog_pass(bundle_db: str | Path) -> dict:
    """Convenience wrapper used by scripts and CLI."""
    return EntityCatalogPass(bundle_db).run()
