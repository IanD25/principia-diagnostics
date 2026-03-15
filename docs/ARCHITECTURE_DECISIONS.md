# Architecture Decisions

> One decision per section. Each records the context, options considered, decision made, and consequences.

---

## ADR-001: Hyperedge Representation for E. coli Core Parser

**Date:** 2026-03-11
**Status:** Decided — Option A (Reification)
**Blocks:** `src/ingestion/parsers/ecoli_core_parser.py`

---

### Context

The E. coli core metabolic model (COBRA JSON, `data/rrp/ecoli_core/raw/e_coli_core.json`) contains:

- **95 reactions**, **72 metabolites**, **137 genes**, 2 compartments
- Reactions are **hyperedges**: a single reaction connects multiple metabolites simultaneously with signed stoichiometric coefficients (negative = substrate, positive = product)
- Average 3.8 metabolites per reaction; max 23 (BIOMASS reaction)
- 47 of 95 reactions are reversible (lower_bound < 0)

Example — PFK (Phosphofructokinase):
```
atp_c(-1) + f6p_c(-1)  →  adp_c(+1) + fdp_c(+1) + h_c(+1)
```

This is a true hyperedge: one node (PFK) with 5 simultaneous participants. Binary links (source → target) cannot represent this natively.

The question: **how should hyperedges be stored in the RRP SQLite bundle schema?**

---

### Option A — Reification (chosen)

Treat each reaction as an **entry node** of type `reaction`. Connect it to participant metabolites via binary links with a `stoichiometry_coef` property.

**Schema changes to RRP bundle (generated artifact, not ds_wiki.db):**

```sql
-- No new tables. One new column on the existing links table:
ALTER TABLE links ADD COLUMN stoichiometry_coef REAL;

-- Link types used:
--   reaction --[consumes]--> metabolite   (coef stored as positive magnitude)
--   reaction --[produces]--> metabolite   (coef stored as positive magnitude)
--   reaction --[catalyzed_by]--> gene
--   gene     --[participates_in]--> reaction  (reverse, optional)
```

**Entry types added:**

| entry_type   | Example ID   | Source            |
|--------------|--------------|-------------------|
| `reaction`   | `rxn_PFK`    | reactions[*]      |
| `metabolite` | `met_atp_c`  | metabolites[*]    |
| `gene`       | `gene_b3916` | genes[*]          |

**Reversibility** stored as `entry_property`: `{ lower_bound, upper_bound, reversible, subsystem, ec_code }`.

**How PFK looks after reification:**

```
entries: rxn_PFK  (type=reaction, title="Phosphofructokinase")
entries: met_atp_c (type=metabolite)
entries: met_f6p_c (type=metabolite)
entries: met_adp_c (type=metabolite)
entries: met_fdp_c (type=metabolite)
entries: met_h_c   (type=metabolite)

links: rxn_PFK --[consumes]--> met_atp_c  (coef=1.0)
links: rxn_PFK --[consumes]--> met_f6p_c  (coef=1.0)
links: rxn_PFK --[produces]--> met_adp_c  (coef=1.0)
links: rxn_PFK --[produces]--> met_fdp_c  (coef=1.0)
links: rxn_PFK --[produces]--> met_h_c    (coef=1.0)
```

**Pros:**
- Zero changes to CrossUniverseQuery, Pass 2b, or ChromaDB pipeline — they only look at `entries` and `links`
- Reactions and metabolites both get embedded → bridges to DS Wiki entries work immediately
- BIOMASS reaction (23 participants) becomes 23 links — fully queryable ("find all reactions that consume ATP")
- `INSERT OR IGNORE` pattern unchanged throughout
- RRP bundle schema stays backward-compatible with zoo_classes and periodic_table parsers

**Cons:**
- The "reaction as a whole" is not a first-class hyperedge — it's approximated by the reaction entry + its links
- Stoichiometry directionality requires reading link_type (`consumes` vs `produces`) + coef
- Gene-reaction rules with boolean logic (`b3916 or b1723`) require a separate `entry_property` for `gene_reaction_rule` text; the OR/AND structure is not modeled as graph edges (acceptable for Phase 2)

---

### Option B — Native Hyperedge Tables (rejected)

Two new tables: `hyperedges` (one row per reaction) and `hyperedge_members` (one row per metabolite participant).

```sql
CREATE TABLE hyperedges (
    id           TEXT PRIMARY KEY,
    edge_type    TEXT,   -- 'reaction'
    name         TEXT,
    subsystem    TEXT,
    lower_bound  REAL,
    upper_bound  REAL
);
CREATE TABLE hyperedge_members (
    hyperedge_id      TEXT REFERENCES hyperedges(id),
    member_id         TEXT REFERENCES entries(id),
    role              TEXT,   -- 'substrate', 'product', 'catalyst'
    stoichiometry_coef REAL
);
```

**Why rejected:**
- CrossUniverseQuery, Pass 2b, and ChromaDB bridge computation all operate on `entries` + binary `links` — would require a major refactor of the core pipeline
- New query patterns incompatible with existing topology analysis tools
- Adds schema complexity that benefits only E. coli-style data; other bundles (zoo, periodic table) have no hyperedges
- The DS Wiki itself uses binary links — a hyperedge schema creates an impedance mismatch when computing bridges to DS Wiki entries

---

### Decision

**Option A (Reification).** Add `stoichiometry_coef REAL` to the RRP bundle links table only. Reactions, metabolites, and genes all become entries. Binary links with `consumes`/`produces`/`catalyzed_by` types encode stoichiometry.

The BIOMASS reaction (23 participants → 23 links) is the stress case — it works correctly and is actually more useful for bridge detection than a single hyperedge node would be.

---

### Implementation Checklist

- [ ] `src/ingestion/parsers/ecoli_core_parser.py` — parse COBRA JSON into entries/links
- [ ] Add `stoichiometry_coef` column to `rrp_bundle.py` `create_bundle_db()` schema
- [ ] Register `consumes`, `produces`, `catalyzed_by` as valid link types in `rrp_bundle.py`
- [ ] Run Pass 1.5 (`EntityCatalogPass`) — classify bundle type first; E. coli will likely classify as `process_network`, not `entity_catalog`
- [ ] Run Pass 2b (CrossUniverseQuery) against 1024-dim ChromaDB index
- [ ] Add parser tests to `tests/test_ecoli_parser.py`

---

### Expected Bundle Stats (estimates)

| Metric | Estimate |
|--------|----------|
| entries (reactions) | 95 |
| entries (metabolites) | 72 |
| entries (genes) | 137 |
| total entries | ~304 |
| links (stoichiometric) | ~360 (avg 3.8 × 95) |
| links (gene-reaction) | ~190 (avg 2 genes/reaction) |
| total links | ~550 |
| cross-universe bridges (est.) | 200–400 |
