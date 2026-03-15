# Principia Formal Diagnostics (PFD) — User Guide

**Last updated:** 2026-03-15 | **Status:** Live (Phases 0–2.3 stable, Phase 3 in development)

---

## What This Is

PFD is a **graph diagnostics engine** — it ingests research datasets (called RRPs: Research Reference Packages) and automatically analyzes them for structural coherence, dimensionality, and cross-domain analogies.

**In plain English:**
- You provide a structured dataset (metabolic networks, taxonomies, power grids, chemical properties, etc.)
- PFD checks: "Is this internally consistent? Are there hidden structural patterns? How does it relate to known scientific foundations?"
- You get back: diagnostic scores, detailed reports, and discovered analogies

**This is a diagnostic tool, not a judge.** Every output shows *why* it reached its conclusion. No black-box verdicts.

---

## Who This Is For

- **Researchers with complex datasets:** metabolic networks, taxonomies, electrical grids, chemical databases, knowledge graphs
- **Data engineers:** validating dataset quality before ML pipelines
- **Educators:** analyzing knowledge structures (curricula, textbook organizations)
- **Anyone building formal models** of their domain

**Not for:** general text analysis, unstructured data, or claims without explicit structure.

---

## What You Can Do (Right Now)

### ✅ Phase 2 — Live (Stable)

1. **Ingest your dataset** → Convert it to PFD's Research Reference Package (RRP) schema
2. **Run internal diagnostics** → Check if your dataset is internally coherent (Tier-1 report)
3. **Run bridge analysis** → Compare your dataset against the reference wiki of scientific foundations
4. **View detailed reports** → Topology hubs, noise characterization, dimensionality analysis

### ⏳ Phase 3 — In Development

- **Claim extraction:** Extract testable claims from paper PDFs or text
- **Claim resolution:** Match claims against the reference wiki for validation
- **Structural alignment:** Signed polarity scoring for paper-based RRPs

### 📋 Phase 4 — Future

- **Formal logic layer:** Deeper axiom-based validation with confidence bounds

---

## The Pipeline (6 Steps)

When you use PFD, here's what happens behind the scenes.

```
Your Dataset (CSV / JSON / MATPOWER / PDF / custom)
        ↓
[Step 1] INGEST         -> Convert to RRP schema (standardized entries + links)
        ↓
[Step 2] BUILD GRAPH    -> Model your dataset's structure
        ↓
[Step 3] INTERNAL DIAG  -> Compute coherence, noise, topology
        ↓ (You get Tier-1 Report here)
        ├─→ Internally Consistent ✓
        ├─→ Marginal (mixed signals)
        └─→ Fragmented (high noise)
        ↓
[Step 4] BUILD BRIDGE   -> Compare to reference wiki (semantic similarity)
        ↓
[Step 5] BRIDGE DIAG    -> Find structural analogies, domain overlaps
        ↓ (You get Tier-2 Report here)
        ├─→ Well-Integrated (high analogy score)
        ├─→ Partial (some connections)
        └─→ Isolated (no analogies found)
        ↓
[Step 6] GENERATE REPORT -> Two-tier output with PFD Score (0.0–1.0)
        ↓
Two-Tier Report + Visualization
```

---

## Quick Start (5 Minutes)

### 1. Install

```bash
git clone https://github.com/IanD25/principia-diagnostics.git
cd principia-diagnostics
bash setup.sh
```

(See [Setup Details](#troubleshooting) below for troubleshooting.)

### 2. Activate and Run

```bash
source .venv/bin/activate

# Run on the included E. coli example dataset
pfd report --rrp data/rrp/ecoli_core/rrp_ecoli_core.db
```

### 3. View Results

Open the generated HTML report in your browser (saved to `data/reports/`).
You'll see an interactive D3.js network visualization, coherence charts, and bridge quality histograms.

### 4. Run Internal-Only Analysis (No Bridge Step)

```bash
pfd internal --rrp data/rrp/ecoli_core/rrp_ecoli_core.db
```

This gives you a Tier-1 report without comparing to the reference wiki — useful for a quick structural check.

---

## Adding Your Own Dataset

PFD analyzes datasets by first converting them into an **RRP (Research Reference Package)** — a standardized SQLite database. The repo includes parsers for several formats, but you can also write your own.

### Option A: Write a Parser

Create `src/ingestion/parsers/your_parser.py` following existing patterns (see `ecoli_core_parser.py` for a complete example):

```python
from ingestion.rrp_bundle import create_rrp_bundle

conn = create_rrp_bundle("my_dataset.db", name="My Dataset", source="my_data.csv", fmt="csv")

# Insert entries
conn.execute("""INSERT INTO entries (id, title, entry_type, domain)
                VALUES (?, ?, ?, ?)""", ("E001", "Pyruvate", "instantiation", "biochemistry"))

# Insert links
conn.execute("""INSERT INTO links (link_type, source_id, source_label, target_id, target_label)
                VALUES (?, ?, ?, ?, ?)""", ("produces", "E001", "E001", "E002", "E002"))

conn.commit()
conn.close()
```

### Option B: Follow the CSV Convention

Prepare two CSV files and write a short parser that reads them:

**entries.csv:**
```csv
entry_id,title,description,entry_type,domain
E001,Pyruvate,Central metabolite in glycolysis,instantiation,biochemistry
E002,Acetyl-CoA,Carrier of acetyl groups,instantiation,biochemistry
```

**links.csv:**
```csv
source_id,target_id,link_type,description
E001,E002,produces,Pyruvate → Acetyl-CoA in TCA cycle
```

### After Creating Your RRP

```bash
# Build semantic bridges to the reference wiki
python scripts/run_entity_catalog_pass.py your_rrp.db data/chroma_db data/ds_wiki.db

# Run full analysis
pfd report --rrp your_rrp.db
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed parser writing instructions.

---

## Understanding Your Report

### Tier-1 Verdict (Internal Coherence)

Answers: **"Is your dataset internally consistent?"**

- **INTERNALLY CONSISTENT** (✓)
  - Non-noise fraction > 66%
  - Meaning: Most links in your data are "signal," not random connections
  - Example: E. coli metabolic network (81.3% coherent)

- **MARGINAL**
  - Non-noise fraction 40–66%
  - Meaning: Mixed signals; some real structure + some noise
  - Example: IEEE power grid case14 (66.7% coherent)

- **FRAGMENTED**
  - Non-noise fraction < 40%
  - Meaning: Mostly noise; hard to identify real structure
  - Action: Review your data for errors, missing entries, or over-linking

### Tier-2 Verdict (Cross-Domain Analogies)

Answers: **"How does your dataset relate to known scientific foundations?"**

- **WELL-INTEGRATED** (✓✓)
  - > 75% of your entries match structures in the reference wiki
  - Mean analogy similarity > 0.75
  - Meaning: Strong structural parallels to established science

- **PARTIAL**
  - 40–75% entry reach, or 0.55–0.75 similarity
  - Meaning: Some analogies exist; domain connections are loose

- **ISOLATED**
  - < 40% entry reach, or < 0.55 similarity
  - Meaning: No strong structural matches; novel or orthogonal domain

### PFD Score (Final Grade)

```
PFD Score = 0.5 × (Tier-1 Coherence) + 0.5 × (Tier-2 Integration)
```

Range: **0.0 (no structure) to 1.0 (perfect coherence + integration)**

Example:
- E. coli (Tier-1: 0.81, Tier-2: 1.0) → **PFD Score 0.91** ✓✓
- CCBH cosmology (Tier-1: 0.77, Tier-2: 1.0) → **PFD Score 0.88** ✓
- IEEE Power Grid (Tier-1: 0.67, Tier-2: 1.0) → **PFD Score 0.83** ✓

---

## Technical Concepts (If You Care)

### D_eff (Effective Dimensionality)

How "hub-like" vs. "distributed" is your network?

- **D_eff ≈ 1.5–2.0** → Planar/near-planar (e.g., IEEE power grids)
- **D_eff ≈ 5–8** → Modular hubs (e.g., taxonomies, organizational structures)
- **D_eff ≈ 10+** → Highly distributed (e.g., metabolic networks, random graphs)

### Regime Classification

How does your network's topology distribute?

- **Isotropic** (balanced): Links spread evenly (like social networks)
- **Radial-Dominated** (hub-like): Few hubs, many leaves (like airline routes)
- **Noise-Dominated** (random): Links appear uncorrelated (suggests data error)

### Similarity Score

When comparing entries to the reference wiki, PFD computes a **similarity score (0.0–1.0)** using BGE semantic embeddings:
- Tier 1: ≥ 0.85 (strong match)
- Tier 1.5: ≥ 0.75 (moderate match)
- Tier 2: < 0.75 (weak match)

---

## Data Format Reference

### entries table

| Column | Type | Required? | Notes |
|--------|------|-----------|-------|
| id | string | ✓ | Unique entry identifier |
| title | string | ✓ | Human-readable name |
| entry_type | enum | ✓ | reference_law, instantiation, method, constraint, etc. |
| domain | string | ✓ | biochemistry, physics, mathematics, etc. |
| source_type | string | | paper_section, database_record, etc. |
| status | string | | established, superseded |
| confidence | string | | Tier 1, Tier 2, etc. |

### links table

| Column | Type | Required? | Notes |
|--------|------|-----------|-------|
| source_id | string | ✓ | Must exist in entries |
| target_id | string | ✓ | Must exist in entries |
| link_type | string | ✓ | derives_from, analogous_to, produces, etc. |
| confidence_tier | string | | 1, 1.5, or 2 |

---

## Example: E. coli Metabolic Network

Here's a real example from the repo:

**Data:**
- 304 entries (metabolites, reactions, pathways)
- 536 links (biochemical dependencies)

**Tier-1 Result:**
```
Coherence:    81.3% (INTERNALLY CONSISTENT ✓)
D_eff:        11.2 (highly distributed metabolic hubs)
Noise:        7.8%
Top Hub:      met_pyr_c (pyruvate — 23 links, central TCA intermediate)
```

**Tier-2 Result:**
```
Bridges:      912 cross-domain analogies
Top Anchor:   CHEM5 (134 bridges — chemistry hub)
Verdict:      WELL-INTEGRATED
```

**Final PFD Score:** 0.973/1.0 ✓✓

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `python: command not found` | Python not installed | Install Python 3.11+ from [python.org](https://python.org) |
| `ModuleNotFoundError` | Virtual environment not activated | Run `source .venv/bin/activate` |
| First run slow (~5 min) | Downloading embedding model (~430MB) | Normal — cached after first download |
| `sqlite3.OperationalError: no such table` | ChromaDB index not built | Run `python3 -m sync` |
| Unicode errors on Windows | Python not in UTF-8 mode | Set `PYTHONUTF8=1` before running |
| `pfd: command not found` | Package not installed | Run `pip install -e .` from repo root |
| `ModuleNotFoundError: pandapower` | IEEE parser needs optional dep | `pip install pandapower` |

For more help: [Open an issue on GitHub](https://github.com/IanD25/principia-diagnostics/issues).

---

## Next Steps

1. **Run the Quick Start** → Get a Tier-1 report on the included E. coli data
2. **Add your own dataset** → Write a parser or follow the CSV convention
3. **Review your report** → Understand what D_eff and coherence mean for your domain
4. **Give feedback** → What was confusing? What would help? [Open an issue](https://github.com/IanD25/principia-diagnostics/issues)

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | GitHub overview + features |
| [USER_GUIDE.md](USER_GUIDE.md) | **← You are here** — High-level workflow |
| [docs/FISHER_PIPELINE_REDESIGN.md](docs/FISHER_PIPELINE_REDESIGN.md) | 6-step pipeline specification (technical) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

---

## License & Attribution

[PFD is licensed under MIT.](LICENSE)

**Citation:**
```bibtex
@software{pfd_2026,
  title={Principia Formal Diagnostics: Graph Coherence Analysis for Research Datasets},
  author={Darling, Ian},
  year={2026},
  url={https://github.com/IanD25/principia-diagnostics}
}
```

---

**Questions?** [Open an issue](https://github.com/IanD25/principia-diagnostics/issues) — we'd love to hear from you.
