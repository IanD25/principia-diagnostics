# Principia Formal Diagnostics (PFD) — User Guide

**Last updated:** 2026-03-11 | **Status:** Live (Phases 0–2.3 stable, Phase 3+ planned)

---

## What This Is

PFD is a **graph diagnostics engine** — it ingests research datasets (called RRPs: Research Reference Packages) and automatically analyzes them for structural coherence, dimensionality, and cross-domain analogies.

**In plain English:**
- You upload 1+ datasets of your research (metabolic networks, taxonomies, power grids, chemical properties, etc.)
- PFD checks: "Is this internally consistent? Are there hidden structural patterns? How does it compare to other datasets?"
- You get back: diagnostic scores, detailed reports, and discovered analogies between your datasets

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
3. **Compare datasets** → Upload 2+ of your own RRPs, run bridge analysis to find structural analogies
4. **View detailed reports** → Topology hubs, noise characterization, dimensionality analysis

### ⏳ Phase 3 — Planned (Q2 2026)

- **Claim validation:** Submit explicit claims about your data; PFD validates them against formal axioms
- **Formal proof paths:** See the logical chain supporting (or refuting) your claims
- **Fallacy detection:** Automatic identification of common logical errors

### 📋 Phase 4 — Future (Q3+ 2026)

- **Formal logic layer:** Deeper axiom-based validation with confidence bounds

---

## The Pipeline (6 Steps)

When you use PFD, here's what happens behind the scenes.

**[→ View Interactive Workflow Diagram](PFD_WORKFLOW_DIAGRAM.png)**

```
Your Dataset (Excel, CSV, JSON, SQLite, etc.)
        ↓
[Step 1] INGEST → Convert to RRP schema (standardized entries + links)
        ↓
[Step 2] BUILD INTERNAL GRAPH → Model your dataset's structure
        ↓
[Step 3] INTERNAL DIAGNOSTICS → Compute coherence, noise, topology
        ↓ (You get Tier-1 Report here)
        ├─→ Internally Consistent ✓
        ├─→ Marginal (mixed signals)
        └─→ Fragmented (high noise)
        ↓
[Step 4] BUILD BRIDGE GRAPH → (Optional) Compare to other RRPs
        ↓
[Step 5] BRIDGE DIAGNOSTICS → Find structural analogies, domain overlaps
        ↓ (You get Tier-2 Report here)
        ├─→ Well-Integrated (high analogy score)
        ├─→ Partial (some connections)
        └─→ Isolated (no analogies found)
        ↓
[Step 6] GENERATE REPORT → Two-tier output with PFD Score (0.0–1.0)
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

(See [Setup Details](#detailed-setup) for troubleshooting.)

### 2. Prepare Your Data

PFD expects two CSV files:

**entries.csv** — Things in your dataset
```csv
entry_id,title,description,entry_type,domain
E001,Pyruvate,Central metabolite in glycolysis,instantiation,biochemistry
E002,Acetyl-CoA,Carrier of acetyl groups,instantiation,biochemistry
...
```

**links.csv** — Relationships between entries
```csv
source_id,target_id,link_type,description
E001,E002,produces,Pyruvate → Acetyl-CoA in TCA cycle
E002,E003,activates,Acetyl-CoA activates downstream reactions
...
```

### 3. Run Ingestion (Step 1 + 2)

```bash
python3 scripts/ingest_your_dataset.py \
    --entries entries.csv \
    --links links.csv \
    --name my_dataset \
    --output data/rrp/my_dataset/
```

Creates: `data/rrp/my_dataset/rrp_my_dataset.db`

### 4. Run Diagnostics (Step 3)

```bash
python scripts/run_fisher_suite.py --mode internal_rrp \
    --rrp-db data/rrp/my_dataset/rrp_my_dataset.db
```

**Output:**
```
RRP Analysis: my_dataset
├─ Entries:        47
├─ Links:          89
├─ Internal Coherence: INTERNALLY CONSISTENT (81.3%)
├─ Mean D_eff:     8.2 (metabolic hub structure)
├─ Noise Fraction: 12.4%
└─ Verdict:        Tier-1 ✓
```

### 5. Compare to Another Dataset (Optional — Steps 4+5)

```bash
python scripts/run_fisher_suite.py --mode bridge \
    --rrp-db data/rrp/my_dataset/rrp_my_dataset.db \
    --rrp-compare data/rrp/another_dataset/rrp_another_dataset.db
```

**Output:**
```
Bridge Analysis: my_dataset ↔ another_dataset
├─ Bridges Found:     23 structural analogies
├─ Top Analogy:       E001 (pyruvate) ↔ X_Hub_7 (network hub)
│                     Similarity: 0.87 (isotropic topology match)
├─ Cross-Domain Reach: 89% (23/26 entries linked)
└─ Verdict:           WELL-INTEGRATED (Phase 2.3)
```

### 6. Get Full Report (All Steps 1–6)

```bash
python scripts/run_fisher_suite.py --mode report \
    --rrp-db data/rrp/my_dataset/rrp_my_dataset.db \
    --rrp-compare data/rrp/another_dataset/rrp_another_dataset.db
```

Generates:
- `rrp_my_dataset_report.json` (Tier-1 + Tier-2 diagnostics)
- `rrp_my_dataset_visualization.html` (interactive graph)
- `rrp_my_dataset_summary.txt` (human-readable verdict)

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

### Tier-2 Verdict (Cross-Dataset Analogies)

Answers: **"How does your dataset relate to other datasets?"**

- **WELL-INTEGRATED** (✓✓)
  - > 75% of your entries match structures in comparison dataset
  - Mean analogy similarity > 0.75
  - Meaning: You found structural analogies; domains are deeply connected

- **PARTIAL**
  - 40–75% entry reach, or 0.55–0.75 similarity
  - Meaning: Some analogies exist; domains are loosely connected

- **ISOLATED**
  - < 40% entry reach, or < 0.55 similarity
  - Meaning: No strong structural matches; domains are orthogonal

### PFD Score (Final Grade)

```
PFD Score = 0.5 × (Tier-1 Coherence) + 0.5 × (Tier-2 Integration)
```

Range: **0.0 (no structure) to 1.0 (perfect coherence + integration)**

Example:
- E. coli (Tier-1: 0.81, Tier-2: 1.0) → **PFD Score 0.91** ✓✓
- IEEE Power Grid (Tier-1: 0.67, Tier-2: 1.0) → **PFD Score 0.83** ✓
- Hypothetical noisy data (Tier-1: 0.35, Tier-2: 0.45) → **PFD Score 0.40** ⚠

---

## Common Workflows

### Workflow A: Validate a Single Dataset

```bash
# Ingest
python3 scripts/ingest_your_dataset.py --entries E.csv --links L.csv --name my_data --output data/rrp/my_data/

# Internal diagnostics
python scripts/run_fisher_suite.py --mode internal_rrp --rrp-db data/rrp/my_data/rrp_my_data.db

# Full report (single)
python scripts/run_fisher_suite.py --mode report --rrp-db data/rrp/my_data/rrp_my_data.db
```

**Output:** Tier-1 report + visualization

---

### Workflow B: Find Analogies Between Two Datasets

```bash
# Ingest both
python3 scripts/ingest_your_dataset.py --entries dataset1_entries.csv --links dataset1_links.csv --name dataset1 --output data/rrp/dataset1/
python3 scripts/ingest_your_dataset.py --entries dataset2_entries.csv --links dataset2_links.csv --name dataset2 --output data/rrp/dataset2/

# Compare
python scripts/run_fisher_suite.py --mode bridge \
    --rrp-db data/rrp/dataset1/rrp_dataset1.db \
    --rrp-compare data/rrp/dataset2/rrp_dataset2.db

# Full two-tier report
python scripts/run_fisher_suite.py --mode report \
    --rrp-db data/rrp/dataset1/rrp_dataset1.db \
    --rrp-compare data/rrp/dataset2/rrp_dataset2.db
```

**Output:** Tier-1 + Tier-2 report + bridge list + visualization

---

### Workflow C: Batch-Analyze Multiple Datasets

(Coming Phase 3+)

```bash
python scripts/batch_analyze.py --input-dir data/rrp/ --output results/
```

Will generate: All pairwise comparisons + summary correlation matrix

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

When comparing two entries across datasets, PFD computes a **similarity score (0.0–1.0)** based on:
- Structural role (both hubs? both leaves?)
- Connectivity pattern (same # of links?)
- Topological neighborhood (similar subgraphs?)

Similarity **> 0.75** is considered a strong analogy.

---

## Data Format Reference

### entries.csv Format

```csv
entry_id,title,description,entry_type,domain
E001,Pyruvate,Central metabolite,instantiation,biochemistry
E002,Acetyl-CoA,Activated group carrier,instantiation,biochemistry
E003,Glycolysis,ATP generation pathway,process,biochemistry
```

| Column | Type | Required? | Notes |
|--------|------|-----------|-------|
| entry_id | string | ✓ | Unique; e.g., E001, BUS_14, GENE_ABC |
| title | string | ✓ | Human-readable name |
| description | text | ✓ | What this entry is (1–2 sentences) |
| entry_type | enum | ✓ | instantiation, process, constraint, property, object, etc. |
| domain | string | ✓ | biochemistry, electrical_engineering, mathematics, etc. |

### links.csv Format

```csv
source_id,target_id,link_type,description
E001,E002,produces,Pyruvate → Acetyl-CoA in TCA cycle
E002,E003,activates,Acetyl-CoA activates TCA cycle
E001,E001,self_regulatory,Pyruvate inhibits upstream glycolysis
```

| Column | Type | Required? | Notes |
|--------|------|-----------|-------|
| source_id | string | ✓ | Must exist in entries.csv |
| target_id | string | ✓ | Must exist in entries.csv; can equal source_id (self-loop) |
| link_type | string | ✓ | activates, inhibits, produces, contains, follows, etc. |
| description | text | ✓ | Why this link exists (1 sentence) |

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

**Tier-2 Result (comparing to Periodic Table dataset):**
```
Bridges:      47 cross-domain analogies
Reach:        100% (all E. coli entries found matches)
Top Analogy:  met_pyr_c (metabolite hub) ↔ CHEM5 (chemistry entry)
              Similarity: 0.89 (both are distribution hubs in their domains)
Verdict:      WELL-INTEGRATED (0.89 bridge quality)
```

**Final PFD Score:** 0.94/1.0 ✓✓

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: pandapower` | Missing dependency | `pip install pandapower -q` |
| `sqlite3.OperationalError: no such table: entries` | Malformed CSV | Verify entries.csv has required columns (entry_id, title, description, entry_type, domain) |
| `Fisher Suite returns empty report` | RRP database missing entries | Re-run ingestion; check output DB with `sqlite3 rrp_my_data.db "SELECT COUNT(*) FROM entries"` |
| `Similarity scores all < 0.5` | Datasets too dissimilar | Expected if domains are unrelated; try comparing within-domain datasets first |
| Permission error on Windows | File locking | Ensure no other process has DB file open; close spreadsheet applications |

For more: Open an issue on [GitHub](https://github.com/IanD25/principia-diagnostics/issues).

---

## Next Steps

1. **Prepare your data** → Convert to entries.csv + links.csv
2. **Run Quick Start steps 1–4** → Get Tier-1 report
3. **Optional: Compare datasets** → Follow Workflow B for Tier-2
4. **Review report** → Understand what D_eff and coherence mean for your domain
5. **Give feedback** → What was confusing? What would help? [Open issue](https://github.com/IanD25/principia-diagnostics/issues)

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

[PFD is licensed under MIT.](LICENSE) See [AUTHORS.md](AUTHORS.md) for contributors.

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

**Questions?** Open an issue, email, or check the [FAQ](FAQ.md) (coming Phase 3).
