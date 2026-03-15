# Principia Diagnostics

[![Tests](https://github.com/IanD25/principia-diagnostics/actions/workflows/tests.yml/badge.svg)](https://github.com/IanD25/principia-diagnostics/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Graph coherence engine for research datasets.**

Give Principia a structured dataset — metabolic networks, taxonomies, power grids, chemical databases, knowledge graphs — and it produces a diagnostic report showing how internally coherent the dataset is and how well it grounds to known scientific foundations.

## What It Does

```
Your Dataset (CSV / JSON / MATPOWER / custom)
    |
[1] INGEST         -> entries + links -> rrp_*.db (SQLite)
[2] BUILD GRAPH    -> NetworkX internal graph
[3] TIER-1         -> Fisher effective dimension (D_eff), coherence score,
    |                  regime distribution -> HTML report + D3.js network viz
    |
[4] BUILD BRIDGE   -> Dataset nodes <-> Reference Wiki nodes (semantic similarity)
[5] TIER-2         -> Bridge quality, anchor distribution, domain coverage
    |                  -> HTML report
    |
[6] PFD SCORE      -> 0.0-1.0 combined diagnostic verdict with full reasoning
```

**Key principle:** No black-box verdicts. Every score shows its full reasoning chain — what was measured, what thresholds applied, and why.

## Example Results

| Dataset | Entries | Links | PFD Score | Verdict |
|---------|---------|-------|-----------|---------|
| E. coli core metabolic network | 304 | 536 | 0.973 | CONSISTENT + WELL-INTEGRATED |
| Zoo animal taxonomy | 426 | 437 | -- | CONSISTENT |
| Periodic Table | 119 | 1,671 | -- | CONSISTENT |
| IEEE Power Grids (case14/57/118) | 14-118 | varies | -- | MARGINAL (domain-correct) |
| CCBH cosmology cluster (3 papers) | 22 | 29 | 0.882 | MARGINAL + WELL-INTEGRATED |

Example HTML reports (Tier-1 and Tier-2 for E. coli) are included in `data/reports/examples/` — open them in any browser to see what the output looks like.

---

## Prerequisites

- **Python 3.11 or higher** (tested on 3.11, 3.12, 3.13)
- **~1GB disk space** (repo data + embedding model download on first run)
- **No API keys needed** — everything runs locally
- Works on: macOS (Intel or Apple Silicon), Linux, Windows (WSL recommended)

---

## Setup (Mac / Linux)

```bash
# 1. Clone the repository
git clone https://github.com/IanD25/principia-diagnostics.git
cd principia-diagnostics

# 2. Run the setup script
#    This will:
#    - Create a Python virtual environment (.venv/)
#    - Install all dependencies via pip
#    - Download the BGE embedding model (~430MB, one-time)
#    - Build the semantic search index from the reference wiki
#    - Optionally run the test suite
#
#    First run takes 2-5 minutes depending on internet speed.
bash setup.sh

# 3. Activate the virtual environment
#    (setup.sh does this internally, but you need to do it
#     again in each new terminal window)
source .venv/bin/activate

# 4. Run your first analysis
pfd report --rrp data/rrp/ecoli_core/rrp_ecoli_core.db
```

The `pfd report` command will print a diagnostic summary to the terminal and save HTML reports to `data/reports/`. Open them in a browser to see the interactive D3.js network graph.

### Setup (Windows)

```powershell
# 1. Clone
git clone https://github.com/IanD25/principia-diagnostics.git
cd principia-diagnostics

# 2. Create virtual environment manually
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Build the semantic index (first run downloads the embedding model)
python -m sync

# 5. Run analysis (PYTHONUTF8=1 handles Unicode math symbols in the reference wiki)
set PYTHONUTF8=1
pfd report --rrp data/rrp/ecoli_core/rrp_ecoli_core.db
```

### Alternative: Manual pip install (Mac / Linux)

If you prefer not to use the setup script:

```bash
git clone https://github.com/IanD25/principia-diagnostics.git
cd principia-diagnostics
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -m sync          # Build semantic index (~1 min first run)
pfd --help
```

---

## Commands

```bash
# Full PFD report (Tier-1 + Tier-2 + PFD Score)
pfd report --rrp data/rrp/ecoli_core/rrp_ecoli_core.db

# Internal structural analysis only (Tier-1, no reference wiki needed)
pfd internal --rrp data/rrp/ecoli_core/rrp_ecoli_core.db

# Bridge analysis only (Tier-2)
pfd bridge --rrp data/rrp/ecoli_core/rrp_ecoli_core.db

# Reference wiki health check
pfd wiki

# Single node deep-dive
pfd node --node-id CHEM5

# Extract claims from text
pfd extract "We find that k = 3.11 at 90% confidence"

# Resolve a claim against the reference wiki
pfd resolve "entropy increases with dimension"

# Structural alignment on an RRP
pfd align --rrp data/rrp/ccbh/rrp_ccbh_cluster.db
```

### What to Expect

Running `pfd report` produces:
- **Terminal output:** Summary statistics — entry count, link count, D_eff distribution, regime breakdown, PFD Score
- **HTML reports:** Saved to `data/reports/` — interactive D3.js network visualization, coherence charts, bridge histograms
- **Runtime:** 10-30 seconds depending on dataset size and hardware

---

## How It Works

### The Reference Wiki

A curated knowledge graph of 209 scientific and logical foundations — conservation laws, thermodynamic bounds, statistical distributions, equilibrium conditions, and more. Each entry has:
- Formal description and implications
- Typed links to related entries (derives_from, analogous_to, constrains, etc.)
- Semantic embeddings for similarity search (generated automatically from entry text)

### Fisher Information Matrix (FIM) Diagnostics

Each node in a graph gets analyzed using FIM decomposition:
- **D_eff (effective dimension)** — how many independent information channels a node participates in
- **Coherence score** — how well a node's local structure matches expected behavior
- **Regime classification** — bulk (well-connected), surface (peripheral), bridge (connecting clusters), or isolated

### Two-Tier Reports

- **Tier-1 (Internal):** How coherent is the dataset's own internal structure? Includes D_eff distribution, regime breakdown, hub identification, and an interactive network graph.
- **Tier-2 (Bridge):** How well does the dataset connect to known scientific foundations? Includes bridge quality scores, domain coverage heatmap, and anchor distribution.

### Claim Extraction

Pattern-based extraction of testable claims from paper prose, with polarity detection (positive/negative/neutral) and mandatory human approval gate before downstream processing. No LLM or API key required.

---

## Adding Your Own Dataset

Principia analyzes datasets by first converting them into an **RRP (Research Reference Package)** — a standardized SQLite database containing entries (nodes), links (edges), and metadata. The repo includes six parsers for different formats.

### Quick version (if your data is already in CSV)

Prepare two CSV files:

**entries.csv** — one row per entity:
```csv
entry_id,title,description,entry_type,domain
NODE1,My First Entry,Description of what this represents,reference_law,physics
NODE2,My Second Entry,Another entity in the dataset,reference_law,chemistry
```

**links.csv** — one row per relationship:
```csv
source_id,target_id,link_type,confidence_tier
NODE1,NODE2,derives_from,1.5
```

Then write a parser (see `src/ingestion/parsers/ecoli_core_parser.py` for a complete example):

```python
from ingestion.rrp_bundle import create_rrp_bundle

# Your parsing logic here — read your data, build entries and links
entries = [...]
links = [...]

create_rrp_bundle("my_dataset.db", entries, links, sections, properties)
```

### Full steps

1. Create `src/ingestion/parsers/your_parser.py` (follow existing parser patterns)
2. Parse your dataset into entries, links, sections, and properties
3. Call `create_rrp_bundle()` to produce an RRP SQLite database
4. Build semantic bridges to the reference wiki:
   ```bash
   python scripts/run_entity_catalog_pass.py your_rrp.db data/chroma_db data/ds_wiki.db
   ```
5. Run analysis:
   ```bash
   pfd report --rrp your_rrp.db
   ```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `python: command not found` | Python not installed or not in PATH | Install Python 3.11+ from [python.org](https://python.org) or via `brew install python@3.13` (Mac) |
| `ModuleNotFoundError` after clone | Virtual environment not activated | Run `source .venv/bin/activate` first |
| First run is slow (~5 min) | Downloading embedding model (~430MB) | Normal — only happens once. Model is cached locally after download. |
| `sqlite3.OperationalError: no such table` | ChromaDB index not built | Run `python3 -m sync` |
| Unicode errors on Windows | Python not in UTF-8 mode | Set `PYTHONUTF8=1` before running commands |
| `pfd: command not found` | Package not installed in editable mode | Run `pip install -e .` from the repo root |

---

## Requirements

- **Python 3.11+** (tested on 3.11, 3.12, 3.13)
- **~1GB disk** (repo data + embedding model)
- **CPU, CUDA (RTX 2000+), or Apple Silicon** — auto-detected, no configuration needed

**Core dependencies** (installed automatically by setup.sh or pip):
```
sentence-transformers  # BGE embeddings (auto-downloaded from HuggingFace)
chromadb               # Semantic vector indexing
numpy                  # Numerics
plotly, matplotlib     # Visualization
networkx               # Graph analysis
```

---

## Project Structure

```
principia-diagnostics/
|-- src/
|   |-- config.py              # All paths, model config, thresholds
|   |-- sync.py                # Rebuild ChromaDB from ds_wiki.db
|   |-- cli.py                 # Unified CLI entry point
|   |-- embedder.py            # BGE embedding pipeline
|   |-- analysis/              # Diagnostic tools
|   |   |-- fisher_diagnostics.py   # Core FIM math
|   |   |-- fisher_report.py        # Two-tier PFD report
|   |   |-- claim_extractor.py      # Phase 3 claim extraction
|   |   |-- result_validator.py     # Claim validation + resolution
|   |   +-- structural_alignment.py # Signed polarity scoring
|   |-- ingestion/             # Dataset parsers
|   |   |-- parsers/           # One parser per format (6 included)
|   |   +-- cross_universe_query.py  # Dataset <-> Wiki bridge detection
|   +-- viz/                   # Visualization (D3.js, Plotly)
|       |-- tier1_dashboard.py # Network graph + charts
|       |-- tier1_report.py    # Tier-1 HTML report
|       +-- tier2_report.py    # Tier-2 HTML report
|-- scripts/
|   |-- run_fisher_suite.py    # Main analysis CLI (6 modes)
|   +-- run_entity_catalog_pass.py
|-- data/
|   |-- ds_wiki.db             # Reference knowledge graph (209 entries)
|   +-- rrp/                   # Example datasets
|       |-- ecoli_core/        # E. coli metabolic network (PFD 0.973)
|       +-- ccbh/              # Cosmological coupling papers (PFD 0.882)
+-- tests/                     # Test suite (450+ tests)
```

---

## Design Philosophy

- **Diagnostic, not judge** — reports show reasoning, not verdicts
- **Probabilistic, not boolean** — confidence scores (0-1), never VALID/INVALID
- **Transparent** — every output shows full reasoning chain
- **Reproducible** — SQLite databases, deterministic pipelines, no API keys needed
- **Falsifiable** — system is provably wrong when it is; failures are understandable and fixable

---

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and code standards.

The most impactful contribution is **adding a parser for a new dataset format** — see `src/ingestion/parsers/` for examples.
