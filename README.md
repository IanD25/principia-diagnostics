# Principia Diagnostics

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

Example HTML reports are included in `data/reports/examples/`.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/IanD25/principia-diagnostics.git
cd principia-diagnostics
bash setup.sh

# Activate environment
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# Run a full two-tier diagnostic report
pfd report --rrp data/rrp/ecoli_core/rrp_ecoli_core.db

# See all commands
pfd --help
```

### Alternative: pip install

```bash
git clone https://github.com/IanD25/principia-diagnostics.git
cd principia-diagnostics
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pfd --help
```

## Commands

```bash
# Full PFD report (Tier-1 + Tier-2 + PFD Score)
pfd report --rrp data/rrp/ecoli_core/rrp_ecoli_core.db

# Internal structural analysis only (Tier-1)
pfd internal --rrp data/rrp/ecoli_core/rrp_ecoli_core.db

# Bridge analysis only (Tier-2)
pfd bridge --rrp data/rrp/ecoli_core/rrp_ecoli_core.db

# Reference wiki health check
pfd wiki

# Single node deep-dive
pfd node --node-id CHEM5

# Extract claims from text (Phase 3)
pfd extract "We find that k = 3.11 at 90% confidence"

# Resolve a claim against the reference wiki
pfd resolve "entropy increases with dimension"

# Structural alignment on an RRP
pfd align --rrp data/rrp/ccbh/rrp_ccbh_cluster.db
```

## How It Works

### The Reference Wiki (DS Wiki)

A curated knowledge graph of 209 scientific and logical foundations — conservation laws, thermodynamic bounds, statistical distributions, equilibrium conditions, and more. Each entry has:
- Formal description and implications
- Typed links to related entries (derives_from, analogous_to, constrains, etc.)
- Semantic embeddings for similarity search

### Fisher Information Matrix (FIM) Diagnostics

Each node in a graph gets analyzed using FIM decomposition:
- **D_eff (effective dimension)** — how many independent information channels a node participates in
- **Coherence score** — how well a node's local structure matches its expected behavior
- **Regime classification** — bulk, surface, bridge, or isolated

### Two-Tier Reports

- **Tier-1 (Internal):** How coherent is the dataset's own internal structure? D_eff distribution, regime breakdown, hub identification.
- **Tier-2 (Bridge):** How well does the dataset connect to known scientific foundations? Bridge quality, domain coverage, anchor distribution.

### Claim Extraction (Phase 3)

Pattern-based extraction of testable claims from paper prose, with polarity detection (positive/negative/neutral) and mandatory human approval gate before downstream processing.

## Requirements

- Python 3.11+
- ~500MB disk (all reference data committed)
- CPU, CUDA (RTX 2000+), or Apple Silicon — auto-detected

**Core dependencies** (installed automatically):
```
sentence-transformers  # BGE embeddings (auto-downloaded from HuggingFace)
chromadb               # Semantic vector indexing
numpy, plotly, matplotlib, networkx  # Numerics + visualization
```

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
|   |   |-- parsers/           # One parser per format
|   |   +-- cross_universe_query.py  # Dataset <-> Wiki bridge detection
|   +-- viz/                   # Visualization (D3.js, Plotly)
|       |-- tier1_dashboard.py # Network graph + charts
|       |-- tier1_report.py    # Tier-1 HTML report
|       +-- tier2_report.py    # Tier-2 HTML report
|-- scripts/
|   |-- run_fisher_suite.py    # Main analysis CLI (6 modes)
|   +-- run_entity_catalog_pass.py
|-- data/
|   |-- ds_wiki.db             # Reference knowledge graph
|   +-- rrp/                   # Example datasets
|       |-- ecoli_core/        # E. coli metabolic network
|       +-- ccbh/              # Cosmological coupling papers
+-- tests/                     # Test suite
```

## Adding Your Own Dataset

The most impactful way to use Principia is to run it on your own data. See existing parsers in `src/ingestion/parsers/` for patterns:

1. Create `src/ingestion/parsers/your_parser.py`
2. Parse your dataset into the RRP schema (entries, links, sections, properties)
3. Use `create_rrp_bundle()` from `ingestion.rrp_bundle`
4. Build cross-universe bridges: `python scripts/run_entity_catalog_pass.py your_rrp.db data/chroma_db data/ds_wiki.db`
5. Run analysis: `pfd report --rrp your_rrp.db`

## Design Philosophy

- **Diagnostic, not judge** — reports show reasoning, not verdicts
- **Probabilistic, not boolean** — confidence scores (0-1), never VALID/INVALID
- **Transparent** — every output shows full reasoning chain
- **Reproducible** — SQLite databases, deterministic pipelines, no API keys needed
- **Falsifiable** — system is provably wrong when it is; failures are understandable and fixable

## License

MIT -- see [LICENSE](LICENSE).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and code standards.
