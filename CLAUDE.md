# Principia Formal Diagnostics — LLM Orchestration Guide

> **This file is auto-loaded by Claude Code (and similar LLM code assistants) on session start.**
> It tells the LLM how to drive the PFD diagnostic pipeline.

---

## What This System Is

Principia Formal Diagnostics (PFD) is a **graph coherence engine** designed to be **driven by an LLM code assistant** with human oversight at key decision points.

The LLM's role:
- Orchestrate the 6-step diagnostic pipeline
- Interpret results in natural language
- Explain *why* a dataset scored the way it did
- Surface surprising structural analogies between the dataset and known science
- Guide the human through claim review (Layer 1 human gate)

The CLI (`pfd`) provides the same pipeline for manual use or scripting, but the intended primary interface is an LLM reading this file and operating the tools.

---

## Pipeline — Step by Step

Run these steps in order. Each step has a CLI command and an explanation of what to check before proceeding.

### Step 1: Ingest — Convert dataset to RRP

**Goal:** Parse raw data into a standardized RRP (Research Reference Package) SQLite database.

```bash
# If a parser already exists for the format:
python src/ingestion/parsers/<parser_name>.py <input_file> <output_rrp.db>

# Check what parsers are available:
ls src/ingestion/parsers/
```

Available parsers: `ecoli_core_parser.py`, `zoo_classes_parser.py`, `periodic_table_parser.py`, `ieee_power_grid_parser.py`, `opera_paper_parser.py`, `ccbh_cluster_parser.py`, `pdf_parser.py`

**If no parser exists:** Write one. Follow `ecoli_core_parser.py` as the reference implementation. Use `create_rrp_bundle()` from `src/ingestion/rrp_bundle.py` to create the SQLite schema.

**For PDF papers:** Use `pdf_parser.py` which extracts sections, detects math formulas, and feeds into the claim extractor.

**Check before proceeding:**
```python
import sqlite3
conn = sqlite3.connect("your_rrp.db")
entries = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
print(f"Entries: {entries}, Links: {links}")
# Both should be > 0
```

### Step 2: Build internal graph (automatic)

This step is handled internally by the Fisher Suite. No manual action needed — Step 3 builds the graph automatically.

### Step 3: Internal diagnostics (Tier-1)

**Goal:** Check if the dataset is internally coherent — independent of any external reference.

```bash
pfd internal --rrp your_rrp.db
```

**What to report to the human:**
- **Verdict:** INTERNALLY CONSISTENT / MARGINAL / FRAGMENTED
- **D_eff distribution:** Are there clear hubs? What are the top 3-5 hubs and what do they represent?
- **Regime breakdown:** What % is bulk vs. surface vs. bridge vs. noise?
- **Non-noise fraction:** > 66% is good, 40-66% is marginal, < 40% is fragmented

**Decision point:** If FRAGMENTED, tell the human the dataset may have structural issues (missing links, data errors, over-linking) before proceeding to bridge analysis.

### Step 4: Build bridges to reference wiki

**Goal:** Find semantic connections between the dataset and the 209-entry reference knowledge graph (`ds_wiki.db`).

```bash
python scripts/run_entity_catalog_pass.py your_rrp.db data/chroma_db data/ds_wiki.db
```

**What this does:** Embeds each RRP entry using BGE-large, queries the ChromaDB index of reference wiki entries, and stores matches above the similarity threshold as "bridges" in the `cross_universe_bridges` table.

**Check before proceeding:**
```python
conn = sqlite3.connect("your_rrp.db")
bridges = conn.execute("SELECT COUNT(*) FROM cross_universe_bridges").fetchone()[0]
print(f"Bridges found: {bridges}")
# Should be > 0. If 0, the dataset may be too domain-specific for the current reference wiki.
```

### Step 5: Bridge diagnostics (Tier-2)

**Goal:** Assess how well the dataset connects to known scientific foundations.

```bash
pfd bridge --rrp your_rrp.db
```

**What to report to the human:**
- **Bridge count and quality:** How many bridges, mean similarity, tier distribution
- **Top anchors:** Which reference wiki entries are most connected — what does that say about the dataset's domain?
- **Domain coverage:** Which scientific domains does this dataset touch?
- **Verdict:** WELL-INTEGRATED / PARTIAL / ISOLATED

### Step 6: Full two-tier report

**Goal:** Generate the combined PFD Score and save HTML reports.

```bash
pfd report --rrp your_rrp.db
```

**What to report to the human:**
- **PFD Score (0.0–1.0):** Combined diagnostic verdict
- **Tier-1 summary:** Internal coherence verdict + key metrics
- **Tier-2 summary:** Bridge quality verdict + key metrics
- **Surprising findings:** Any unexpected bridges or domain connections
- **Actionable gaps:** Missing links, isolated entries, or underrepresented domains

**HTML reports** are saved to `data/reports/` — tell the human to open them in a browser for interactive D3.js network visualizations.

---

## Claim Extraction (Phase 3 — Paper Analysis)

For analyzing research papers rather than structured datasets:

### Extract claims from text or PDF

```bash
# From text
pfd extract "We find that k = 3.11 ± 1.19 at 90% confidence"

# From PDF
pfd extract --file paper.pdf
```

**MANDATORY HUMAN GATE:** The output is formatted for human review. Present the extracted claims to the human and ask them to:
1. Confirm each claim is correctly extracted
2. Remove any false positives
3. Add any claims the extractor missed

**Do NOT proceed to resolution without human approval.**

### Resolve claims against the reference wiki

```bash
pfd resolve "entropy increases with dimension"
```

**What to report:** Which reference wiki entries support, contradict, or relate to the claim, with similarity scores and reasoning.

### Structural alignment (signed polarity)

```bash
pfd align --rrp your_rrp.db
```

**What to report:** Mean polarity score, contested entries (mixed support/contradiction from reference wiki), and aligned entries.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/cli.py` | CLI entry point — all `pfd` commands route through here |
| `src/config.py` | All paths, model name, thresholds — check here first |
| `src/sync.py` | Rebuild ChromaDB from ds_wiki.db (run after wiki changes) |
| `src/embedder.py` | BGE embedding pipeline |
| `src/analysis/fisher_diagnostics.py` | Core FIM math — `analyze_node`, `sweep_graph`, `build_bridge_graph` |
| `src/analysis/fisher_report.py` | Two-tier PFD report generator |
| `src/analysis/claim_extractor.py` | Pattern-based claim extraction with human gate |
| `src/analysis/result_validator.py` | Claim resolution against reference wiki |
| `src/analysis/structural_alignment.py` | Signed polarity scoring |
| `src/ingestion/rrp_bundle.py` | RRP SQLite schema + `create_rrp_bundle()` |
| `src/ingestion/cross_universe_query.py` | Bridge detection (RRP → reference wiki) |
| `scripts/run_fisher_suite.py` | Fisher Suite CLI (6 modes) — called by `pfd` |
| `scripts/run_entity_catalog_pass.py` | Pass 1.5 + Pass 2 bridge building |
| `data/ds_wiki.db` | Reference knowledge graph (209 entries, 573 links) — READ ONLY |
| `data/chroma_db/` | ChromaDB semantic index (bge-large 1024-dim) |

---

## Architectural Constraints

1. **Never schema-alter `ds_wiki.db`** — it is the read-only reference knowledge graph
2. **Probabilistic, not boolean** — never say VALID/INVALID. Always report confidence scores (0–1)
3. **Mandatory human gate at claim extraction** — always pause for human review before resolution
4. **Transparency** — every score must be explainable. Show what was measured, what thresholds applied, why
5. **INSERT OR IGNORE** — all scripts safe to re-run
6. **No API keys needed** — everything runs locally (embeddings, similarity, diagnostics)

---

## Reference Wiki Scale

- **209 entries:** conservation laws, thermodynamic bounds, distributions, equilibrium conditions, symmetry principles, complexity theory, information theory, biology fundamentals
- **573 typed links:** derives_from, analogous_to, constrains, predicts_for, tensions_with
- **1,486 ChromaDB chunks** with BGE-large 1024-dim embeddings
- Domains: physics, chemistry, biology, mathematics, computer science, information theory, statistics, crystallography, materials science

---

## Example Workflow

A human says: "Analyze this E. coli metabolic dataset for me."

1. Check if an RRP already exists: `ls data/rrp/ecoli_core/`
2. It does → skip ingestion
3. Run `pfd report --rrp data/rrp/ecoli_core/rrp_ecoli_core.db`
4. Read the terminal output
5. Report to the human:
   - "The E. coli metabolic network scores **0.973/1.0** — INTERNALLY CONSISTENT and WELL-INTEGRATED."
   - "Top hub: pyruvate (D_eff=11, 23 links) — central TCA intermediate as expected."
   - "Strongest reference wiki connection: CHEM5 (134 bridges) — the dataset is deeply grounded in chemistry fundamentals."
   - "HTML reports saved to `data/reports/` — open in browser for interactive network visualization."
6. Point the human to the HTML reports for visual exploration

---

## MCP Server (Optional — Advanced)

For LLM code assistants that support Model Context Protocol (MCP), the working/development repo includes `src/mcp_server.py` which exposes all diagnostic tools as MCP endpoints. This allows direct tool invocation without CLI subprocess calls.

The public repo uses the CLI interface instead. Both interfaces access the same underlying pipeline.
