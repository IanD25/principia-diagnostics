# PFD Pipeline Redesign — Fisher Suite in the Full Diagnostic Architecture
**Version:** 1.0 | **Date:** 2026-03-11 | **Status:** Design — approved for implementation
**Supersedes:** Phase D–F scope in original Fisher Suite specification
**Prerequisite:** Fisher Suite Phase A–C complete (all tests passing)

---

## 0. The Problem This Solves

The original Fisher Suite spec (v1.0) treated DS Wiki as the primary analysis target and RRP
universes as things that get *compared to* DS Wiki. This is architecturally backwards.

**DS Wiki is a reference lake — a formal foundation for comparison.**
**The RRP universe is the subject under diagnosis.**

The PFD (Principia Formal Diagnostics) system should tell a researcher:
1. *Is my knowledge base internally consistent?* (internal analysis)
2. *How well does it connect to established formal foundations?* (bridge analysis)

These are two distinct questions requiring two distinct graph analyses, run in sequence.

---

## 1. The Six-Step Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — INGEST                                                │
│  Parse RRP source → entries + links verified and stored         │
│  Tools: Pass 1 parser, Pass 1.5 EntityCatalogPass              │
│  Output: rrp_<name>.db with entries + links tables             │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  STEP 2 — INTERNAL GRAPH                                        │
│  Build within-universe graph G_internal from entries + links    │
│  Nodes: RRP entries  |  Edges: within-RRP links (weighted)     │
│  Tool: build_wiki_graph(rrp_db)  ← already works today         │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  STEP 3 — INTERNAL DIAGNOSTICS                                  │
│  Run full Fisher suite on G_internal                            │
│  Run hypothesis consistency, link quality checks                │
│  Output: Tier-1 Report — internal consistency score            │
│  Questions answered:                                            │
│    - Are the within-universe links forming coherent clusters?   │
│    - Does each entry sit in an appropriate FIM regime?          │
│    - Are identified constraints consistent with FIM geometry?   │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  STEP 4 — BRIDGE GRAPH                                          │
│  Build G_bridge: full extended Option B graph                   │
│  Nodes: all RRP entries + all DS Wiki entries                   │
│  Edges:                                                         │
│    - within-RRP links (from Step 2)                             │
│    - within-DS Wiki links (from ds_wiki.db)                     │
│    - cross-universe bridges (from Pass 2 CrossUniverseQuery)    │
│  Tool: build_bridge_graph(rrp_db, wiki_db)  ← to be built      │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  STEP 5 — BRIDGE DIAGNOSTICS                                    │
│  Run full Fisher suite on G_bridge                              │
│  Output: Tier-2 Report — bridge quality score                  │
│  Questions answered:                                            │
│    - Does the RRP integrate into DS Wiki topology, or just      │
│      touch it at isolated points?                               │
│    - Which RRP entries are genuine cross-domain bridges         │
│      (isotropic in G_bridge, high d_eff)?                       │
│    - Which DS Wiki anchor nodes are over-loaded (too many       │
│      RRP entries mapping to one DS Wiki node)?                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│  STEP 6 — TWO-TIER OUTPUT                                       │
│  Combine Tier-1 + Tier-2 into unified diagnostic report         │
│  See Section 5 for report schema                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Option B Bridge Graph — Exact Definition

The bridge graph is a **full extended graph** containing both universes and all edge types.

### 2.1 Node Set

```
V(G_bridge) = V(G_rrp) ∪ V(G_wiki)
```

Node IDs are prefixed to prevent collision:
- RRP nodes: `rrp::<entry_id>` (e.g., `rrp::rxn_PFK`)
- DS Wiki nodes: `wiki::<entry_id>` (e.g., `wiki::CHEM5`)

### 2.2 Edge Set

Three edge types, all stored with a `type` attribute:

| Edge type | Source | Weight | `type` tag |
|-----------|--------|--------|-----------|
| `internal_rrp` | within-RRP links table | tier-based (same as internal) | `"rrp"` |
| `internal_wiki` | within-DS Wiki links table | tier-based | `"wiki"` |
| `bridge` | cross_universe_bridges table | `1.0 - similarity` (distance semantics) | `"bridge"` |

Bridge edge weight = `1.0 - similarity` so that high-similarity bridges are *short* (close)
and low-similarity bridges are *long* (far), consistent with Dijkstra distance semantics used
throughout the Fisher pipeline.

### 2.3 What Fisher Analysis Reveals on G_bridge

When `analyze_node()` is called on an RRP node in G_bridge:

| Result | Interpretation |
|--------|----------------|
| `regime = RADIAL_DOMINATED` | Entry connects to DS Wiki at one narrow point — specialized, not cross-domain |
| `regime = ISOTROPIC` | Entry genuinely distributes across multiple independent DS Wiki dimensions — true cross-domain instantiation |
| `regime = NOISE_DOMINATED` | Entry's bridge connections are statistically random — low-quality bridges or entry is under-specified |
| `d_eff = 1` | Single dominant pathway to DS Wiki foundation |
| `d_eff ≥ 3` | Entry simultaneously instantiates 3+ independent formal principles |

When `analyze_node()` is called on a DS Wiki node in G_bridge:

| Result | Interpretation |
|--------|----------------|
| High PR, many RRP bridges | This DS Wiki node is an over-loaded anchor — many RRP entries converge here |
| `d_eff = 1` on wiki node | The RRP entries that connect here all approach from the same direction — expected for narrow laws |
| `d_eff ≥ 2` on wiki node | Multiple independent RRP concepts co-instantiate this principle — strong validation |

---

## 3. New Phase Map (D–F Redefined)

The original Phase D–F from the Fisher Suite specification are redefined as follows.

### Phase D — `build_bridge_graph()` + Internal RRP CLI

**Deliverables:**
- `fisher_diagnostics.py`: add `build_bridge_graph(rrp_db, wiki_db) → nx.Graph`
- `run_fisher_suite.py`: add `--mode internal_rrp --rrp-db <path>` (Step 3 entry point)
- `run_fisher_suite.py`: add `--mode bridge --rrp-db <path>` (Step 5 entry point)
- Tests: `TestBuildBridgeGraph` in `test_fisher_diagnostics.py`

**`build_bridge_graph` contract:**
```python
def build_bridge_graph(
    rrp_db: str | Path,
    wiki_db: str | Path,
    min_bridge_similarity: float = 0.75,
) -> tuple[nx.Graph, dict[str, str]]:
    """
    Returns (G_bridge, node_source_map) where:
      G_bridge: undirected weighted graph
      node_source_map: {node_id → "rrp" | "wiki"}
    All node IDs are prefixed: "rrp::<id>" or "wiki::<id>"
    All edges carry attr: type ("rrp" | "wiki" | "bridge"), weight (float)
    """
```

**`min_bridge_similarity`:** Bridges below this threshold are excluded from G_bridge.
Default 0.75 matches tier-1.5 quality floor used throughout the system.

### Phase E — MCP Tools (Internal + Bridge modes)

**Deliverables:** Three analysis tools accessible via CLI:

| Tool | Description |
|------|-------------|
| `pfd node --node-id X` | Analyze a single node in either G_internal or G_bridge |
| `pfd internal --rrp X` | Run full sweep on an RRP universe's internal graph |
| `pfd bridge --rrp X` | Run full sweep on the bridge graph for an RRP+Wiki pair |

**Note:** DS Wiki self-analysis (`pfd wiki`) remains available for validating
that the reference wiki's own geometry is healthy.

### Phase F — `fisher_report.py` + Two-Tier Output

**Deliverables:**
- `src/analysis/fisher_report.py`: generates Tier-1 + Tier-2 diagnostic reports
- CLI: `run_fisher_suite.py --mode report --rrp-db <path>` runs Steps 3+5 and writes report

**Two-tier report schema:** See Section 5.

### Phase G — Documentation Update

Final documentation pass. All project documentation updated to reflect the full 6-step
pipeline as the canonical PFD architecture.

---

## 4. Internal Consistency Metrics (Tier-1 Report Inputs)

The following metrics are computed from the internal graph sweep (Step 3):

| Metric | How Computed | Good Sign | Warning Sign |
|--------|-------------|-----------|--------------|
| **Internal coherence score** | Fraction of nodes with `regime != NOISE_DOMINATED` | > 0.80 | < 0.60 |
| **Mean d_eff** | Average across all non-degenerate nodes | 1.5–3.0 | > 5 (too diffuse) or = 1 everywhere (too rigid) |
| **Hub concentration** | Fraction of edges incident to top-5 degree nodes | < 0.40 | > 0.70 (over-centralized) |
| **Isolation rate** | Fraction of nodes with degree < 2 (skipped) | < 0.10 | > 0.30 (sparse graph) |
| **PR distribution** | Histogram of participation ratios across nodes | Bimodal (hubs + bridges) | Flat (no structure) |

These map directly to `FisherSweepResult` fields already computed in Phase A/B.

---

## 5. Two-Tier Report Schema

```
PFD Diagnostic Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RRP Universe:     <name>
DS Wiki version:  <snapshot_id>
Run date:         <ISO timestamp>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 1 — INTERNAL CONSISTENCY
  Entries analyzed:     <n> / <total>  (<pct>% connectable)
  Internal coherence:   <score>  [GOOD | MARGINAL | POOR]
  Mean d_eff:           <value>
  Regime distribution:  <n_radial> radial | <n_iso> isotropic | <n_noise> noise
  Top internal hubs:    <list: entry_id, regime, d_eff>
  Isolation rate:       <pct>%  (<n> entries with degree < 2)
  Verdict:              INTERNALLY CONSISTENT | MARGINAL | FRAGMENTED

TIER 2 — BRIDGE QUALITY (vs DS Wiki)
  Bridge edges used:    <n>  (similarity ≥ <threshold>)
  RRP nodes bridged:    <n> / <total>  (<pct>% reach DS Wiki)
  DS Wiki nodes reached:<n> unique anchors
  Mean bridge d_eff:    <value>  (on G_bridge, RRP nodes only)
  Bridge regime dist:   <n_radial> | <n_iso> | <n_noise>
  Strongest bridges:    <list: rrp_id → wiki_id, similarity, regime>
  Weakest coverage:     <list: rrp_id, no bridge or bridge_sim < 0.75>
  Anchor load:          <list: wiki_id, n_rrp_bridges>  (over-loaded anchors)
  Verdict:              WELL-INTEGRATED | PARTIAL | ISOLATED

OVERALL
  PFD Score:  <Tier1_score * 0.5 + Tier2_score * 0.5>  (0.0 – 1.0)
  Summary:    <1-2 sentence natural language summary>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 6. What Does NOT Change

The following components are correct and stable — no redesign needed:

| Component | Why stable |
|-----------|-----------|
| `decompose_fim()`, `build_fim()`, all math | Universe-agnostic, tested |
| `analyze_node()`, `sweep_graph()` | Work on any nx.Graph regardless of origin |
| `build_wiki_graph(db_path)` | Already reads any SQLite DB with entries+links schema |
| `fisher_bridge_filter.py` | Phase C bridge scoring remains valid as a per-bridge utility; Tier-2 report wraps it |
| All Phase A–C tests | No changes needed |
| DS Wiki self-analysis (`--mode ds_wiki`) | Remains valid — the reference lake testing its own geometry |

---

## 7. Implementation Order

```
Phase D:  build_bridge_graph() + CLI --mode internal_rrp + --mode bridge
Phase E:  CLI tools (pfd node, pfd internal, pfd bridge)
Phase F:  fisher_report.py + --mode report + two-tier output
Phase G:  Documentation update
```

Each phase ends with all tests passing before the next begins.

---

## 8. Design Decisions Recorded

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Bridge graph node prefix | `rrp::` / `wiki::` | Prevents ID collision without schema change |
| Bridge edge weight | `1.0 - similarity` | Maintains distance semantics for Dijkstra |
| Min bridge similarity | 0.75 default | Matches existing tier-1.5 quality floor |
| Bridge graph storage | In-memory nx.Graph only | No new DB table needed; graph is derived from existing tables |
| Report format | Plain text + structured dict | Machine-readable dict for MCP; human-readable text for CLI |
| DS Wiki in G_bridge | Full graph, all edges | Option B: integration into DS Wiki topology, not just point contact |
| PFD Score weighting | 50/50 Tier1/Tier2 | Both internal consistency and bridge quality are equally important |
