"""
fisher_diagnostics.py — Fisher Information Matrix (FIM) diagnostic suite.

Implements DS Wiki entries:
  M6: Fisher Information Rank       — the measurement method
  T1: Fisher Rank Monotonicity      — D_eff non-increasing under coarse-graining
  X0_FIM_Regimes                    — three output regimes (output vocabulary)

Reference: See docs/FISHER_PIPELINE_REDESIGN.md for the 6-step pipeline specification.

Entry points:
    build_wiki_graph(db_path)               → (G, labels)           single-universe graph
    build_bridge_graph(rrp_db, wiki_db)     → (G, node_source_map)  Option B extended graph
    analyze_node(G, node_id, ...)           → FisherResult          single node
    sweep_graph(G, source, ...)             → FisherSweepResult     full graph
    save_sweep_to_db(sweep, ...)            → persists to wiki_history.db
    ensure_fisher_table(path)               → creates DB schema if absent
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

# ── Path bootstrap (mirrors pattern in gap_analyzer.py) ──────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import SOURCE_DB, HISTORY_DB  # noqa: E402

logger = logging.getLogger(__name__)

MAX_SCORE: float = 20.0   # cap for 0/non-0 log-ratio crossings
EPS:       float = 1e-10  # numerical guard for near-zero values


# ── Enumerations ──────────────────────────────────────────────────────────────

class KernelType(str, Enum):
    EXPONENTIAL  = "exponential"    # f(d) = exp(-alpha * d)
    CORRELATION  = "correlation"    # f(u) = cosine_sim(emb_v0, emb_u)
    WEIGHTED_HOP = "weighted_hop"   # product of both


class RegimeType(str, Enum):
    RADIAL_DOMINATED = "radial_dominated"   # State 1: η < 0.35
    ISOTROPIC        = "isotropic"          # State 2: 0.35 ≤ η < 0.65
    NOISE_DOMINATED  = "noise_dominated"    # State 3: η ≥ 0.65
    DEGENERATE       = "degenerate"         # degree < 2, cannot compute FIM


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class FisherResult:
    node_id:       str
    kernel_type:   KernelType
    alpha:         float            # exponential decay (ignored for CORRELATION)
    center_degree: int              # k = degree(v0)
    d_eff:         int              # gap-based effective dimension
    pr:            float            # participation ratio (continuous dimension)
    eta:           float            # disorder index (regime classifier)
    regime:        RegimeType       # one of the three X0 states
    sv_profile:    list             # normalized singular values [1.0, ...]
    raw_sigmas:    list             # unnormalized singular values
    n_vertices:    int              # |V| at computation time
    skipped:       bool = False     # True if degree < 2 or degenerate FIM
    skip_reason:   str  = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["kernel_type"] = self.kernel_type.value
        d["regime"]      = self.regime.value
        return d


@dataclass
class FisherSweepResult:
    graph_source:  str
    kernel_type:   KernelType
    alpha:         float
    n_analyzed:    int = 0
    n_skipped:     int = 0
    results:       dict = field(default_factory=dict)   # {node_id: FisherResult}
    mean_d_eff:    float = 0.0
    median_eta:    float = 0.0
    regime_counts: dict = field(default_factory=dict)

    def top_hubs(self, n: int = 10) -> list:
        """Top n entries by d_eff (then PR for ties)."""
        valid = [r for r in self.results.values() if not r.skipped]
        return sorted(valid, key=lambda r: (r.d_eff, r.pr), reverse=True)[:n]

    def ordered_nodes(self) -> list:
        """All non-skipped results sorted by d_eff desc, eta asc."""
        valid = [r for r in self.results.values() if not r.skipped]
        return sorted(valid, key=lambda r: (-r.d_eff, r.eta))

    def _compute_aggregates(self) -> None:
        valid = [r for r in self.results.values() if not r.skipped]
        skipped = [r for r in self.results.values() if r.skipped]
        self.n_analyzed = len(valid)
        self.n_skipped  = len(skipped)
        if not valid:
            return
        d_effs = [r.d_eff for r in valid]
        etas   = [r.eta   for r in valid]
        self.mean_d_eff = float(np.mean(d_effs))
        self.median_eta = float(np.median(etas))
        counts: dict = {}
        for r in valid:
            counts[r.regime.value] = counts.get(r.regime.value, 0) + 1
        counts["degenerate"] = len(skipped)
        self.regime_counts = counts


# ── Graph Construction ────────────────────────────────────────────────────────

_TIER_WEIGHTS: dict = {"1": 1.0, "1.5": 1.5, "2": 2.0, None: 3.0}


def build_bridge_graph(
    rrp_db: Path,
    wiki_db: Path,
    min_bridge_similarity: float = 0.75,
) -> tuple:
    """
    Build the full Option B extended bridge graph (FISHER_PIPELINE_REDESIGN.md §2).

    Nodes:
        RRP entries  → prefixed "rrp::<id>"
        DS Wiki entries → prefixed "wiki::<id>"

    Edges (stored with 'type' and 'weight' attributes):
        type="rrp"    within-RRP links      weight = tier-based distance
        type="wiki"   within-DS Wiki links  weight = tier-based distance
        type="bridge" cross_universe_bridges weight = 1.0 - similarity
                      (high similarity → short edge → reachable via Dijkstra)

    Bridge edges below min_bridge_similarity are excluded.

    Returns:
        G_bridge     — undirected weighted nx.Graph
        node_source  — {prefixed_node_id: "rrp" | "wiki"}
    """
    rrp_db  = Path(rrp_db)
    wiki_db = Path(wiki_db)
    G: nx.Graph = nx.Graph()
    node_source: dict = {}

    conn_rrp  = sqlite3.connect(rrp_db)
    conn_wiki = sqlite3.connect(wiki_db)
    try:
        # ── Load RRP universe ────────────────────────────────────────────────
        for row in conn_rrp.execute("SELECT id FROM entries"):
            nid = f"rrp::{row[0]}"
            G.add_node(nid)
            node_source[nid] = "rrp"

        for row in conn_rrp.execute(
            "SELECT source_id, target_id, confidence_tier FROM links"
        ):
            src = f"rrp::{row[0]}"
            tgt = f"rrp::{row[1]}"
            tier   = row[2]
            weight = _TIER_WEIGHTS.get(tier, 3.0)
            if G.has_node(src) and G.has_node(tgt):
                if G.has_edge(src, tgt):
                    if G[src][tgt]["weight"] > weight:
                        G[src][tgt]["weight"] = weight
                        G[src][tgt]["type"]   = "rrp"
                else:
                    G.add_edge(src, tgt, weight=weight, type="rrp")

        # ── Load DS Wiki universe ────────────────────────────────────────────
        for row in conn_wiki.execute("SELECT id FROM entries"):
            nid = f"wiki::{row[0]}"
            G.add_node(nid)
            node_source[nid] = "wiki"

        for row in conn_wiki.execute(
            "SELECT source_id, target_id, confidence_tier FROM links"
        ):
            src = f"wiki::{row[0]}"
            tgt = f"wiki::{row[1]}"
            tier   = row[2]
            weight = _TIER_WEIGHTS.get(tier, 3.0)
            if G.has_node(src) and G.has_node(tgt):
                if G.has_edge(src, tgt):
                    if G[src][tgt]["weight"] > weight:
                        G[src][tgt]["weight"] = weight
                        G[src][tgt]["type"]   = "wiki"
                else:
                    G.add_edge(src, tgt, weight=weight, type="wiki")

        # ── Load cross-universe bridges ──────────────────────────────────────
        for row in conn_rrp.execute(
            "SELECT rrp_entry_id, ds_entry_id, similarity "
            "FROM cross_universe_bridges "
            "WHERE similarity >= ?",
            (min_bridge_similarity,),
        ):
            rrp_nid  = f"rrp::{row[0]}"
            wiki_nid = f"wiki::{row[1]}"
            sim      = float(row[2])
            weight   = 1.0 - sim   # distance semantics: closer = more similar
            if G.has_node(rrp_nid) and G.has_node(wiki_nid):
                if not G.has_edge(rrp_nid, wiki_nid):
                    G.add_edge(rrp_nid, wiki_nid, weight=weight, type="bridge")
                elif G[rrp_nid][wiki_nid]["weight"] > weight:
                    G[rrp_nid][wiki_nid]["weight"] = weight

    finally:
        conn_rrp.close()
        conn_wiki.close()

    return G, node_source


def build_wiki_graph(db_path: Path) -> tuple:
    """
    Build undirected NetworkX graph from DS Wiki or RRP bundle SQLite DB.

    Nodes = entry IDs (strings). Edges = links with tier-based weights.
    Tier encoding: '1' → 1.0, '1.5' → 1.5, '2' → 2.0, None → 3.0.
    Multiple edges between the same pair keep the minimum weight.

    Returns: (G, labels) where labels = {entry_id: title}.
    """
    db_path = Path(db_path)
    conn    = sqlite3.connect(db_path)
    G       = nx.Graph()
    labels: dict = {}

    for row in conn.execute("SELECT id, title FROM entries"):
        node_id = str(row[0])
        G.add_node(node_id)
        labels[node_id] = row[1] or node_id

    for row in conn.execute(
        "SELECT source_id, target_id, confidence_tier FROM links"
    ):
        src, tgt = str(row[0]), str(row[1])
        tier     = row[2]
        weight   = _TIER_WEIGHTS.get(tier, 3.0)
        if G.has_node(src) and G.has_node(tgt):
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] = min(G[src][tgt]["weight"], weight)
            else:
                G.add_edge(src, tgt, weight=weight)

    conn.close()
    return G, labels


def build_distance_matrix(G: nx.Graph, source) -> dict:
    """
    Dijkstra shortest-path distances from source to all reachable nodes.
    Uses edge 'weight' attribute; falls back to hop count (weight=1).
    Unreachable nodes are absent from the returned dict.
    """
    try:
        lengths = nx.single_source_dijkstra_path_length(G, source, weight="weight")
        return dict(lengths)
    except nx.NodeNotFound:
        return {}


# ── Kernel Functions ──────────────────────────────────────────────────────────

def exponential_kernel(distances: dict, alpha: float = 1.0) -> dict:
    """f(d) = exp(-alpha * d). Returns {node: kernel_value}."""
    return {node: float(np.exp(-alpha * float(d))) for node, d in distances.items()}


def correlation_kernel(
    center_emb:     np.ndarray,
    all_embeddings: dict,
) -> dict:
    """
    f(u) = max(0, cosine_sim(center_emb, emb_u)).
    Returns {node_id: kernel_value}.
    """
    center_norm = float(np.linalg.norm(center_emb))
    if center_norm < EPS:
        return {nid: 0.0 for nid in all_embeddings}
    c = center_emb / center_norm
    result = {}
    for nid, emb in all_embeddings.items():
        n = float(np.linalg.norm(emb))
        if n < EPS:
            result[nid] = 0.0
        else:
            result[nid] = float(max(0.0, float(np.dot(c, emb / n))))
    return result


def weighted_hop_kernel(
    distances:      dict,
    center_emb:     np.ndarray,
    all_embeddings: dict,
    alpha:          float = 1.0,
) -> dict:
    """
    f(d, u) = max(0, cosine_sim(center_emb, emb_u)) * exp(-alpha * d).
    Combines structural position with semantic similarity.
    """
    corr = correlation_kernel(center_emb, all_embeddings)
    exp_vals = exponential_kernel(distances, alpha)
    return {
        nid: corr.get(nid, 0.0) * exp_vals.get(nid, 0.0)
        for nid in all_embeddings
    }


# ── Distribution and Score Vectors ───────────────────────────────────────────

def build_distribution(kernel_values: dict, vertex_ids: list) -> np.ndarray:
    """
    p(u) = kernel_value(u) / sum_u kernel_value(u).

    Returns np.ndarray of length |vertex_ids|. Nodes absent from
    kernel_values get value 0. Zero denominator → uniform distribution.
    """
    vals  = np.array([kernel_values.get(v, 0.0) for v in vertex_ids], dtype=float)
    total = vals.sum()
    if total < EPS:
        return np.full(len(vertex_ids), 1.0 / max(len(vertex_ids), 1))
    return vals / total


def build_score_vectors(
    neighbor_kvs: list,
    center_kv:    dict,
    vertex_ids:   list,
) -> np.ndarray:
    """
    Build k × |V| score matrix.  S[j, u] = log(p_wj(u)) - log(p_v0(u)).

    Special cases (Step 4 of spec):
        p_v0(u) = 0 AND p_wj(u) = 0  →  S[j,u] = 0
        p_v0(u) = 0 AND p_wj(u) > 0  →  S[j,u] = +MAX_SCORE
        p_v0(u) > 0 AND p_wj(u) = 0  →  S[j,u] = -MAX_SCORE
    """
    p_center = build_distribution(center_kv, vertex_ids)
    k = len(neighbor_kvs)
    n = len(vertex_ids)
    S = np.zeros((k, n), dtype=float)

    for j, nb_kv in enumerate(neighbor_kvs):
        p_nb = build_distribution(nb_kv, vertex_ids)
        pc_zero = p_center < EPS
        pn_zero = p_nb     < EPS

        # Both zero → 0 (already set)
        # Center zero, neighbor nonzero → +MAX_SCORE
        S[j, pc_zero & ~pn_zero]  =  MAX_SCORE
        # Center nonzero, neighbor zero → -MAX_SCORE
        S[j, ~pc_zero & pn_zero]  = -MAX_SCORE
        # Both nonzero → log ratio
        both = ~pc_zero & ~pn_zero
        S[j, both] = np.log(p_nb[both]) - np.log(p_center[both])

    return S


# ── FIM Construction ──────────────────────────────────────────────────────────

def build_fim(score_matrix: np.ndarray, center_distribution: np.ndarray) -> np.ndarray:
    """
    F[i,j] = sum_u S[i,u] * S[j,u] * p_v0(u)

    Vectorized: F = (S * p_v0) @ S.T
    Returns k×k real symmetric matrix.
    """
    weighted = score_matrix * center_distribution[np.newaxis, :]   # (k, n)
    return weighted @ score_matrix.T                                 # (k, k)


# ── SVD Decomposition ─────────────────────────────────────────────────────────

def decompose_fim(F: np.ndarray) -> tuple:
    """
    SVD of F → (d_eff, pr, sv_profile, eta).

    d_eff      : position of largest σᵢ/σᵢ₊₁ gap + 1 (1-indexed)
    pr         : (Σσ)² / Σσ²  participation ratio
    sv_profile : σ / σ[0]     normalized, first = 1.0
    eta        : σ[d_eff] / σ[d_eff-1]  disorder index
    """
    sigma_raw = np.linalg.svd(F, compute_uv=False)

    # Trim near-zero singular values (relative threshold)
    threshold = EPS * (sigma_raw[0] if sigma_raw[0] > EPS else 1.0)
    sigma = sigma_raw[sigma_raw > threshold]

    if len(sigma) <= 1:
        sv = sigma.tolist() if len(sigma) == 1 else [1.0]
        return 1, 1.0, [v / sv[0] for v in sv], 0.0

    # Largest gap in consecutive ratio spectrum
    ratios = sigma[:-1] / (sigma[1:] + EPS)
    d_eff  = int(np.argmax(ratios)) + 1          # 1-indexed

    # Participation ratio
    pr = float(np.sum(sigma) ** 2 / np.sum(sigma ** 2))

    # Normalized SV profile
    sv_profile = (sigma / sigma[0]).tolist()

    # Disorder index: ratio right after the gap
    eta = float(sigma[d_eff] / sigma[d_eff - 1]) if d_eff < len(sigma) else 0.0

    return d_eff, pr, sv_profile, eta


# ── Regime Classification ─────────────────────────────────────────────────────

def classify_regime(eta: float) -> RegimeType:
    """
    Map disorder index η to X0_FIM_Regimes state.
    Thresholds: RADIAL < 0.35 ≤ ISOTROPIC < 0.65 ≤ NOISE_DOMINATED.
    """
    if eta < 0.35:
        return RegimeType.RADIAL_DOMINATED
    elif eta < 0.65:
        return RegimeType.ISOTROPIC
    else:
        return RegimeType.NOISE_DOMINATED


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _degenerate(
    node_id: str, kernel_type: KernelType, alpha: float,
    degree: int, n_vertices: int, reason: str,
) -> FisherResult:
    return FisherResult(
        node_id=node_id, kernel_type=kernel_type, alpha=alpha,
        center_degree=degree, d_eff=0, pr=0.0, eta=0.0,
        regime=RegimeType.DEGENERATE, sv_profile=[], raw_sigmas=[],
        n_vertices=n_vertices, skipped=True, skip_reason=reason,
    )


def _resolve_node(G: nx.Graph, node_id: str):
    """
    Flexible node lookup: try string first, then int conversion.
    Returns the actual node key in G, or None if not found.
    """
    if node_id in G:
        return node_id
    try:
        int_id = int(node_id)
        if int_id in G:
            return int_id
    except (ValueError, TypeError):
        pass
    return None


# ── Primary Entry Points ──────────────────────────────────────────────────────

def analyze_node(
    G:           nx.Graph,
    node_id:     str,
    kernel_type: KernelType = KernelType.EXPONENTIAL,
    alpha:       float = 1.0,
    embeddings:  Optional[dict] = None,
) -> FisherResult:
    """
    Full FIM pipeline for a single center node.

    Preconditions:
        node_id must be in G (string form; int lookup attempted as fallback)
        degree(node_id) ≥ 2  (else returns skipped=True result)
        CORRELATION / WEIGHTED_HOP kernels require embeddings dict

    Steps: neighbors → distance matrices → kernel → distributions →
           score matrix → FIM → SVD → regime classification → FisherResult
    """
    actual_node = _resolve_node(G, node_id)
    if actual_node is None:
        return _degenerate(node_id, kernel_type, alpha, 0,
                           G.number_of_nodes(), f"node '{node_id}' not in graph")

    neighbors  = list(G.neighbors(actual_node))
    k          = len(neighbors)
    n_vertices = G.number_of_nodes()

    if k < 2:
        return _degenerate(node_id, kernel_type, alpha, k,
                           n_vertices, f"degree={k} < 2")

    # Canonical string vertex ordering for all numpy arrays
    vertex_ids = [str(v) for v in G.nodes()]

    # Distance matrices (str-keyed)
    center_dist_raw = build_distance_matrix(G, actual_node)
    center_dist     = {str(nd): v for nd, v in center_dist_raw.items()}

    neighbor_dists = []
    for nb in neighbors:
        raw = build_distance_matrix(G, nb)
        neighbor_dists.append({str(nd): v for nd, v in raw.items()})

    # Apply selected kernel
    if kernel_type == KernelType.EXPONENTIAL:
        center_kv = exponential_kernel(center_dist, alpha)
        nb_kvs    = [exponential_kernel(nd, alpha) for nd in neighbor_dists]

    elif kernel_type == KernelType.CORRELATION:
        if embeddings is None:
            return _degenerate(node_id, kernel_type, alpha, k, n_vertices,
                               "embeddings required for CORRELATION kernel")
        c_emb = embeddings.get(str(actual_node))
        if c_emb is None:
            return _degenerate(node_id, kernel_type, alpha, k, n_vertices,
                               f"no embedding for '{node_id}'")
        center_kv = correlation_kernel(c_emb, embeddings)
        nb_kvs    = []
        for nb in neighbors:
            nb_emb = embeddings.get(str(nb))
            if nb_emb is None:
                return _degenerate(node_id, kernel_type, alpha, k, n_vertices,
                                   f"no embedding for neighbor '{nb}'")
            nb_kvs.append(correlation_kernel(nb_emb, embeddings))

    elif kernel_type == KernelType.WEIGHTED_HOP:
        if embeddings is None:
            return _degenerate(node_id, kernel_type, alpha, k, n_vertices,
                               "embeddings required for WEIGHTED_HOP kernel")
        c_emb = embeddings.get(str(actual_node))
        if c_emb is None:
            return _degenerate(node_id, kernel_type, alpha, k, n_vertices,
                               f"no embedding for '{node_id}'")
        center_kv = weighted_hop_kernel(center_dist, c_emb, embeddings, alpha)
        nb_kvs    = []
        for nb, nd in zip(neighbors, neighbor_dists):
            nb_emb = embeddings.get(str(nb))
            if nb_emb is None:
                return _degenerate(node_id, kernel_type, alpha, k, n_vertices,
                                   f"no embedding for neighbor '{nb}'")
            nb_kvs.append(weighted_hop_kernel(nd, nb_emb, embeddings, alpha))

    else:
        return _degenerate(node_id, kernel_type, alpha, k, n_vertices,
                           f"unknown kernel: {kernel_type}")

    # Build distributions, score matrix, FIM, SVD
    p_center   = build_distribution(center_kv, vertex_ids)
    S          = build_score_vectors(nb_kvs, center_kv, vertex_ids)
    F          = build_fim(S, p_center)
    d_eff, pr, sv_profile, eta = decompose_fim(F)
    regime     = classify_regime(eta)
    raw_sigmas = np.linalg.svd(F, compute_uv=False).tolist()

    return FisherResult(
        node_id=node_id, kernel_type=kernel_type, alpha=alpha,
        center_degree=k, d_eff=d_eff, pr=pr, eta=eta,
        regime=regime, sv_profile=sv_profile, raw_sigmas=raw_sigmas,
        n_vertices=n_vertices, skipped=False,
    )


def sweep_graph(
    G:            nx.Graph,
    graph_source: str,
    kernel_type:  KernelType = KernelType.EXPONENTIAL,
    alpha:        float = 1.0,
    embeddings:   Optional[dict] = None,
    min_degree:   int = 2,
) -> FisherSweepResult:
    """
    Run analyze_node for every node in G with degree ≥ min_degree.
    Nodes below min_degree are recorded as skipped, not omitted.
    Populates FisherSweepResult with per-node results and aggregates.
    """
    sweep = FisherSweepResult(
        graph_source=graph_source,
        kernel_type=kernel_type,
        alpha=alpha,
    )

    for node in G.nodes():
        node_str = str(node)
        if G.degree(node) < min_degree:
            sweep.results[node_str] = _degenerate(
                node_str, kernel_type, alpha,
                G.degree(node), G.number_of_nodes(),
                f"degree={G.degree(node)} < min_degree={min_degree}",
            )
        else:
            sweep.results[node_str] = analyze_node(
                G, node_str, kernel_type, alpha, embeddings
            )

    sweep._compute_aggregates()
    return sweep


# ── Embedding Loader (stub — Phase B implementation) ─────────────────────────

def load_embeddings_from_chroma(
    collection_name: str,
    chroma_dir:      Path,
) -> dict:
    """
    Load all embedding vectors from ChromaDB, mean-pooled to entry level.
    Returns {entry_id: np.ndarray}.

    Phase A stub — raises NotImplementedError.
    Full implementation (Phase B) will mean-pool chunk embeddings per entry.
    """
    raise NotImplementedError(
        "load_embeddings_from_chroma is a Phase B implementation. "
        "Use KernelType.EXPONENTIAL for Phase A testing."
    )


# ── DB Persistence ────────────────────────────────────────────────────────────

_CREATE_FISHER_SQL = """
CREATE TABLE IF NOT EXISTS fisher_metrics (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id           TEXT,
    entry_id              TEXT NOT NULL,
    graph_source          TEXT NOT NULL,
    kernel_type           TEXT NOT NULL,
    alpha                 REAL,
    d_eff                 INTEGER,
    participation_ratio   REAL,
    disorder_index        REAL,
    regime                TEXT,
    sv_profile_json       TEXT,
    raw_sigmas_json       TEXT,
    center_degree         INTEGER,
    n_vertices            INTEGER,
    skipped               INTEGER DEFAULT 0,
    skip_reason           TEXT,
    computed_at           TEXT DEFAULT (datetime('now')),
    UNIQUE(entry_id, graph_source, kernel_type, alpha)
);
CREATE INDEX IF NOT EXISTS idx_fisher_entry  ON fisher_metrics(entry_id);
CREATE INDEX IF NOT EXISTS idx_fisher_regime ON fisher_metrics(regime);
CREATE INDEX IF NOT EXISTS idx_fisher_d_eff  ON fisher_metrics(d_eff DESC);
"""


def ensure_fisher_table(history_db: Optional[Path] = None) -> None:
    """Create fisher_metrics table in wiki_history.db if not present."""
    path = Path(history_db) if history_db else HISTORY_DB
    conn = sqlite3.connect(path)
    conn.executescript(_CREATE_FISHER_SQL)
    conn.commit()
    conn.close()


def save_sweep_to_db(
    sweep:       FisherSweepResult,
    history_db:  Optional[Path] = None,
    snapshot_id: Optional[str]  = None,
) -> int:
    """
    Persist all FisherResult rows to fisher_metrics table.
    Uses INSERT OR REPLACE — safe to re-run.
    Returns number of rows written.
    """
    path = Path(history_db) if history_db else HISTORY_DB
    ensure_fisher_table(path)
    conn    = sqlite3.connect(path)
    written = 0

    for result in sweep.results.values():
        conn.execute(
            """
            INSERT OR REPLACE INTO fisher_metrics
                (snapshot_id, entry_id, graph_source, kernel_type, alpha,
                 d_eff, participation_ratio, disorder_index, regime,
                 sv_profile_json, raw_sigmas_json, center_degree, n_vertices,
                 skipped, skip_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                snapshot_id,
                result.node_id,
                sweep.graph_source,
                result.kernel_type.value,
                result.alpha,
                result.d_eff   if not result.skipped else None,
                result.pr      if not result.skipped else None,
                result.eta     if not result.skipped else None,
                result.regime.value,
                json.dumps(result.sv_profile),
                json.dumps(result.raw_sigmas),
                result.center_degree,
                result.n_vertices,
                1 if result.skipped else 0,
                result.skip_reason or None,
            ),
        )
        written += 1

    conn.commit()
    conn.close()
    return written
