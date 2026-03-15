"""
embedder.py — Embed chunks into ChromaDB and record a snapshot in wiki_history.db.
Every call is additive. Nothing is ever deleted from wiki_history.db.
"""
import json
import sqlite3
from datetime import datetime, timezone
from hashlib import sha256

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION, CHROMA_DIR, DEVICE, EMBED_MODEL, EMBED_DIM,
    HISTORY_DB, TOP_K_NEIGHBORS, DRIFT_THRESHOLD,
)
from extractor import Chunk


# ── History DB bootstrap ──────────────────────────────────────────────────────

def _init_history_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS wiki_snapshots (
            snapshot_id   TEXT PRIMARY KEY,
            created_at    TEXT NOT NULL,
            trigger       TEXT NOT NULL,
            chunk_count   INTEGER NOT NULL,
            notes         TEXT,
            embed_model   TEXT
        );

        CREATE TABLE IF NOT EXISTS chunk_embedding_history (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id       TEXT NOT NULL REFERENCES wiki_snapshots(snapshot_id),
            chunk_id          TEXT NOT NULL,
            entry_id          TEXT NOT NULL,
            content_hash      TEXT NOT NULL,
            embedding         BLOB NOT NULL,
            top5_neighbors    TEXT NOT NULL,
            centroid_distance REAL NOT NULL,
            UNIQUE(snapshot_id, chunk_id)
        );

        CREATE TABLE IF NOT EXISTS topology_metrics (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id   TEXT NOT NULL REFERENCES wiki_snapshots(snapshot_id),
            metric_name   TEXT NOT NULL,
            metric_value  TEXT NOT NULL,
            UNIQUE(snapshot_id, metric_name)
        );

        CREATE INDEX IF NOT EXISTS idx_ceh_entry    ON chunk_embedding_history(entry_id);
        CREATE INDEX IF NOT EXISTS idx_ceh_snapshot ON chunk_embedding_history(snapshot_id);
        CREATE INDEX IF NOT EXISTS idx_ceh_chunk    ON chunk_embedding_history(chunk_id);
    """)
    conn.commit()


# ── Previous snapshot loader ──────────────────────────────────────────────────

def _load_previous_snapshot(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    """Return {chunk_id: embedding_array} for the most recent snapshot.

    Returns an empty dict if no prior snapshot exists or if the prior snapshot
    used a different embedding model — cross-model drift comparison is undefined.
    """
    row = conn.execute(
        "SELECT snapshot_id, embed_model FROM wiki_snapshots ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return {}
    snap_id, prev_model = row[0], row[1]
    if prev_model != EMBED_MODEL:
        print(f"  Model changed ({prev_model} → {EMBED_MODEL}): skipping drift comparison")
        return {}
    rows = conn.execute(
        "SELECT chunk_id, embedding FROM chunk_embedding_history WHERE snapshot_id = ?",
        (snap_id,),
    ).fetchall()
    return {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in rows}


# ── Topology metrics ──────────────────────────────────────────────────────────

def _compute_topology(
    chunk_ids: list[str],
    embeddings: np.ndarray,          # (N, 384) normalised
    previous: dict[str, np.ndarray],
) -> tuple[list[dict], dict]:
    """
    Returns:
      per_chunk  — list of dicts with top5_neighbors + centroid_distance per chunk
      metrics    — dict of scalar/aggregate topology metrics
    """
    N = len(chunk_ids)
    id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

    # Centroid (normalise after mean)
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid /= norm

    # Pairwise cosine sims (dot product — embeddings are already unit-normed)
    sim_matrix = embeddings @ embeddings.T  # (N, N)

    per_chunk: list[dict] = []
    drifts: list[float] = []
    new_chunks: list[str] = []
    changed_chunks: list[str] = []
    neighbor_instability: dict[str, float] = {}

    for i, cid in enumerate(chunk_ids):
        # centroid distance (cosine)
        c_dist = float(1.0 - float(embeddings[i] @ centroid))

        # top-5 neighbours (exclude self)
        sims = sim_matrix[i].copy()
        sims[i] = -2.0
        top_idx = np.argsort(sims)[-TOP_K_NEIGHBORS:][::-1]
        top5 = [{"id": chunk_ids[j], "score": round(float(sims[j]), 4)} for j in top_idx]

        per_chunk.append({
            "chunk_id":         cid,
            "top5_neighbors":   top5,
            "centroid_distance": round(c_dist, 4),
        })

        # drift vs previous snapshot
        if cid in previous:
            prev_emb = previous[cid]
            drift = float(1.0 - float(embeddings[i] @ prev_emb))
            drifts.append((cid, drift))
            if drift > DRIFT_THRESHOLD:
                changed_chunks.append(cid)

            # neighbourhood instability (Jaccard on top-5 id sets)
            prev_top5_set = set()          # we don't store previous top5 in this pass
            # (computed in topology.py from history — skip here for efficiency)
        else:
            new_chunks.append(cid)

    # Converging pairs (pairs that moved closer since last snapshot)
    converging: list[dict] = []
    if previous:
        prev_ids = [cid for cid in chunk_ids if cid in previous]
        if len(prev_ids) >= 2:
            prev_matrix = np.stack([previous[cid] for cid in prev_ids])
            prev_sim    = prev_matrix @ prev_matrix.T
            curr_indices = [id_to_idx[cid] for cid in prev_ids]
            curr_sim     = sim_matrix[np.ix_(curr_indices, curr_indices)]
            delta        = curr_sim - prev_sim
            # upper triangle only, exclude diagonal
            triu = np.triu(delta, k=1)
            flat_idx = np.argsort(triu.ravel())[-5:][::-1]
            for fi in flat_idx:
                r, c = divmod(int(fi), len(prev_ids))
                d = float(triu[r, c])
                if d > 0:
                    converging.append({
                        "a": prev_ids[r], "b": prev_ids[c],
                        "delta": round(d, 4)
                    })

    centroid_distances = [p["centroid_distance"] for p in per_chunk]

    sorted_drifts = sorted(drifts, key=lambda x: x[1], reverse=True)
    max_drift = {"chunk_id": sorted_drifts[0][0], "drift": round(sorted_drifts[0][1], 4)} \
        if sorted_drifts else {}

    metrics = {
        "corpus_centroid":       centroid.tolist(),
        "chunk_count":           N,
        "mean_centroid_distance": round(float(np.mean(centroid_distances)), 4),
        "max_drift_chunk":       max_drift,
        "mean_drift":            round(float(np.mean([d for _, d in drifts])), 4) if drifts else 0.0,
        "new_chunks":            new_chunks,
        "changed_chunks":        changed_chunks,
        "converging_pairs":      converging,
        "isolated_chunks":       _find_isolated(chunk_ids, centroid_distances),
    }
    return per_chunk, metrics


def _find_isolated(chunk_ids: list[str], distances: list[float]) -> list[str]:
    arr = np.array(distances)
    threshold = arr.mean() + 2 * arr.std()
    return [cid for cid, d in zip(chunk_ids, distances) if d > threshold]


# ── Main entry point ──────────────────────────────────────────────────────────

def embed_and_store(
    chunks: list[Chunk],
    trigger: str = "manual",
    notes: str = "",
) -> str:
    """Embed all chunks, update ChromaDB, write snapshot to wiki_history.db.
    Returns snapshot_id."""

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c.embed_text for c in chunks]
    ids   = [c.chunk_id   for c in chunks]

    print(f"Embedding {len(chunks)} chunks…")
    embeddings_raw = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=DEVICE,         # Auto-detected: cuda/mps/cpu
    )
    embeddings = np.array(embeddings_raw, dtype=np.float32)  # (N, EMBED_DIM)

    # ── ChromaDB (full rebuild) ───────────────────────────────────────────────
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[c.metadata for c in chunks],
    )
    print(f"ChromaDB updated: {len(chunks)} vectors in collection '{CHROMA_COLLECTION}'")

    # ── History DB ────────────────────────────────────────────────────────────
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    hconn = sqlite3.connect(HISTORY_DB)
    _init_history_db(hconn)

    previous = _load_previous_snapshot(hconn)
    print(f"Previous snapshot: {len(previous)} chunks loaded for drift comparison")

    per_chunk, metrics = _compute_topology(ids, embeddings, previous)

    snapshot_id = f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    created_at  = datetime.now(timezone.utc).isoformat()

    hconn.execute(
        "INSERT INTO wiki_snapshots VALUES (?,?,?,?,?,?)",
        (snapshot_id, created_at, trigger, len(chunks), notes or None, EMBED_MODEL),
    )

    # Build content_hash for each chunk
    hash_map = {c.chunk_id: sha256(c.embed_text.encode()).hexdigest() for c in chunks}

    rows = [
        (
            snapshot_id,
            p["chunk_id"],
            next(c.entry_id for c in chunks if c.chunk_id == p["chunk_id"]),
            hash_map[p["chunk_id"]],
            embeddings[ids.index(p["chunk_id"])].tobytes(),
            json.dumps(p["top5_neighbors"]),
            p["centroid_distance"],
        )
        for p in per_chunk
    ]
    hconn.executemany(
        "INSERT OR IGNORE INTO chunk_embedding_history "
        "(snapshot_id, chunk_id, entry_id, content_hash, embedding, top5_neighbors, centroid_distance) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )

    # Store topology metrics (skip the large centroid vector to keep DB lean)
    metrics_to_store = {k: v for k, v in metrics.items() if k != "corpus_centroid"}
    for name, value in metrics_to_store.items():
        hconn.execute(
            "INSERT OR IGNORE INTO topology_metrics (snapshot_id, metric_name, metric_value) VALUES (?,?,?)",
            (snapshot_id, name, json.dumps(value)),
        )

    hconn.commit()
    hconn.close()

    print(f"\nSnapshot recorded: {snapshot_id}")
    print(f"  new chunks    : {len(metrics['new_chunks'])}")
    print(f"  changed chunks: {len(metrics['changed_chunks'])}")
    print(f"  mean drift    : {metrics['mean_drift']}")
    print(f"  isolated      : {metrics['isolated_chunks']}")

    return snapshot_id
