"""
topology.py — Read-only query functions over wiki_history.db.
Used by mcp_server.py to answer questions about semantic evolution.
"""
import json
import sqlite3

import numpy as np

from config import HISTORY_DB, TOP_K_NEIGHBORS


def _conn() -> sqlite3.Connection:
    if not HISTORY_DB.exists():
        raise FileNotFoundError(
            f"History DB not found at {HISTORY_DB}. Run sync.py first."
        )
    c = sqlite3.connect(HISTORY_DB)
    c.row_factory = sqlite3.Row
    return c


def _latest_snapshot_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT snapshot_id FROM wiki_snapshots ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return row["snapshot_id"] if row else None


# ── Public API (called by mcp_server.py) ─────────────────────────────────────

def list_snapshots() -> list[dict]:
    """All snapshots, newest first."""
    conn = _conn()
    rows = conn.execute(
        "SELECT snapshot_id, created_at, trigger, chunk_count, notes "
        "FROM wiki_snapshots ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_drift_report(snapshot_id: str | None = None) -> dict:
    """
    Semantic change report between the given snapshot and the one before it.
    Defaults to the latest snapshot vs its predecessor.
    """
    conn = _conn()
    snaps = conn.execute(
        "SELECT snapshot_id, created_at, trigger FROM wiki_snapshots ORDER BY created_at DESC"
    ).fetchall()

    if not snaps:
        conn.close()
        return {"error": "No snapshots found. Run sync first."}

    if snapshot_id is None:
        target = snaps[0]
        prev   = snaps[1] if len(snaps) > 1 else None
    else:
        target = next((s for s in snaps if s["snapshot_id"] == snapshot_id), None)
        if target is None:
            conn.close()
            return {"error": f"Snapshot {snapshot_id} not found."}
        idx  = list(snaps).index(target)
        prev = snaps[idx + 1] if idx + 1 < len(snaps) else None

    snap_id = target["snapshot_id"]

    # Fetch topology metrics stored at embed time
    metrics_rows = conn.execute(
        "SELECT metric_name, metric_value FROM topology_metrics WHERE snapshot_id = ?",
        (snap_id,),
    ).fetchall()
    metrics = {r["metric_name"]: json.loads(r["metric_value"]) for r in metrics_rows}

    # Top-drifted chunks: need to compare embeddings with previous snapshot
    top_drifted = []
    if prev:
        prev_id = prev["snapshot_id"]
        curr_rows = conn.execute(
            "SELECT chunk_id, embedding FROM chunk_embedding_history WHERE snapshot_id=?",
            (snap_id,),
        ).fetchall()
        prev_map = {
            r["chunk_id"]: np.frombuffer(r["embedding"], dtype=np.float32)
            for r in conn.execute(
                "SELECT chunk_id, embedding FROM chunk_embedding_history WHERE snapshot_id=?",
                (prev_id,),
            ).fetchall()
        }
        drifts = []
        for r in curr_rows:
            cid = r["chunk_id"]
            if cid in prev_map:
                curr_emb = np.frombuffer(r["embedding"], dtype=np.float32)
                drift = float(1.0 - float(curr_emb @ prev_map[cid]))
                drifts.append({"chunk_id": cid, "drift": round(drift, 4)})
        drifts.sort(key=lambda x: x["drift"], reverse=True)
        top_drifted = drifts[:10]

    conn.close()
    return {
        "snapshot_id":    snap_id,
        "created_at":     target["created_at"],
        "trigger":        target["trigger"],
        "previous_snap":  prev["snapshot_id"] if prev else None,
        "mean_drift":     metrics.get("mean_drift", 0),
        "top_drifted":    top_drifted,
        "new_chunks":     metrics.get("new_chunks", []),
        "changed_chunks": metrics.get("changed_chunks", []),
        "converging_pairs": metrics.get("converging_pairs", []),
        "isolated_chunks": metrics.get("isolated_chunks", []),
    }


def get_entry_trajectory(entry_id: str) -> list[dict]:
    """
    For a given entry, return its chunks' semantic positions across all snapshots.
    Shows how the entry has moved through semantic space over time.
    """
    conn = _conn()
    snaps = conn.execute(
        "SELECT snapshot_id, created_at, trigger FROM wiki_snapshots ORDER BY created_at ASC"
    ).fetchall()

    trajectory = []
    prev_embeddings: dict[str, np.ndarray] = {}

    for snap in snaps:
        snap_id = snap["snapshot_id"]
        rows = conn.execute(
            "SELECT chunk_id, embedding, top5_neighbors, centroid_distance "
            "FROM chunk_embedding_history "
            "WHERE snapshot_id=? AND entry_id=?",
            (snap_id, entry_id),
        ).fetchall()

        if not rows:
            continue

        chunks_out = []
        for r in rows:
            cid      = r["chunk_id"]
            curr_emb = np.frombuffer(r["embedding"], dtype=np.float32)
            drift    = None
            if cid in prev_embeddings:
                drift = round(float(1.0 - float(curr_emb @ prev_embeddings[cid])), 4)
            prev_embeddings[cid] = curr_emb

            chunks_out.append({
                "chunk_id":         cid,
                "centroid_distance": r["centroid_distance"],
                "top5_neighbors":   json.loads(r["top5_neighbors"]),
                "drift_from_previous": drift,
            })

        trajectory.append({
            "snapshot_id": snap_id,
            "created_at":  snap["created_at"],
            "trigger":     snap["trigger"],
            "chunks":      chunks_out,
        })

    conn.close()
    return trajectory


def get_neighborhood_history(chunk_id: str) -> list[dict]:
    """
    How the top-5 nearest neighbours of a specific chunk have changed
    across all snapshots.
    """
    conn = _conn()
    rows = conn.execute(
        "SELECT s.snapshot_id, s.created_at, s.trigger, h.top5_neighbors "
        "FROM chunk_embedding_history h "
        "JOIN wiki_snapshots s ON s.snapshot_id = h.snapshot_id "
        "WHERE h.chunk_id = ? "
        "ORDER BY s.created_at ASC",
        (chunk_id,),
    ).fetchall()
    conn.close()

    result = []
    prev_neighbors: set[str] = set()
    for r in rows:
        neighbors = json.loads(r["top5_neighbors"])
        curr_set  = {n["id"] for n in neighbors}
        jaccard   = None
        if prev_neighbors:
            union        = prev_neighbors | curr_set
            intersection = prev_neighbors & curr_set
            jaccard      = round(1.0 - len(intersection) / len(union), 4) if union else 0.0
        prev_neighbors = curr_set
        result.append({
            "snapshot_id": r["snapshot_id"],
            "created_at":  r["created_at"],
            "trigger":     r["trigger"],
            "top5_neighbors": neighbors,
            "neighborhood_change_jaccard": jaccard,
        })
    return result


def get_isolated_chunks(snapshot_id: str | None = None) -> list[dict]:
    """Chunks that sit far from the corpus centroid (semantic outliers)."""
    conn = _conn()
    snap_id = snapshot_id or _latest_snapshot_id(conn)
    if not snap_id:
        conn.close()
        return []

    rows = conn.execute(
        "SELECT chunk_id, entry_id, centroid_distance "
        "FROM chunk_embedding_history WHERE snapshot_id=? "
        "ORDER BY centroid_distance DESC",
        (snap_id,),
    ).fetchall()
    conn.close()

    distances = [r["centroid_distance"] for r in rows]
    if not distances:
        return []
    arr       = np.array(distances)
    threshold = arr.mean() + 2 * arr.std()

    return [
        {
            "chunk_id":         r["chunk_id"],
            "entry_id":         r["entry_id"],
            "centroid_distance": r["centroid_distance"],
            "z_score": round(
                float((r["centroid_distance"] - arr.mean()) / arr.std()), 2
            ) if arr.std() > 0 else 0.0,
        }
        for r in rows
        if r["centroid_distance"] > threshold
    ]


def compare_snapshots(snap_a: str, snap_b: str) -> dict:
    """Full diff between two snapshots."""
    conn = _conn()

    def _load(sid: str) -> dict[str, dict]:
        rows = conn.execute(
            "SELECT chunk_id, content_hash, centroid_distance, embedding "
            "FROM chunk_embedding_history WHERE snapshot_id=?",
            (sid,),
        ).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    a = _load(snap_a)
    b = _load(snap_b)
    conn.close()

    added   = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = [
        {
            "chunk_id": cid,
            "drift": round(
                float(
                    1.0 - float(
                        np.frombuffer(b[cid]["embedding"], dtype=np.float32)
                        @ np.frombuffer(a[cid]["embedding"], dtype=np.float32)
                    )
                ), 4
            ),
        }
        for cid in set(a) & set(b)
        if a[cid]["content_hash"] != b[cid]["content_hash"]
    ]
    changed.sort(key=lambda x: x["drift"], reverse=True)

    return {
        "snap_a":  snap_a,
        "snap_b":  snap_b,
        "added":   added,
        "removed": removed,
        "changed": changed,
        "unchanged_count": len(set(a) & set(b)) - len(changed),
    }


def get_cluster_evolution() -> list[dict]:
    """
    Per snapshot: top-10 most similar chunk pairs.
    Reveals stable clusters vs shifting relationships over time.
    """
    conn = _conn()
    snaps = conn.execute(
        "SELECT snapshot_id, created_at FROM wiki_snapshots ORDER BY created_at ASC"
    ).fetchall()

    result = []
    for snap in snaps:
        snap_id = snap["snapshot_id"]
        rows = conn.execute(
            "SELECT chunk_id, embedding FROM chunk_embedding_history WHERE snapshot_id=?",
            (snap_id,),
        ).fetchall()
        if len(rows) < 2:
            continue

        ids  = [r["chunk_id"] for r in rows]
        embs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        sim  = embs @ embs.T
        np.fill_diagonal(sim, -1)

        flat   = sim.ravel()
        top_fi = np.argsort(flat)[-10:][::-1]
        pairs  = []
        seen   = set()
        for fi in top_fi:
            r, c = divmod(int(fi), len(ids))
            key  = tuple(sorted([r, c]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "a": ids[r], "b": ids[c],
                "similarity": round(float(sim[r, c]), 4)
            })

        result.append({
            "snapshot_id": snap_id,
            "created_at":  snap["created_at"],
            "top_pairs":   pairs,
        })

    conn.close()
    return result
