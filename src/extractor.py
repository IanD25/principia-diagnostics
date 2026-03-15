"""
extractor.py — Read ds_wiki.db and produce a flat list of Chunk objects.
Pure read. No writes anywhere.
"""
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from config import SOURCE_DB


@dataclass
class Chunk:
    chunk_id:     str
    entry_id:     str
    chunk_type:   str          # "section" | "conjecture" | "gate" | "bridge"
    title:        str
    section_name: str
    embed_text:   str          # text sent to embedding model
    metadata:     dict = field(default_factory=dict)


def _slug(text: str) -> str:
    """Normalise a string for use as part of a chunk_id."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def extract_chunks(db_path: Path = SOURCE_DB) -> list[Chunk]:
    if not db_path.exists():
        raise FileNotFoundError(f"Source DB not found: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    chunks: list[Chunk] = []

    # ── 1. Entry sections ────────────────────────────────────────────────────
    cur.execute("""
        SELECT e.id, e.title, e.entry_type, e.scale, e.domain,
               e.status, e.confidence, e.type_group,
               s.section_name, s.content, s.section_order
        FROM entries e
        JOIN sections s ON s.entry_id = e.id
        ORDER BY e.id, s.section_order
    """)
    for row in cur.fetchall():
        entry_id     = row["id"]
        entry_type   = row["entry_type"] or ""
        section_name = row["section_name"]
        content      = (row["content"] or "").strip()
        if not content:
            continue

        # ── Clean base layer: strip DS-specific sections from reference_law ──
        # DS Cross-References: entirely DS framework content, excluded from
        # reference_law embeddings so the base vector layer is domain-pure.
        if entry_type == "reference_law" and section_name == "DS Cross-References":
            continue

        # Mathematical Archetype: keep the archetype name (first line) only.
        # The boilerplate paragraph contains Ω_D / D-sensitive / D-invariant
        # language that pulls all reference_law entries toward each other.
        if entry_type == "reference_law" and section_name == "Mathematical Archetype":
            content = content.split("\n\n")[0].strip()
            if not content:
                continue

        chunk_id = f"{entry_id}_{_slug(section_name)}"
        embed_text = f"{row['title']}\n[{section_name}]\n{content}"

        chunks.append(Chunk(
            chunk_id=chunk_id,
            entry_id=entry_id,
            chunk_type="section",
            title=row["title"],
            section_name=section_name,
            embed_text=embed_text,
            metadata={
                "entry_id":   entry_id,
                "chunk_type": "section",
                "section_name": section_name,
                "entry_type": row["entry_type"] or "",
                "scale":      row["scale"] or "",
                "domain":     row["domain"] or "",
                "status":     row["status"] or "",
                "confidence": row["confidence"] or "",
                "type_group": row["type_group"] or "",
            },
        ))

    # ── 2. Conjectures ───────────────────────────────────────────────────────
    cur.execute("""
        SELECT id, title, claim, depends_on, would_confirm,
               would_kill, critical_gaps, gate
        FROM conjectures
        ORDER BY conjecture_order
    """)
    for row in cur.fetchall():
        parts = [f"Conjecture {row['id']}: {row['title']}"]
        if row["claim"]:        parts.append(f"Claim: {row['claim']}")
        if row["would_confirm"]:parts.append(f"Would confirm: {row['would_confirm']}")
        if row["would_kill"]:   parts.append(f"Would kill: {row['would_kill']}")
        if row["critical_gaps"]:parts.append(f"Critical gaps: {row['critical_gaps']}")

        chunks.append(Chunk(
            chunk_id=f"conj_{row['id']}",
            entry_id=f"conj_{row['id']}",
            chunk_type="conjecture",
            title=f"Conjecture {row['id']}: {row['title']}",
            section_name="conjecture",
            embed_text="\n".join(parts),
            metadata={
                "entry_id":   f"conj_{row['id']}",
                "chunk_type": "conjecture",
                "section_name": "conjecture",
                "gate":       row["gate"] or "",
                "entry_type": "conjecture",
                "scale": "", "domain": "", "status": "conjectured",
                "confidence": "", "type_group": "P",
            },
        ))

    # ── 3. Gates ─────────────────────────────────────────────────────────────
    cur.execute("SELECT id, claim, priority, blocking FROM gates")
    for row in cur.fetchall():
        parts = [f"Gate {row['id']}"]
        if row["claim"]:    parts.append(f"Claim: {row['claim']}")
        if row["priority"]: parts.append(f"Priority: {row['priority']}")
        if row["blocking"]: parts.append(f"Blocking: {row['blocking']}")

        chunks.append(Chunk(
            chunk_id=f"gate_{row['id']}",
            entry_id=f"gate_{row['id']}",
            chunk_type="gate",
            title=f"Gate {row['id']}",
            section_name="gate",
            embed_text="\n".join(parts),
            metadata={
                "entry_id":   f"gate_{row['id']}",
                "chunk_type": "gate",
                "section_name": "gate",
                "priority":   row["priority"] or "",
                "entry_type": "gate",
                "scale": "", "domain": "", "status": "",
                "confidence": "", "type_group": "G",
            },
        ))

    # ── 4. Bridge content ────────────────────────────────────────────────────
    cur.execute("""
        SELECT section_name, content
        FROM bridge_content
        ORDER BY section_order
    """)
    for row in cur.fetchall():
        content = (row["content"] or "").strip()
        if not content:
            continue

        section_name = row["section_name"]
        chunks.append(Chunk(
            chunk_id=f"bridge_{_slug(section_name)}",
            entry_id="bridge",
            chunk_type="bridge",
            title=f"Bridge: {section_name}",
            section_name=section_name,
            embed_text=f"[Bridge: {section_name}]\n{content}",
            metadata={
                "entry_id":   "bridge",
                "chunk_type": "bridge",
                "section_name": section_name,
                "entry_type": "bridge",
                "scale": "", "domain": "", "status": "",
                "confidence": "", "type_group": "bridge",
            },
        ))

    conn.close()

    counts = {
        "section":    sum(1 for c in chunks if c.chunk_type == "section"),
        "conjecture": sum(1 for c in chunks if c.chunk_type == "conjecture"),
        "gate":       sum(1 for c in chunks if c.chunk_type == "gate"),
        "bridge":     sum(1 for c in chunks if c.chunk_type == "bridge"),
    }
    print(f"Extracted {len(chunks)} chunks: {counts}")
    return chunks


if __name__ == "__main__":
    chunks = extract_chunks()
    # Print first 3 chunks as a sanity check
    for c in chunks[:3]:
        print(f"\n{'─'*60}")
        print(f"  chunk_id : {c.chunk_id}")
        print(f"  type     : {c.chunk_type}")
        print(f"  title    : {c.title}")
        print(f"  section  : {c.section_name}")
        print(f"  embed_text preview: {c.embed_text[:120]!r}")
        print(f"  metadata : {c.metadata}")
