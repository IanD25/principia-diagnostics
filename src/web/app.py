"""
app.py — FastAPI web interface for Principia Formal Diagnostics.

Single-page app: upload a PDF, see extracted sections and claims,
review at the human gate, then run the full PFD diagnostic pipeline.

Usage:
    python -m web.app                # http://localhost:8000
    uvicorn web.app:app --reload     # development mode

Endpoints:
    GET  /                → main page
    POST /upload          → upload PDF, extract sections + claims
    POST /approve         → approve/reject claims at human gate
    POST /analyze         → run PFD pipeline on approved claims
    GET  /demo            → run demo on bundled OPERA dataset
    GET  /api/status      → pipeline status check
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import sys
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

app = FastAPI(title="Principia Formal Diagnostics", version="0.1.0")

# Templates and static files
_WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")

# Project paths
PROJECT_ROOT = _SRC.parent
DATA_DIR = PROJECT_ROOT / "data"
WIKI_DB = DATA_DIR / "ds_wiki.db"

# Session state (in-memory, single-user for now)
_session: dict = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page."""
    return templates.TemplateResponse(request, "index.html", {
        "session": _session,
    })


@app.post("/upload", response_class=HTMLResponse)
async def upload_pdf(request: Request, pdf_file: UploadFile = File(...)):
    """Upload a PDF and extract sections + claims."""
    global _session
    _session = {}

    try:
        # Save uploaded PDF to temp file
        suffix = Path(pdf_file.filename or "paper.pdf").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await pdf_file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        _session["filename"] = pdf_file.filename
        _session["pdf_path"] = str(tmp_path)

        # Extract sections
        try:
            from ingestion.parsers.pdf_parser import extract_sections, detect_math_regions
            sections = extract_sections(tmp_path)
            _session["sections"] = sections
            _session["section_count"] = len(sections)

            # Detect math regions
            full_text = "\n\n".join(sections.values())
            math_flags = detect_math_regions(full_text)
            _session["math_flag_count"] = len(math_flags)
        except ImportError:
            # PyMuPDF not installed — fall back to info message
            _session["error"] = "PyMuPDF not installed. Run: pip install pymupdf"
            return templates.TemplateResponse(request, "index.html", {
                "session": _session,
            })

        # Extract claims
        from analysis.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor()
        claims = extractor.extract_from_sections(sections)
        _session["claims"] = [
            {
                "idx": i,
                "text": c.text,
                "subject": c.subject,
                "relationship": c.relationship,
                "object": c.object,
                "confidence": c.confidence.value,
                "polarity": c.polarity_hint.value,
                "section": c.source_section,
                "approved": False,
            }
            for i, c in enumerate(claims)
        ]
        _session["claim_count"] = len(claims)
        _session["step"] = "review"

    except Exception as e:
        _session["error"] = f"Upload failed: {e}\n{traceback.format_exc()}"

    return templates.TemplateResponse(request, "index.html", {
        "session": _session,
    })


@app.post("/approve", response_class=HTMLResponse)
async def approve_claims(request: Request):
    """Approve/reject claims at the human gate."""
    form = await request.form()

    if "claims" not in _session:
        _session["error"] = "No claims to approve. Upload a PDF first."
        return templates.TemplateResponse(request, "index.html", {
            "session": _session,
        })

    # Update approval status from checkboxes
    approved_indices = set()
    for key in form.keys():
        if key.startswith("claim_"):
            try:
                idx = int(key.replace("claim_", ""))
                approved_indices.add(idx)
            except ValueError:
                pass

    for claim in _session["claims"]:
        claim["approved"] = claim["idx"] in approved_indices

    approved_count = len(approved_indices)
    _session["approved_count"] = approved_count

    if approved_count == 0:
        _session["error"] = "No claims approved. Select at least one claim to proceed."
        return templates.TemplateResponse(request, "index.html", {
            "session": _session,
        })

    _session["step"] = "approved"
    _session.pop("error", None)

    # Build a minimal RRP from approved claims and run PFD
    try:
        report_text, report_data = _run_pfd_on_claims(
            [c for c in _session["claims"] if c["approved"]],
            _session.get("sections", {}),
        )
        _session["report_text"] = report_text
        _session["report_data"] = report_data
        _session["step"] = "report"
    except Exception as e:
        _session["error"] = f"Analysis failed: {e}\n{traceback.format_exc()}"

    return templates.TemplateResponse(request, "index.html", {
        "session": _session,
    })


@app.get("/demo", response_class=HTMLResponse)
async def run_demo(request: Request):
    """Run demo on the bundled OPERA dataset."""
    global _session
    _session = {}

    try:
        from analysis.fisher_report import generate_report

        # Use OPERA paper RRP (bundled in repo)
        opera_rrp = DATA_DIR / "rrp" / "opera" / "rrp_opera_paper.db"
        if not opera_rrp.exists():
            _session["error"] = "OPERA RRP not found. Run the demo from the project directory."
            return templates.TemplateResponse(request, "index.html", {
                "session": _session,
            })

        report = generate_report(opera_rrp, WIKI_DB)
        _session["filename"] = "OPERA Neutrino Experiment (bundled demo)"
        _session["report_text"] = report.as_text()
        _session["report_data"] = {
            "pfd_score": report.pfd_score,
            "tier1_verdict": report.tier1_verdict,
            "tier2_verdict": report.tier2_verdict,
            "tier1_coherence": report.tier1_coherence,
            "formality_weight": report.formality_weight,
            "formality_breakdown": report.formality_breakdown,
            "tier2_bridge_edges": report.tier2_bridge_edges,
            "tier2_wiki_anchors_reached": report.tier2_wiki_anchors_reached,
        }
        _session["step"] = "report"
        _session["demo"] = True

    except Exception as e:
        _session["error"] = f"Demo failed: {e}\n{traceback.format_exc()}"

    return templates.TemplateResponse(request, "index.html", {
        "session": _session,
    })


@app.get("/api/status")
async def api_status():
    """Health check + current pipeline status."""
    return {
        "status": "ok",
        "wiki_db_exists": WIKI_DB.exists(),
        "wiki_entries": _count_wiki_entries(),
        "session_step": _session.get("step", "none"),
    }


def _count_wiki_entries() -> int:
    if not WIKI_DB.exists():
        return 0
    conn = sqlite3.connect(WIKI_DB)
    count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    conn.close()
    return count


def _run_pfd_on_claims(
    approved_claims: list[dict],
    sections: dict[str, str],
) -> tuple[str, dict]:
    """
    Build a temporary RRP from approved claims and run PFD diagnostics.

    Returns (report_text, report_data_dict).
    """
    from ingestion.rrp_bundle import create_rrp_bundle
    from analysis.fisher_report import generate_report

    # Create a temporary RRP DB from claims
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        rrp_path = Path(tmp.name)

    conn = create_rrp_bundle(rrp_path)

    # Insert entries from claims
    for i, claim in enumerate(approved_claims):
        entry_id = f"claim_{i:03d}"
        conn.execute(
            "INSERT OR IGNORE INTO entries (id, title, entry_type, domain) VALUES (?, ?, ?, ?)",
            (entry_id, claim["text"][:80], "instantiation", ""),
        )
        # Add a section with the full claim text
        conn.execute(
            "INSERT OR IGNORE INTO sections (entry_id, section_name, content) VALUES (?, ?, ?)",
            (entry_id, "claim", claim["text"]),
        )

    # Insert links based on claim relationships
    for i, c1 in enumerate(approved_claims):
        for j, c2 in enumerate(approved_claims):
            if i >= j:
                continue
            # Link claims from the same section
            if c1["section"] and c1["section"] == c2["section"]:
                conn.execute(
                    "INSERT OR IGNORE INTO links (source_id, target_id, link_type) VALUES (?, ?, ?)",
                    (f"claim_{i:03d}", f"claim_{j:03d}", "couples to"),
                )

    conn.commit()

    # Run cross-universe query to build bridges
    try:
        from ingestion.cross_universe_query import CrossUniverseQuery
        from config import CHROMA_DIR

        chroma_dir = DATA_DIR / "chroma_db"
        if not chroma_dir.exists():
            chroma_dir = Path(CHROMA_DIR)

        query = CrossUniverseQuery(
            rrp_db=rrp_path,
            chroma_dir=chroma_dir,
            wiki_db=WIKI_DB,
        )
        query.run()
    except Exception:
        # If cross-universe query fails, continue with internal-only report
        pass

    conn.close()

    # Generate PFD report
    report = generate_report(rrp_path, WIKI_DB)

    report_data = {
        "pfd_score": report.pfd_score,
        "tier1_verdict": report.tier1_verdict,
        "tier2_verdict": report.tier2_verdict,
        "tier1_coherence": report.tier1_coherence,
        "formality_weight": report.formality_weight,
        "formality_breakdown": report.formality_breakdown,
        "tier2_bridge_edges": report.tier2_bridge_edges,
        "tier2_wiki_anchors_reached": report.tier2_wiki_anchors_reached,
        "claim_count": len(approved_claims),
    }

    return report.as_text(), report_data


# ── CLI entry ────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    print("\n  Principia Formal Diagnostics — Web Interface")
    print(f"  DS Wiki: {WIKI_DB} ({'found' if WIKI_DB.exists() else 'NOT FOUND'})")
    print(f"  Open: http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
