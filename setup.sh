#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Principia Diagnostics — One-command setup
# Run from repo root: bash setup.sh
#
# Works on: macOS (Apple Silicon), Linux, Windows WSL
# Requires: Python 3.11+
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "═══════════════════════════════════════════════════════"
echo "  Principia Diagnostics — Environment Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── 1. Detect Python ────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$($cmd --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
            PYTHON="$cmd"
            echo "✓ Python found: $cmd ($VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "✗ Python 3.11+ not found. Please install Python 3.11 or higher."
    echo "  macOS:   brew install python@3.13"
    echo "  Ubuntu:  sudo apt install python3.13 python3.13-venv"
    echo "  Windows: https://www.python.org/downloads/"
    exit 1
fi

# ── 2. Create virtual environment ───────────────────────────────────────────
if [ -d ".venv" ]; then
    echo "✓ .venv already exists — skipping creation"
else
    echo "→ Creating virtual environment..."
    $PYTHON -m venv .venv
    echo "✓ .venv created"
fi

# ── 3. Activate ─────────────────────────────────────────────────────────────
echo "→ Activating .venv..."
# shellcheck disable=SC1091
source .venv/bin/activate

# ── 4. Upgrade pip ──────────────────────────────────────────────────────────
echo "→ Upgrading pip..."
pip install --upgrade pip -q

# ── 5. Install package (editable) ──────────────────────────────────────────
echo "→ Installing Principia Diagnostics (pip install -e .)..."
pip install -e ".[dev]" -q
echo "✓ Installed with all dependencies"

# ── 6. Verify data ─────────────────────────────────────────────────────────
echo ""
echo "→ Checking data files..."
if [ -f "data/ds_wiki.db" ]; then
    SIZE=$(du -sh data/ds_wiki.db | cut -f1)
    echo "  ✓ data/ds_wiki.db ($SIZE)"
else
    echo "  ✗ data/ds_wiki.db not found — check git clone was complete"
    exit 1
fi

# ── 7. Build semantic index (ChromaDB) ──────────────────────────────────────
echo ""
if [ -d "data/chroma_db" ] && [ -f "data/wiki_history.db" ]; then
    echo "✓ Semantic index already built"
    echo "  Rebuild if needed: python3 -m src.sync"
else
    echo "→ Building semantic index (first run — downloads ~430MB embedding model)..."
    echo "  This takes 1-2 minutes on first run."
    python3 -m src.sync
    echo "✓ Semantic index built"
fi

# ── 8. Optional: run tests ─────────────────────────────────────────────────
echo ""
read -r -p "→ Run test suite now? [Y/n]: " RUNTESTS
RUNTESTS="${RUNTESTS:-Y}"
if [[ "$RUNTESTS" =~ ^[Yy]$ ]]; then
    echo "→ Running pytest..."
    python3 -m pytest tests/ -v --tb=short -q
else
    echo "  Skipped — run 'python3 -m pytest tests/ -v' manually when ready"
fi

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    source .venv/bin/activate"
echo "    pfd report --rrp data/rrp/ecoli_core/rrp_ecoli_core.db"
echo "    pfd --help"
echo "═══════════════════════════════════════════════════════"
