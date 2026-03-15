"""
viz_runner.py — Orchestrator: run all three cross-universe visualizations.

Usage (Python API):
    from viz import run_all_viz
    result = run_all_viz("data/rrp/zoo_classes/rrp_zoo_classes.db")

Usage (CLI):
    python -m src.viz.viz_runner data/rrp/zoo_classes/rrp_zoo_classes.db
    python -m src.viz.viz_runner data/rrp/zoo_classes/rrp_zoo_classes.db --ds data/ds_wiki.db
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from config import SOURCE_DB, DATA_DIR
except ImportError:
    SOURCE_DB = None
    DATA_DIR  = Path("data")

# ── Default thresholds ────────────────────────────────────────────────────────

DEFAULT_NET_THRESHOLD  = 0.82   # tighter: keeps network readable (~432 edges at this level)
DEFAULT_SIM_THRESHOLD  = 0.75   # histogram + heatmap: show all stored bridges


# ── Output dir derivation ─────────────────────────────────────────────────────

def _derive_output_dir(bundle_db: str | Path) -> Path:
    """
    Auto-derive output directory from the bundle_db path.
        data/rrp/zoo_classes/rrp_zoo_classes.db  →  data/viz/zoo_classes/
    Falls back to data/viz/{stem}/ if the path structure differs.
    """
    p = Path(bundle_db).resolve()
    bundle_name = p.parent.name   # e.g. "zoo_classes"
    # Navigate up to data/ then into viz/
    output_dir = p.parent.parent.parent / "viz" / bundle_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all_viz(
    bundle_db:     str | Path,
    ds_wiki_db:    str | Path | None = None,
    net_threshold: float = DEFAULT_NET_THRESHOLD,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    output_dir:    str | Path | None = None,
) -> dict[str, Any]:
    """
    Run all three cross-universe visualizations for an RRP bundle.

    Args:
        bundle_db:     Path to the RRP bundle .db file.
        ds_wiki_db:    Path to ds_wiki.db (defaults to config.SOURCE_DB).
        net_threshold: Similarity cutoff for bridge_network (default 0.82).
        sim_threshold: Similarity cutoff for histogram + heatmap (default 0.75).
        output_dir:    Override output directory (default auto-derived).

    Returns:
        {
          "network":   {"png": Path, "html": Path, "stats": dict},
          "histogram": {"png": Path, "html": Path, "stats": dict},
          "heatmap":   {"png": Path, "html": Path, "stats": dict},
          "output_dir": Path,
        }
    """
    bundle_db = Path(bundle_db)
    if ds_wiki_db is None and SOURCE_DB is not None:
        ds_wiki_db = Path(SOURCE_DB)
    elif ds_wiki_db is not None:
        ds_wiki_db = Path(ds_wiki_db)

    out = Path(output_dir) if output_dir else _derive_output_dir(bundle_db)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations for: {bundle_db.name}")
    print(f"  DS Wiki : {ds_wiki_db}")
    print(f"  Output  : {out}")
    print(f"  Network threshold: sim ≥ {net_threshold}")
    print(f"  Hist/heatmap threshold: sim ≥ {sim_threshold}")
    print()

    from viz.similarity_hist import SimilarityHist
    from viz.domain_heatmap  import DomainHeatmap
    from viz.bridge_network  import BridgeNetwork

    results: dict[str, Any] = {"output_dir": out}

    results["histogram"] = SimilarityHist(bundle_db).generate(
        output_dir    = out,
        sim_threshold = sim_threshold,
    )
    results["heatmap"] = DomainHeatmap(bundle_db, ds_wiki_db).generate(
        output_dir    = out,
        sim_threshold = sim_threshold,
    )
    results["network"] = BridgeNetwork(bundle_db, ds_wiki_db).generate(
        output_dir    = out,
        sim_threshold = net_threshold,
    )

    print(f"\nAll done → {out}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate all cross-universe bridge visualizations for an RRP bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.viz.viz_runner data/rrp/zoo_classes/rrp_zoo_classes.db
  python -m src.viz.viz_runner data/rrp/zoo_classes/rrp_zoo_classes.db --ds data/ds_wiki.db
  python -m src.viz.viz_runner data/rrp/zoo_classes/rrp_zoo_classes.db --net-threshold 0.85
        """,
    )
    parser.add_argument("bundle_db", help="Path to RRP bundle .db file")
    parser.add_argument("--ds",  default=None,
                        help="Path to ds_wiki.db (default: config.SOURCE_DB)")
    parser.add_argument("--net-threshold", type=float, default=DEFAULT_NET_THRESHOLD,
                        help=f"Network sim threshold (default {DEFAULT_NET_THRESHOLD})")
    parser.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD,
                        help=f"Hist/heatmap sim threshold (default {DEFAULT_SIM_THRESHOLD})")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: auto-derived)")
    args = parser.parse_args()

    results = run_all_viz(
        bundle_db     = args.bundle_db,
        ds_wiki_db    = args.ds,
        net_threshold = args.net_threshold,
        sim_threshold = args.sim_threshold,
        output_dir    = args.out,
    )

    print("\n── Output files ──────────────────────────────────────────────────────")
    for viz_type in ("histogram", "heatmap", "network"):
        r = results[viz_type]
        if r.get("png"):
            print(f"  {viz_type:<12s}: {r['png'].name}  {r['html'].name}")
    print(f"  directory    : {results['output_dir']}")


if __name__ == "__main__":
    main()
