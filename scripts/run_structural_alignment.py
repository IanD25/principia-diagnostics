"""
run_structural_alignment.py — CLI for link-type weighted bridge scoring

Replaces SPT (Semantic Position Test) with a structurally grounded approach:
polarity is read from the RRP's link_type field, not inferred from LLM framing.

Usage:
    python scripts/run_structural_alignment.py \
        --rrp data/rrp/opera/rrp_opera_paper.db \
        [--output data/reports/opera/structural_alignment.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.structural_alignment import run_structural_alignment, print_sa_report


def main():
    parser = argparse.ArgumentParser(
        description="Run structural alignment analysis on an RRP database."
    )
    parser.add_argument("--rrp",    required=True, help="RRP database with cross_universe_bridges populated")
    parser.add_argument("--output", default=None,  help="JSON output path (optional)")
    args = parser.parse_args()

    if not Path(args.rrp).exists():
        print(f"ERROR: RRP database not found: {args.rrp}")
        sys.exit(1)

    result = run_structural_alignment(args.rrp)
    print_sa_report(result)

    output_path = args.output or (
        Path("data/reports") / Path(args.rrp).stem.replace("rrp_", "") / "structural_alignment.json"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "rrp_db": result.rrp_db,
        "mean_polarity": result.mean_polarity,
        "n_contested": len(result.contested_entries),
        "n_aligned": len(result.aligned_entries),
        "ds_wiki_summary": result.ds_wiki_summary(),
        "entries": [
            {
                "entry_id": ea.entry_id,
                "entry_type": ea.entry_type,
                "net_polarity": ea.net_polarity,
                "bridges": [
                    {
                        "ds_entry_id": b.ds_entry_id,
                        "raw_sim": b.raw_sim,
                        "polarity": b.polarity,
                        "signed_score": b.signed_score,
                        "alignment": b.alignment_label,
                        "path": b.path_description,
                        "hop": b.hop,
                    }
                    for b in ea.bridges
                    if abs(b.polarity) > 0.1
                ],
            }
            for ea in result.entries
        ],
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Structural alignment saved: {output_path}")


if __name__ == "__main__":
    main()
