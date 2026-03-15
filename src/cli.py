"""
cli.py — Unified CLI entry point for PFD.

Usage after `pip install -e .`:
    pfd report --rrp data/rrp/ecoli_core/rrp_ecoli_core.db
    pfd internal --rrp data/rrp/ecoli_core/rrp_ecoli_core.db
    pfd bridge --rrp data/rrp/ecoli_core/rrp_ecoli_core.db
    pfd node --node-id CHEM5
    pfd wiki
    pfd sync
    pfd extract "We find that k = 3"
    pfd resolve "We find that k = 3"
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="pfd",
        description="Principia Formal Diagnostics — graph coherence engine",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── Fisher Suite modes ────────────────────────────────────────────────────
    for mode in ("report", "internal", "bridge", "node", "wiki"):
        p = sub.add_parser(mode, help=f"Fisher Suite: --mode {mode}")
        p.add_argument("--rrp", help="Path to RRP bundle database")
        p.add_argument("--db", help="Path to DS Wiki database (default: data/ds_wiki.db)")
        p.add_argument("--node-id", help="Entry ID (for 'node' mode)")
        p.add_argument("--min-sim", type=float, default=0.75, help="Min bridge similarity")
        p.add_argument("--save", action="store_true", help="Save results to wiki_history.db")

    # ── Sync ──────────────────────────────────────────────────────────────────
    sub.add_parser("sync", help="Rebuild ChromaDB + wiki_history from ds_wiki.db")

    # ── Claim extraction ──────────────────────────────────────────────────────
    p_extract = sub.add_parser("extract", help="Extract claims from text")
    p_extract.add_argument("text", nargs="?", help="Text to extract claims from")
    p_extract.add_argument("--file", "-f", help="Read text from file")

    # ── Claim resolution ──────────────────────────────────────────────────────
    p_resolve = sub.add_parser("resolve", help="Resolve a claim against DS Wiki")
    p_resolve.add_argument("claim", help="Claim text to resolve")
    p_resolve.add_argument("--top-k", type=int, default=5, help="Number of channels")

    # ── Structural alignment ──────────────────────────────────────────────────
    p_sa = sub.add_parser("align", help="Structural alignment on an RRP")
    p_sa.add_argument("--rrp", required=True, help="RRP database path")
    p_sa.add_argument("--output", help="JSON output path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # ── Dispatch ──────────────────────────────────────────────────────────────

    if args.command in ("report", "internal", "bridge", "node", "wiki"):
        # Map to Fisher Suite CLI
        mode_map = {"internal": "internal_rrp", "wiki": "ds_wiki"}
        mode = mode_map.get(args.command, args.command)
        fisher_args = ["--mode", mode]
        if args.rrp:
            fisher_args += ["--rrp", args.rrp]
        if args.db:
            fisher_args += ["--db", args.db]
        if args.node_id:
            fisher_args += ["--node", args.node_id]
        if args.min_sim != 0.75:
            fisher_args += ["--min-sim", str(args.min_sim)]
        if args.save:
            fisher_args += ["--save"]

        # Re-invoke run_fisher_suite via subprocess
        import subprocess
        from pathlib import Path
        script = Path(__file__).resolve().parent.parent / "scripts" / "run_fisher_suite.py"
        result = subprocess.run(
            [sys.executable, str(script)] + fisher_args,
            cwd=str(script.parent.parent),
        )
        return result.returncode

    if args.command == "sync":
        from sync import main as sync_main
        sync_main()
        return 0

    if args.command == "extract":
        from analysis.claim_extractor import ClaimExtractor
        from pathlib import Path

        if args.file:
            file_path = Path(args.file)
            if file_path.suffix.lower() == ".pdf":
                # PDF extraction — section-aware claim extraction
                from ingestion.parsers.pdf_parser import extract_to_claims
                claims = extract_to_claims(file_path)
                extractor = ClaimExtractor()
                print(extractor.format_for_human_gate(claims))
                return 0
            else:
                text = file_path.read_text(encoding="utf-8")
        elif args.text:
            text = args.text
        else:
            print("Error: provide text as argument or use --file", file=sys.stderr)
            return 1

        extractor = ClaimExtractor()
        claims = extractor.extract_claims(text)
        print(extractor.format_for_human_gate(claims))
        return 0

    if args.command == "resolve":
        from analysis.result_validator import ResultValidator
        validator = ResultValidator()
        resolution = validator.resolve_claim(args.claim, top_k=args.top_k)
        print(resolution.as_markdown())
        return 0

    if args.command == "align":
        from analysis.structural_alignment import run_structural_alignment
        import json
        result = run_structural_alignment(args.rrp)
        # Print summary
        print(f"\nMean polarity: {result.mean_polarity:+.3f}")
        print(f"Contested: {len(result.contested_entries)} | Aligned: {len(result.aligned_entries)}")
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(
                json.dumps(result.__dict__, default=str, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {args.output}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
