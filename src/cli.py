"""
cli.py — Unified CLI entry point for PFD.

Usage after `pip install -e .`:
    pfd demo                              # zero-config demo on bundled paper
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

    # ── Demo ─────────────────────────────────────────────────────────────────
    p_demo = sub.add_parser(
        "demo",
        help="Run a zero-config demo on a bundled paper (OPERA neutrino experiment)",
    )
    p_demo.add_argument(
        "--dataset",
        choices=["opera", "ecoli", "ccbh", "universal_paralogs"],
        default="opera",
        help="Which bundled dataset to demo (default: opera)",
    )

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

    if args.command == "demo":
        return _run_demo(args.dataset)

    return 0


def _run_demo(dataset: str) -> int:
    """Run a zero-config demo on a bundled dataset."""
    import time
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    # Dataset configs
    datasets = {
        "opera": {
            "rrp": project_root / "data" / "rrp" / "opera" / "rrp_opera_paper.db",
            "title": "OPERA Neutrino Experiment",
            "paper": "T. Adam et al. (2012)",
            "description": (
                "The OPERA experiment initially reported faster-than-light neutrinos\n"
                "  in 2011, challenging special relativity. A systematic timing error\n"
                "  (loose fiber-optic cable) was later identified, and the corrected\n"
                "  result confirmed subluminal neutrino velocity.\n\n"
                "  PFD diagnostics reveal how the corrected result integrates cleanly\n"
                "  with foundational physics, while the original FTL claim generated\n"
                "  structural tension against Lorentz invariance."
            ),
            "entries": 25,
            "bridges": 60,
        },
        "ecoli": {
            "rrp": project_root / "data" / "rrp" / "ecoli_core" / "rrp_ecoli_core.db",
            "title": "E. coli Core Metabolic Network",
            "paper": "Orth et al. (2010) — iAF1260 core model",
            "description": (
                "The E. coli core metabolic model: 72 metabolites, 95 reactions,\n"
                "  137 genes. A gold-standard systems biology dataset.\n\n"
                "  PFD diagnostics show strong internal coherence (pyruvate as the\n"
                "  top hub, d_eff=11) and deep integration with thermodynamics,\n"
                "  chemistry, and biology foundations in the DS Wiki."
            ),
            "entries": 304,
            "bridges": 912,
        },
        "ccbh": {
            "rrp": project_root / "data" / "rrp" / "ccbh" / "rrp_ccbh_cluster.db",
            "title": "Cosmologically Coupled Black Holes (CCBH)",
            "paper": "Farrah et al. (2023) + Croker et al. (2024) cluster",
            "description": (
                "A 3-paper cluster proposing that black holes gain mass through\n"
                "  cosmological coupling (k ≈ 3), linking dark energy to BH interiors.\n\n"
                "  PFD diagnostics show marginal internal consistency (the hypothesis\n"
                "  is genuinely contested) but strong grounding in general relativity\n"
                "  and cosmology foundations — exactly what you'd expect from a bold\n"
                "  but well-constructed physics conjecture."
            ),
            "entries": 22,
            "bridges": 60,
        },
        "universal_paralogs": {
            "rrp": project_root / "data" / "rrp" / "universal_paralogs" / "rrp_universal_paralogs.db",
            "title": "Universal Paralogs & Pre-LUCA Evolution",
            "paper": "Goldman et al. (2026) — Cell Genomics",
            "description": (
                "Analysis of universal paralog families to reconstruct the gene\n"
                "  content of the Last Universal Common Ancestor (LUCA).\n\n"
                "  PFD diagnostics show high internal consistency and strong\n"
                "  grounding across biology, chemistry, and information theory\n"
                "  foundations in the DS Wiki."
            ),
            "entries": 19,
            "bridges": 57,
        },
    }

    cfg = datasets[dataset]
    rrp_db = cfg["rrp"]
    wiki_db = project_root / "data" / "ds_wiki.db"

    if not rrp_db.exists():
        print(f"Error: RRP bundle not found: {rrp_db}", file=sys.stderr)
        print("Run `pfd sync` first, or check your data/ directory.", file=sys.stderr)
        return 1

    if not wiki_db.exists():
        print(f"Error: DS Wiki database not found: {wiki_db}", file=sys.stderr)
        return 1

    # ── Banner ────────────────────────────────────────────────────────────────
    bar = "=" * 64
    thin = "-" * 64
    print()
    print(bar)
    print("  PRINCIPIA FORMAL DIAGNOSTICS — DEMO")
    print(bar)
    print()
    print(f"  Dataset : {cfg['title']}")
    print(f"  Paper   : {cfg['paper']}")
    print(f"  RRP     : {cfg['entries']} entries, {cfg['bridges']} cross-universe bridges")
    print()
    print(f"  {cfg['description']}")
    print()
    print(thin)
    print()
    print("  WHAT THIS DEMO DOES:")
    print("  1. Analyzes the RRP's internal graph structure (Tier-1 coherence)")
    print("  2. Evaluates cross-universe bridges to DS Wiki (Tier-2 integration)")
    print("  3. Applies formality-tier weighting (Phase 4.3)")
    print("  4. Produces a two-tier PFD diagnostic report")
    print()
    print(thin)
    print("  Running analysis", end="", flush=True)

    t0 = time.time()

    # ── Run the report ────────────────────────────────────────────────────────
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from analysis.fisher_report import generate_report
    report = generate_report(rrp_db, wiki_db)

    elapsed = time.time() - t0
    print(f" ... done ({elapsed:.1f}s)")
    print()

    # ── Print the full report ─────────────────────────────────────────────────
    print(report.as_text())

    # ── Interpretation guide ──────────────────────────────────────────────────
    print()
    print(thin)
    print("  HOW TO READ THIS REPORT:")
    print(thin)
    print()
    print("  PFD Score (0.0 - 1.0):")
    print("    0.90+  = Strong: internally coherent + well-grounded in formal foundations")
    print("    0.70-0.89 = Moderate: some structural gaps or weak grounding")
    print("    <0.70  = Weak: fragmented structure or poor formal grounding")
    print()
    print("  Tier-1 (Internal Consistency):")
    print("    Measures how well the paper's own claims connect to each other.")
    print("    RADIAL hubs = well-structured knowledge centers")
    print("    NOISE nodes = poorly connected or incoherent claims")
    print()
    print("  Tier-2 (Bridge Quality):")
    print("    Measures how well the paper connects to established science.")
    print("    Formality weight reflects the rigor of the anchoring foundations:")
    print("    Tier-1 anchors (physics/math) = full weight (1.0)")
    print("    Tier-2 anchors (chemistry/bio) = 0.85 weight")
    print("    Tier-3 anchors (soft science)  = 0.70 weight")
    print()
    print(thin)
    print("  TRY NEXT:")
    print(f"    pfd report --rrp <your_rrp.db>     # analyze your own dataset")
    print(f"    pfd extract --file <paper.pdf>      # extract claims from a PDF")
    print(f"    pfd resolve \"<scientific claim>\"     # check a claim against DS Wiki")
    if dataset != "opera":
        print(f"    pfd demo                            # try the default OPERA demo")
    if dataset != "ecoli":
        print(f"    pfd demo --dataset ecoli            # try the E. coli metabolic demo")
    if dataset != "ccbh":
        print(f"    pfd demo --dataset ccbh             # try the CCBH cosmology demo")
    print()
    print(bar)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
