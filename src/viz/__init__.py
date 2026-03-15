"""
viz/ — Visualization package for PFD analysis.

Generates static PNG charts and interactive D3.js graphs.

Tier-1 Internal Analysis:
    from viz import Tier1Dashboard
    dashboard = Tier1Dashboard(rrp_db)
    dashboard.generate_coherence_png(output_path)
    dashboard.generate_regime_png(output_path)
    dashboard.generate_network_html(output_path)

Cross-universe Bridges (Tier-2):
    from viz import BridgeNetwork, SimilarityHist, DomainHeatmap, Tier2Report
    BridgeNetwork(bundle_db, ds_wiki_db).generate(output_dir)
    SimilarityHist(bundle_db).generate(output_dir)
    DomainHeatmap(bundle_db, ds_wiki_db).generate(output_dir)
    Tier2Report(bundle_db, ds_wiki_db).generate(output_path)
"""

def __getattr__(name):
    """Lazy-load submodules so partial installs don't block each other."""
    if name == "Tier1Dashboard":
        from .tier1_dashboard import Tier1Dashboard
        return Tier1Dashboard
    if name == "Tier1Report":
        from .tier1_report import Tier1Report
        return Tier1Report
    if name == "Tier2Report":
        from .tier2_report import Tier2Report
        return Tier2Report
    if name == "BridgeNetwork":
        from .bridge_network import BridgeNetwork
        return BridgeNetwork
    if name == "SimilarityHist":
        from .similarity_hist import SimilarityHist
        return SimilarityHist
    if name == "DomainHeatmap":
        from .domain_heatmap import DomainHeatmap
        return DomainHeatmap
    if name == "run_all_viz":
        from .viz_runner import run_all_viz
        return run_all_viz
    raise AttributeError(f"module 'viz' has no attribute {name!r}")

__all__ = ["Tier1Dashboard", "Tier1Report", "Tier2Report", "BridgeNetwork", "SimilarityHist", "DomainHeatmap", "run_all_viz"]
