# Contributing to Principia Diagnostics

Thank you for your interest! This document explains how to get set up and contribute.

## Quick Setup

```bash
git clone https://github.com/IanD25/principia-diagnostics.git
cd principia-diagnostics
bash setup.sh                    # creates .venv, installs deps, builds index
source .venv/bin/activate
python -m pytest tests/ -v       # tests should pass
```

Or with pip (editable install):

```bash
git clone https://github.com/IanD25/principia-diagnostics.git
cd principia-diagnostics
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m src.sync               # build ChromaDB index
python -m pytest tests/ -v
```

## How to Contribute

### Adding a new dataset parser

This is the most impactful contribution. See existing parsers in `src/ingestion/parsers/` for patterns.

1. Create `src/ingestion/parsers/your_parser.py`
2. Parse your dataset into the RRP schema (entries, links, sections, properties)
3. Use `create_rrp_bundle()` from `ingestion.rrp_bundle`
4. Run Pass 2 to build bridges: `python scripts/run_entity_catalog_pass.py your_rrp.db data/chroma_db data/ds_wiki.db`
5. Run analysis: `pfd report --rrp your_rrp.db`
6. Add tests in `tests/`

### Improving claim extraction

The claim extractor (`src/analysis/claim_extractor.py`) uses pattern-based extraction. Contributions welcome for:
- New polarity markers (domain-specific negative/positive indicators)
- Better SRO extraction patterns
- Domain-specific claim indicators

## Code Standards

- **Tests required** -- every new module needs tests in `tests/`
- **INSERT OR IGNORE** -- all database writes must be idempotent
- **No schema changes to ds_wiki.db** -- it's read-only; new tables go in `wiki_history.db`
- **Probabilistic, not boolean** -- never return VALID/INVALID; return confidence scores (0-1)
- **numpy only in ingestion** -- no scipy dependency in the ingestion pipeline

## Running Tests

```bash
python -m pytest tests/ -v                             # all tests
python -m pytest tests/test_claim_extractor.py -v      # specific module
python -m pytest tests/ -k "integration" -v            # integration only
```

## Architectural Constraints

These are hard rules:

1. **ds_wiki.db is read-only** (schema never altered)
2. **Mandatory human gate** at claim extraction (Layer 1)
3. **Probabilistic pipeline** (confidence 0-1, never binary verdicts)
4. **Transparency** -- every output shows full reasoning
5. **formality_tier caps** -- Tier 1 max 0.95, Tier 2 max 0.85, Tier 3 max 0.70

## License

MIT -- see [LICENSE](LICENSE).
