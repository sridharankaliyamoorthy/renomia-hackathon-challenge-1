---
sidebar_position: 3
title: Codebase
---

# Main modules

| File | Role |
|------|------|
| `main.py` | FastAPI app, **`/solve`** endpoint, orchestration, `ThreadPoolExecutor` (max 4 workers) |
| `extract.py` | Gemini extraction: document sorting, vision fallback, two-pass logic |
| `rank.py` | Deterministic win-count ranking and tiebreakers |
| `normalize.py` | `parse_number()`, `normalize_text_for_compare()`, document-type helpers |
| `cache.py` | SHA-256 cache keys, load/save from PostgreSQL |
| `preprocess.py` | `build_preferred_offer_text()`, assembly of document text |

## Deprecated code

These files remain in the repo but are **not imported** anywhere:

`segment_router.py`, `auto_extractor.py`, `yacht_extractor.py`, `extractors.py`, `text_fields.py`, `cache_utils.py` (under `deprecated/`).
