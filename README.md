# Team MrRooT.Ai — Challenge 1: Insurance Offer Comparison

**Team:** MrRooT.Ai (Team 15)  
**GitHub:** sridharankaliyamoorthy  
**Score:** 0.589 overall (auta 0.853 / lodě 0.496 / odpovědnost 0.419)

---

## Solution Architecture

A dynamic AI extraction + deterministic ranking pipeline for any insurance segment.

- **Extraction:** Gemini 2.5 Flash with segment-aware Czech prompts
- **Document sorting:** Quotation/pricing docs first, VPP/conditions last
- **Two-pass extraction (odpovědnost only):** Pass 1 on quotation docs (66 fields). If >20 fields still missing → Pass 2 on VPP/conditions docs for remaining fields only
- **PDF vision fallback:** Triggered only for docs with <50 OCR chars — uses Gemini Files API, file deleted after extraction
- **Wrong-segment filtering:** Skips documents that clearly belong to a different segment (e.g. vehicle insurance docs inside a boat offer)
- **Multi-option rule:** If a doc presents Option A / Option B, extract the higher coverage option
- **Ranking:** Pure deterministic Python win-count. Gemini never influences ranking. NUMBER fields weighted 3x (limits: higher = better, premiums: lower = better). STRING fields weighted 1x (Ano > Ne, Neomezeno > Omezeno, celý svět > Evropa > ČR)
- **Caching:** SHA-256 key = offer_id + insurer + segment + fields + doc content hashes + model version. Stored in PostgreSQL sidecar. Near-zero latency on repeat evaluation runs

---

## Results

| Segment | Extraction | Ranking | Best Offer | Total |
|---|---|---|---|---|
| auta | 0.754 | 1.000 | 1.000 | **0.853** |
| lodě | 0.160 | 1.000 | 1.000 | **0.496** |
| odpovědnost | 0.136 | 0.750 | 1.000 | **0.419** |
| **Overall** | | | | **0.589** |

Ranking and best-offer detection are correct across all segments. Field extraction is the main bottleneck, particularly for deeply nested sublimit tables in odpovědnost and scanned PDFs in lodě.

---

## Active Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, `/solve` endpoint, orchestration, ThreadPoolExecutor (max 4 workers) |
| `extract.py` | All Gemini extraction logic, document sorting, vision fallback, two-pass logic |
| `rank.py` | Deterministic win-count ranking, tiebreakers |
| `normalize.py` | `parse_number()`, `normalize_text_for_compare()`, doc type helpers |
| `cache.py` | SHA-256 cache key computation, load/save from PostgreSQL |
| `preprocess.py` | `build_preferred_offer_text()`, document text assembly |

Deprecated files (`segment_router.py`, `auto_extractor.py`, `yacht_extractor.py`, `extractors.py`, `text_fields.py`, `cache_utils.py`) are kept in the repo but never imported.

---

## Roadblocks

- **odpovědnost 66 fields:** Many "Tier II" sublimits are buried in VPP/conditions documents with no clear table structure. Gemini correctly extracts Tier I limits but misses Tier II repetitions across sections — 32/66 fields return N/A for some insurers
- **lodě — Pantaenius:** `Conditions.pdf` has 0 OCR characters in training data. PDF vision fallback (Gemini Files API) was added but doesn't reliably parse pricing values from scanned/image-based tables
- **Range values in odpovědnost:** Expected format is `"CZK 248,923–281,136"` (a range string). Documents split Tier I and Tier II across different pages — combining them into a single range string requires knowing both values exist before formatting
- **Token limits:** odpovědnost documents exceed 200K tokens per insurer. Required two-pass extraction split to stay within Gemini context window and avoid truncation

---

## What We Would Do Next

- Field-specific extraction prompts for the 10 hardest odpovědnost sublimit fields
- Table-aware parsing for lodě pricing rows (currently treated as free text)
- Persistent shared cache via Cloud SQL to survive Cloud Run restarts between evaluation rounds

---

## Local Development

```bash
# Start app + PostgreSQL sidecar
docker compose up --build

# Test the endpoint
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "segment": "auta",
    "fields_to_extract": ["Roční pojistné"],
    "field_types": {"Roční pojistné": "number"},
    "offers": [{"id": "test", "insurer": "Test", "label": "Test", "documents": [{"filename": "test.pdf", "ocr_text": "Roční pojistné: 125 000 Kč"}]}]
  }'

# Health check
curl http://localhost:8080/

# Token usage
curl http://localhost:8080/metrics
```

## Deployment

Push to GitHub — Cloud Build auto-triggers, builds Docker image, and deploys to Cloud Run.
