---
sidebar_position: 2
title: Architecture
---

# Solution architecture

```
┌─────────────────────────────────────────────────┐
│                   INPUT                          │
│  Insurance PDFs + fields_to_extract + RFP        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              DOCUMENT PIPELINE                   │
│  1. Filter & sort docs (quotation first)         │
│  2. Clean OCR (LaTeX artifacts removed)          │
│  3. Reconstruct split OCR tables                 │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           GEMINI EXTRACTION                      │
│  • Split by field type (number vs string)        │
│  • Two-pass for missing fields                   │
│  • PDF vision fallback for 0-OCR documents       │
│  • Segment-specific prompts                      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         POST-PROCESSING                          │
│  • Czech string canonicalization                 │
│  • Range value reconstruction                    │
│  • Insurer-specific overrides                    │
│  • Hallucination guard                           │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           RANKING ENGINE                         │
│  • Win-count algorithm per field                 │
│  • Direction inference (higher/lower/qualitative)│
│  • 3x weight for number fields                   │
│  • Deterministic tiebreakers                     │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              OUTPUT                              │
│  offers_parsed + ranking + best_offer_id         │
└─────────────────────────────────────────────────┘
```

**Tools:** Python, Gemini 2.5 Flash, PostgreSQL, FastAPI, Google Cloud Run, pdfplumber.

## Key design decisions

- **Extraction:** Gemini 2.5 Flash with segment-aware Czech prompts and RFP injection.
- **Document sorting:** Quotation / pricing documents first; VPP / conditions last.
- **Two-pass extraction (odpovědnost only):** Pass 1 on quotation docs (66 fields). If more than 20 fields still missing → Pass 2 on VPP/conditions for the remaining fields only.
- **PDF vision fallback:** Only for documents with **&lt; 50 OCR characters** — uses Gemini Files API; uploaded file is deleted after extraction.
- **Wrong-segment filtering:** Skips documents that clearly belong to another segment (e.g. vehicle docs inside a boat offer).
- **Multi-option rule:** If a document presents Option A / Option B, extract the **higher coverage** option.
- **Ranking:** Pure **deterministic Python** win-count. **Gemini does not influence ranking.**  
  - **NUMBER** fields: weight **3×** (limits: higher is better; premiums: lower is better).  
  - **STRING** fields: weight **1×** (e.g. Ano &gt; Ne, Neomezeno &gt; Omezeno, celý svět &gt; Evropa &gt; ČR).
- **Caching:** SHA-256 key from offer id, insurer, segment, fields, document content hashes, and model version — stored in **PostgreSQL** sidecar for fast repeat runs.
