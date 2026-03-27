---
sidebar_position: 5
title: Results & roadmap
---

# Hackathon results

| Segment | Extraction | Ranking | Best offer | Total |
|---------|------------|---------|------------|-------|
| auta | 0.754 | 1.000 | 1.000 | **0.853** |
| lodě | 0.160 | 1.000 | 1.000 | **0.496** |
| odpovědnost | 0.136 | 0.750 | 1.000 | **0.419** |
| **Overall** | | | | **0.589** |

Ranking and best-offer detection were correct across segments; **extraction** was the limiting factor (nested tables, scans, token limits).

## Known limitations

- **odpovědnost (66 fields):** Tier II sublimits often buried in VPP/conditions with weak table structure; Tier I tends to extract better than Tier II repetitions — many fields can be N/A for some insurers.
- **lodě — Pantaenius:** `Conditions.pdf` can have **0 OCR** characters; vision fallback does not reliably parse pricing from scanned tables.
- **Range strings:** Expected formats like `"CZK 248,923–281,136"` need both Tier I and Tier II values merged when they appear on different pages.
- **Token limits:** odpovědnost payloads can exceed **~200k tokens** per insurer; two-pass extraction was required to stay within Gemini context and avoid truncation.

## Next steps (from the team)

- Field-specific prompts for the hardest odpovědnost sublimit fields.
- Table-aware parsing for lodě pricing (today largely free-text).
- Persistent shared cache (e.g. Cloud SQL) so Cloud Run restarts do not drop cache between evaluation rounds.
