# Challenge 1: Porovnání pojistných nabídek (Insurance Offer Comparison)

Compare multiple insurance offers across different segments, extract key parameters, and rank them.

## What you need to do

Implement the `solve()` function in `main.py`. Your endpoint receives OCR-extracted text from insurance offer documents and must:

1. **Parse** each offer to extract the specific fields listed in `fields_to_extract`
2. **Rank** offers from best to worst (best coverage, lowest cost)
3. **Identify** the single best offer

## Input format

The input includes **dynamic fields** — each segment (odpovědnost, auta, lodě, majetek, ...) has a different set of fields. The input tells you exactly which fields to extract and their types.

```json
POST /solve
{
  "segment": "auta",
  "fields_to_extract": [
    "Roční pojistné",
    "Povinné ručení – limit",
    "Havarijní pojištění – limit",
    "Spoluúčast havarijní",
    "..."
  ],
  "field_types": {
    "Roční pojistné": "number",
    "Povinné ručení – limit": "number",
    "Havarijní pojištění – limit": "string",
    "Spoluúčast havarijní": "number"
  },
  "offers": [
    {
      "id": "allianz",
      "insurer": "Allianz",
      "label": "Allianz",
      "documents": [
        {
          "filename": "Allianz_Redigováno.pdf",
          "ocr_text": "... OCR extracted text ...",
          "pdf_url": "https://storage.googleapis.com/..."
        }
      ]
    },
    {
      "id": "generali",
      "insurer": "Generali",
      "label": "Generali",
      "documents": [{"filename": "...", "ocr_text": "...", "pdf_url": "..."}]
    }
  ],
  "rfp": {
    "filename": "poptavka.pdf",
    "ocr_text": "... RFP/request text (if available) ...",
    "pdf_url": "https://storage.googleapis.com/..."
  }
}
```

### Key input fields

| Field | Description |
|-------|-------------|
| `segment` | Insurance segment name (e.g., odpovědnost, auta, lodě, majetek) |
| `fields_to_extract` | Ordered list of Czech field names you must extract from each offer's documents |
| `field_types` | Type of each field: `"number"` or `"string"` — determines how your answer is scored |
| `offers` | Array of insurance offers, each with one or more OCR-extracted documents |
| `offers[].id` | Unique offer identifier — pass through to output unchanged |
| `offers[].documents` | Array of documents for this offer, each with `filename`, `ocr_text`, and `pdf_url` |
| `offers[].documents[].ocr_text` | Full OCR-extracted text from the document — this is your primary data source |
| `offers[].documents[].pdf_url` | Direct URL to the original PDF (for verification/debugging) |
| `rfp` | Request for Proposal document (not always present) — contains the client's requirements |

### Important notes about the input

- **Field names are in Czech** and vary per segment (odpovědnost has 66 fields, auta has 17, lodě has 16, etc.)
- **Each offer may have multiple documents** — you need to extract fields from across all of them
- **Some segments include an RFP** — the RFP describes what the client needs, which helps with ranking
- **The `ocr_text` is already extracted** — you don't need to download or process PDFs unless you want to verify

## Expected output

For each offer, extract the fields listed in `fields_to_extract` into a `fields` dict:

```json
{
  "offers_parsed": [
    {
      "id": "allianz",
      "insurer": "Allianz",
      "fields": {
        "Roční pojistné": "125000",
        "Povinné ručení – limit": "100000000",
        "Havarijní pojištění – limit": "Nová cena",
        "Spoluúčast havarijní": "10000"
      }
    },
    {
      "id": "generali",
      "insurer": "Generali",
      "fields": {
        "Roční pojistné": "98000",
        "Povinné ručení – limit": "50000000",
        "Havarijní pojištění – limit": "Časová cena",
        "Spoluúčast havarijní": "5000"
      }
    }
  ],
  "ranking": ["allianz", "generali"],
  "best_offer_id": "allianz"
}
```

### Output structure

| Field | Description |
|-------|-------------|
| `offers_parsed` | Array of parsed offers — one per input offer |
| `offers_parsed[].id` | Must match the offer `id` from input exactly |
| `offers_parsed[].insurer` | Insurer name (pass through from input) |
| `offers_parsed[].fields` | Dict mapping field name (from `fields_to_extract`) → extracted value as string |
| `ranking` | Array of all offer IDs ordered from best to worst |
| `best_offer_id` | ID of the single best offer (should match `ranking[0]`) |

### Field value format

- **Number fields** (`field_types` = `"number"`): Return the numeric value as a string. Formats like `"50000000"`, `"50 000 000"`, `"CZK 150,000,000"` are all accepted — the scorer parses and compares numerically with ±10% tolerance.
- **String fields** (`field_types` = `"string"`): Return the text value. Scored with fuzzy string matching after normalization (case-insensitive, whitespace-collapsed). Similarity >50% gets partial credit.
- **Missing values**: Return `"N/A"` for fields you can't find in the documents.

## Training data segments

| Segment | Fields | Insurers | Docs | Description |
|---------|--------|----------|------|-------------|
| odpovědnost | 66 | 4 | 11 | Liability insurance — has RFP, multiple docs per insurer |
| auta | 17 | 4 | 4 | Vehicle fleet insurance — one doc per insurer |
| lodě | 16 | 3 | 5 | Boat/yacht insurance — multiple docs per insurer |

The evaluation uses different segments with different field counts and insurer counts. Your solution must handle any segment dynamically based on `fields_to_extract`.

## Scoring

| Component | Weight | Details |
|-----------|--------|---------|
| Field extraction | 60% | Per-field scores averaged across all offers and fields |
| Ranking order | 25% | Correct relative ordering of offers (partial credit for close positions) |
| Best offer ID | 15% | Exact match on the top pick |

### Scoring details

- **Number fields**: Exact match = 1.0, within ±10% = partial credit (0.5–1.0), within ±20% = 0.25, beyond = 0.0
- **String fields**: Exact match (after normalization) = 1.0, fuzzy ratio >50% = partial credit, below = 0.0
- **Ranking**: Each offer in the correct position = 1.0, displaced by N positions = `max(0, 1.0 - N*0.25)`
- **Best offer**: 1.0 if correct, 0.0 otherwise
- Only fields present in the expected output are scored — if a field is not extractable from a particular offer's documents, it may not be scored for that offer

## Local development

```bash
# Start the app + sidecar database
docker compose up --build

# Test your endpoint with a simple payload
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "segment": "auta",
    "fields_to_extract": ["Roční pojistné", "Povinné ručení – limit"],
    "field_types": {"Roční pojistné": "number", "Povinné ručení – limit": "number"},
    "offers": [
      {
        "id": "offer_a",
        "insurer": "Test Insurer",
        "label": "Test",
        "documents": [{"filename": "test.pdf", "ocr_text": "Roční pojistné: 125 000 Kč\nLimit povinného ručení: 50 000 000 Kč"}]
      }
    ]
  }'

# Check health
curl http://localhost:8080/

# Check token usage
curl http://localhost:8080/metrics
```

### Fetching training data

You can fetch training data from the training database to test against real inputs:

```python
import psycopg2, json

conn = psycopg2.connect(
    host="35.234.124.49", port=5432,
    user="hackathon", password="hackathon",
    dbname="hackathon_training"
)
cur = conn.cursor()
cur.execute("SELECT input, expected_output FROM training_data WHERE challenge_id = 1")
for input_data, expected_output in cur.fetchall():
    # input_data is a dict with segment, fields_to_extract, field_types, offers, rfp
    # expected_output is a dict with offers_parsed, field_types, ranking, best_offer_id
    print(f"Segment: {input_data['segment']}, Fields: {len(input_data['fields_to_extract'])}")
```

## Available tools

- **Gemini API** — use the pre-configured `gemini` object: `response = gemini.generate("your prompt")`. Token usage is tracked automatically.
- **PostgreSQL sidecar** — available at `DATABASE_URL` for caching. A `cache` table (key TEXT, value JSONB) is created on startup.

## Deployment

Push to your GitHub repo — Cloud Build will automatically build and deploy to Cloud Run.

## How ranking works

The ranking should order insurers from best to worst based on a **field-by-field comparison**.

For each field in `fields_to_extract`, compare the values across all insurers and determine which insurer has the "best" value for that field:
- **Coverage limits** (e.g., "Obecná odpovědnost limit"): higher is better
- **Deductibles / spoluúčast** (e.g., "Spoluúčast havarijní"): lower is better
- **Premium / pojistné** (e.g., "Roční pojistné"): lower is better
- **String fields**: compare qualitatively (e.g., broader territorial scope is better)

The insurer that "wins" the most fields (has the best value in the most categories) should be ranked first. The ranking is essentially: **count how many fields each insurer is the best at, and rank by that count** (highest count = best offer).

The `best_offer_id` should be the first insurer in your ranking (the one that wins the most field comparisons).

## Tips

- Use `fields_to_extract` and `field_types` from the input to build your Gemini extraction prompt dynamically — list all field names and tell Gemini which are numbers vs. strings
- Each offer may have multiple documents — concatenate all OCR texts for one offer before sending to Gemini
- Czech insurance documents use terms like "limit plnění", "spoluúčast", "pojistné", "pojistná částka"
- Numbers may appear as "50 000 000 Kč", "50.000.000,- Kč", "CZK 150,000,000" — the scorer handles normalization, just return the value as you find it
- For ranking: higher coverage limits + lower deductibles + lower premium = better offer. The RFP (if present) may specify what the client prioritizes
- Use the sidecar DB to cache parsed results and avoid redundant Gemini calls across evaluation rounds
- Return `"N/A"` for fields you can't find rather than guessing — wrong values score worse than missing ones
- The `pdf_url` field lets you download the original PDF if you want to use your own OCR or verify the text
