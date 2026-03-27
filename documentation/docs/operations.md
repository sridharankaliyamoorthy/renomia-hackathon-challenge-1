---
sidebar_position: 4
title: Operations
---

# Local development

From the repository root:

```bash
docker compose up --build
```

This starts the FastAPI app and the PostgreSQL sidecar.

## API

### Health

```bash
curl http://localhost:8080/
```

### Solve (example)

```bash
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "segment": "auta",
    "fields_to_extract": ["Roční pojistné"],
    "field_types": {"Roční pojistné": "number"},
    "offers": [{"id": "test", "insurer": "Test", "label": "Test", "documents": [{"filename": "test.pdf", "ocr_text": "Roční pojistné: 125 000 Kč"}]}]
  }'
```

### Metrics (token usage)

```bash
curl http://localhost:8080/metrics
```

## Configuration

Copy `.env.example` to `.env` and fill in secrets (e.g. Gemini API keys) as required by your deployment. **Never commit real secrets.**

## Deployment

Push to GitHub: **Cloud Build** builds the Docker image and deploys to **Cloud Run** (see `cloudbuild.yaml`, `service.yaml`, `Dockerfile` in the repo).

## Documentation site

This Docusaurus site lives in the `documentation/` folder:

```bash
cd documentation
npm install
npm start
```

Build static files with `npm run build`; output is in `documentation/build/`.
