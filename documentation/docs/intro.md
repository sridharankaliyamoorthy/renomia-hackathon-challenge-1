---
sidebar_position: 1
title: Overview
---

# Renomia Hackathon — Challenge 1

**Insurance offer comparison** pipeline: OCR insurance PDFs → **Gemini 2.5 Flash** extraction → deterministic **ranking** → best-offer selection.

| | |
|---|---|
| **Team** | MrRooT.Ai (Team 15) |
| **Repository** | [github.com/sridharankaliyamoorthy/renomia-hackathon-challenge-1](https://github.com/sridharankaliyamoorthy/renomia-hackathon-challenge-1) |
| **Stack** | Python, FastAPI, PostgreSQL, Google Cloud Run, Gemini 2.5 Flash, pdfplumber |

## Score (hackathon)

**Overall: 0.589** — auta **0.853** · lodě **0.496** · odpovědnost **0.419**

Ranking and best-offer detection were correct across segments; **field extraction** was the main bottleneck (nested tables, scanned PDFs, long documents).

## What this documentation covers

- [**Architecture**](./architecture.md) — end-to-end pipeline and design decisions
- [**Codebase**](./codebase.md) — main Python modules and responsibilities
- [**Operations**](./operations.md) — local Docker setup, API usage, deployment
- [**Results & roadmap**](./results.md) — metrics, known limitations, next steps

Presentation assets (one-pager, pipeline diagram, slideshow PDF) live under `presentation/` in the repo.
