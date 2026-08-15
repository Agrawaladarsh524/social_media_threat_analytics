<div align="center">

# OSINT-Guard

**Turn a public Twitter/X or LinkedIn footprint into an explainable exposure-risk score — fully local, no LLM required.**

Ingest a Twitter/X or LinkedIn export, run it through a deterministic rule-based
risk engine plus local NLP (spaCy NER) and unsupervised ML (Isolation Forest,
KMeans), and get back an itemized 0-100 risk score, extracted PII signals,
anomaly flags, and behavioral clusters — all visualized in an interactive
dashboard. Every point in every score is traceable to a specific, auditable
signal — not a black-box LLM guess.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00)](https://www.sqlalchemy.org/)

</div>

---

## Overview

Security teams and OSINT analysts routinely need to answer one question about a
target: *how much can an attacker infer about this person from what they've
already made public?* OSINT-Guard automates that assessment end-to-end — ingest,
engineer features, score, detect anomalies, cluster, and visualize — across two
independent public-data sources at once.

The core intelligence pipeline is **100% local**: a weighted rule-based scoring
engine, spaCy named-entity recognition, and scikit-learn's Isolation Forest /
KMeans all run on-device with no external API calls, no API key, and no cost.
An optional OpenAI + Apify path (live scraping and LLM-generated summaries)
still exists for users who want to add those keys later, but nothing in the
core pipeline depends on it.

> **Intended use:** authorized security research, red-team reconnaissance
> exercises, and OSINT-awareness training. Only assess accounts you're
> authorized to assess.

## Key Features

- 🧮 **Explainable risk engine** — a transparent, weighted scoring system (PII
  exposure, behavioral predictability, content/status sensitivity, exposure
  reach) with every point itemized and traceable to a real signal.
- 🕵️ **Local NLP extraction** — spaCy NER pulls real locations, organizations,
  and PII-adjacent entities out of tweet text, bios, and LinkedIn headlines —
  no LLM call involved.
- 🚨 **Unsupervised anomaly & persona detection** — Isolation Forest flags
  statistically anomalous accounts; KMeans clusters profiles into behavioral
  personas. Both fit fresh on your data, no labeled training set required.
- 🌐 **Two independent OSINT sources, one platform** — Twitter/X (engagement,
  posting patterns, tweet content) and LinkedIn (disclosed location, employer
  history, job-seeking status) share one scoring architecture via pluggable
  feature extractors.
- 📊 **Interactive, purposeful analytics** — every chart reflects a real
  computed signal (risk distribution, anomaly scores, cluster sizes, top
  disclosed locations) instead of a static placeholder.
- 🔓 **Optional AI path, not required** — live Apify scraping + OpenAI-generated
  summaries are still available behind an explicit "Advanced" toggle for users
  who add those API keys later.

## Architecture

```mermaid
flowchart LR
    CSV[Twitter/X or LinkedIn\nCSV export] --> ING[Local ingestion pipeline]
    ING --> STATS[stats.py\nreal engagement metrics]
    ING --> NER[nlp_features.py\nspaCy NER + keyword rules]
    STATS --> ENGINE[risk_engine.py\nweighted rule-based scorer]
    NER --> ENGINE
    ENGINE --> DB[(SQLite / Postgres)]
    DB --> ML[ml_models.py\nIsolation Forest + KMeans]
    ML --> DB
    DB --> API[FastAPI service]
    API --> ST[Streamlit dashboard]
    U([Analyst]) --> ST

    subgraph Optional, not required
      APIFY[Apify scraper] -.-> API
      OAI[OpenAI LLM] -.-> API
    end
```

The frontend never talks to the database or any external service directly —
every call is proxied through the FastAPI service, which is the single source
of truth for stored profiles and the only component holding credentials.

## Tech Stack

| Layer | Technology | Role |
| --- | --- | --- |
| UI | **Streamlit** + **Plotly** | Interactive dashboard — lookup, ingestion, analytics, model insights, profile browser |
| API | **FastAPI** + **Uvicorn** | Async REST API, request validation, routing |
| Local NLP | **spaCy** (`en_core_web_sm`) | Named-entity recognition for PII/location extraction — no external calls |
| Unsupervised ML | **scikit-learn** (Isolation Forest, KMeans) | Anomaly detection + behavioral clustering, fit fresh on each ingestion |
| Risk scoring | Custom weighted engine (`risk_engine.py`) | Deterministic, itemized 0-100 score — no training data needed |
| Data wrangling | **pandas**, **numpy** | Feature engineering across both platforms |
| Validation & config | **Pydantic v2**, `pydantic-settings` | Typed request models, `.env`-driven settings |
| Persistence | **SQLAlchemy 2.0** | ORM over SQLite (default) or any SQL database via `DATABASE_URL` |
| Testing | **pytest** | Unit tests for the risk engine and NLP extractors, FastAPI `TestClient` route tests |
| *Optional* | **OpenAI API**, **Apify** client | Live scraping + LLM-generated summaries, behind an explicit opt-in — not required |

## Project Structure

```text
osint-guard/
├── backend/                        FastAPI service
│   ├── app/
│   │   ├── main.py                 App factory, CORS, startup table creation
│   │   ├── config.py               Settings loaded from .env
│   │   ├── database.py             SQLAlchemy engine / session / declarative base
│   │   ├── models.py               TwitterProfile + LinkedInProfile ORM models
│   │   ├── schemas.py              Pydantic request models
│   │   ├── routes.py               All /api/* endpoints
│   │   └── services/
│   │       ├── risk_engine.py      Platform-agnostic weighted risk-scoring core
│   │       ├── nlp_features.py     spaCy NER + regex + feature extraction (Twitter & LinkedIn)
│   │       ├── ml_models.py        Isolation Forest (anomaly) + KMeans (clustering)
│   │       ├── local_ingest.py     CSV -> features -> score -> persist, no API key
│   │       ├── stats.py            Deterministic tweet-engagement metrics
│   │       ├── analytics.py        Seaborn/matplotlib chart export (PNG)
│   │       ├── apify_collect.py    Optional: Apify actor runs → normalized tweets
│   │       └── openai_insights.py  Optional: chained OpenAI calls → LLM risk bundle
│   ├── tests/                      pytest suite (risk engine, NLP, routes)
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/                       Streamlit UI
    ├── streamlit_app.py            Target Lookup · Bulk Ingestion · Analytics · Model Insights · Database
    ├── .streamlit/config.toml      Native theme (no custom HTML/CSS)
    └── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.11+
- No API keys required for the core pipeline.
- *(Optional)* An [Apify](https://apify.com/) API token + [OpenAI](https://platform.openai.com/) API key, only if you want the live-scrape/LLM-summary path.

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # one-time local NLP model download
cp .env.example .env                      # optional: only needed for the Apify/OpenAI path
uvicorn app.main:app --reload             # http://127.0.0.1:8000
```

Database tables are created automatically on first run. Interactive API docs
are served at `/docs` (Swagger) and `/redoc`.

### 2. Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py  # http://localhost:8501
```

### 3. Ingest a dataset

In the **Bulk Ingestion** tab, pick a platform (Twitter/X or LinkedIn), upload
a CSV export, and click **"Ingest locally (no AI, no API key)"**. Then use
**Model Insights → Recompute anomaly/cluster models** to fit Isolation Forest
and KMeans across the newly ingested profiles.

> The Apify/OpenAI path is entirely optional — everything else (ingestion,
> scoring, NER, anomaly detection, clustering, analytics) works with zero API
> keys configured.

## Environment Variables

Configured in `backend/.env` (see `backend/.env.example`) — **only relevant if
you're using the optional live-scrape/LLM path**; the local pipeline needs none of these:

| Variable | Default | Description |
| --- | --- | --- |
| `DEBUG` | `true` | Enables permissive CORS for local dev; set `false` in production |
| `CORS_ALLOWED_ORIGINS` | `["http://localhost:8501", ...]` | Browser origins allowed to call the API when `DEBUG=false` |
| `DATABASE_URL` | SQLite file in `backend/` | Any SQLAlchemy-compatible connection string (Postgres, MySQL, etc.) |
| `APIFY_API_TOKEN` | *(empty)* | Optional — required only for live Twitter/X scraping |
| `APIFY_ACTOR_TWITTER` | `dy7gIgPRMhrOrfW0f` | Apify actor ID used for Twitter/X collection |
| `OPENAI_API_KEY` | *(empty)* | Optional — required only for the LLM-generated risk bundle |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat Completions model used for the optional AI path |
| `OPENAI_REQUEST_TIMEOUT` | `30` | Per-request timeout, in seconds |
| `AI_CACHE_TTL_HOURS` | `24` | How long a generated AI bundle is cached before re-analysis |

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health/` | Liveness probe |
| `POST` | `/api/ingest-local/` | **Primary path.** CSV -> local risk engine, per platform (`twitter` \| `linkedin`), no API key |
| `POST` | `/api/recompute-models/` | Refit Isolation Forest + KMeans across all stored profiles for a platform |
| `GET` | `/api/profiles-summary/` | Uncapped, lightweight profile list for client-side charting |
| `GET` | `/api/analytics/` | Render static PNG charts (matplotlib/seaborn export) |
| `GET` | `/api/check-db/` | Fetch the 50 most recently scanned profiles |
| `GET` | `/api/profile-detail/{platform}/{identifier}/` | Full itemized risk-factor breakdown for one profile |
| `GET` | `/api/usernames/` | List every stored handle/identifier for a platform |
| `GET / POST / DELETE` | `/api/clear-db/` | Wipe stored profiles for a platform |
| `POST` | `/api/collect/` | *Optional.* Scrape a handle's public tweets via Apify |
| `POST` | `/api/insights/` | *Optional.* Run the OpenAI risk bundle, DB-cached |
| `POST` | `/api/upload-bulk/` | *Optional.* CSV/JSON ingestion scored via chained OpenAI calls |
| `GET` | `/api/profile/{username}/` | Legacy Twitter-only quick lookup |

Full request/response schemas: `http://127.0.0.1:8000/docs`

## Usage

The dashboard is organized into five tabs, with a platform switcher (Twitter/X
vs. LinkedIn) at the top that scopes everything below it:

1. **Target Lookup** — look up an already-ingested profile and see its full
   itemized risk breakdown. An optional "Advanced" section exposes the live
   Apify + OpenAI scan for users who've added those keys.
2. **Bulk Ingestion** — upload a CSV export and ingest it through the local
   pipeline in one click. The OpenAI-scored path is available as a collapsed,
   optional alternative.
3. **Analytics** — interactive Plotly charts (risk distribution, anomaly
   scores, cluster sizes, top disclosed locations, engagement) built from the
   local pipeline's output, plus a static-PNG export option.
4. **Model Insights** — Isolation Forest / KMeans diagnostics (silhouette
   score, flagged anomalies, cluster profiles) and a plain-language
   explanation of exactly how the risk score is computed.
5. **Database** — browse, inspect, export (CSV), or clear stored profiles per
   platform.

## Testing

```bash
cd backend
pip install -r requirements.txt   # includes pytest
pytest tests/ -v
```

Covers the risk engine (score bounds, tier ordering, sample-size confidence
weighting), the NLP feature extractors (entity extraction, keyword matching),
and the ingestion/analytics routes end to end via FastAPI's `TestClient`
against an isolated temp database.

## Roadmap

- [ ] Supervised distillation model (RandomForestRegressor trained on the
      rule engine's own output) with SHAP-based feature importance
- [ ] Containerization (Dockerfile + docker-compose for both services)
- [ ] Authentication / role-based access for multi-analyst deployments
- [ ] CI pipeline (lint, type-check, test on push)
- [ ] Cross-platform identity resolution (linking the same person across
      Twitter and LinkedIn when both are present)

## Contributing

Issues and pull requests are welcome. For substantial changes, please open an
issue first to discuss what you'd like to change.

## License

No license has been specified yet — all rights reserved by default. Add a
`LICENSE` file (MIT, Apache-2.0, etc.) if you intend this project to be reused or
distributed.
