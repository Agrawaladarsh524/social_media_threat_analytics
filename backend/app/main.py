"""
OSINT-Guard API — FastAPI backend.

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title='OSINT-Guard API',
    description=(
        'Local exposure-risk intelligence for Twitter/X and LinkedIn OSINT data: '
        'deterministic rule-based scoring, spaCy NER + semantic content analysis, '
        'and unsupervised anomaly/persona detection. No external API key required. '
        'Optional Apify/OpenAI endpoints are available but not depended on.'
    ),
    version='3.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'] if settings.DEBUG else settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=not settings.DEBUG,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router, prefix='/api')


@app.get('/')
def root() -> dict[str, str]:
    return {'status': 'ok', 'service': 'osint-guard-api', 'docs': '/docs'}
