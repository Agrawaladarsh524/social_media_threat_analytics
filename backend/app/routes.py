"""API routes. Paths match the Streamlit frontend in ../../frontend."""

from __future__ import annotations

import csv
import io
import json
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from .models import TwitterProfile
from .schemas import CollectRequest, InsightsRequest
from .services.analytics import build_charts
from .services.apify_collect import collect_osint
from .services.openai_insights import generate_ai_bundle
from .services.stats import build_profile_defaults

router = APIRouter()


def _fmt(dt: datetime | None) -> str | None:
    return dt.strftime('%Y-%m-%d %H:%M:%S') if dt else None


def _upsert_profile(db: Session, username: str, defaults: dict[str, Any]) -> TwitterProfile:
    profile = db.scalar(select(TwitterProfile).where(TwitterProfile.username == username))
    if profile is None:
        profile = TwitterProfile(username=username)
        db.add(profile)
    for key, value in defaults.items():
        setattr(profile, key, value)
    db.commit()
    return profile


def _extract_username(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    if item.get('username'):
        return str(item['username']).strip()
    author = item.get('author')
    if isinstance(author, dict) and author.get('userName'):
        return str(author['userName']).strip()
    known = {'username', 'user_name', 'screen_name', 'handle', 'twitter_handle', 'author', 'user', 'twitter'}
    for k, v in item.items():
        if k.lower().strip() in known and v:
            return str(v).strip()
    for k, v in item.items():
        if ('user' in k.lower() or 'handle' in k.lower()) and v:
            return str(v).strip()
    return None


@router.get('/health/')
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'osint-guard-api'}


@router.post('/collect/')
def collect(payload: CollectRequest) -> JSONResponse:
    """Scrape a live Twitter handle through Apify."""
    if not settings.APIFY_API_TOKEN:
        return JSONResponse({'ok': False, 'error': 'APIFY Key missing'}, status_code=503)

    data, errors = collect_osint(
        settings.APIFY_API_TOKEN,
        twitter=payload.twitter,
        actor_twitter=settings.APIFY_ACTOR_TWITTER,
    )
    return JSONResponse({'ok': True, 'data': data, 'errors': errors})


@router.post('/insights/')
def insights(payload: InsightsRequest, db: Session = Depends(get_db)) -> JSONResponse:
    """Run the OpenAI risk bundle over collected data, with a DB-backed cache."""
    if not settings.OPENAI_API_KEY:
        return JSONResponse({'ok': False, 'error': 'OpenAI API key missing'}, status_code=503)

    username = payload.username
    if username:
        existing = db.scalar(select(TwitterProfile).where(TwitterProfile.username == username))
        if existing and existing.ai_bundle and existing.bundle_generated_at:
            generated = existing.bundle_generated_at
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - generated < timedelta(hours=settings.AI_CACHE_TTL_HOURS):
                return JSONResponse({'ok': True, 'cached': True, **existing.ai_bundle})

    try:
        bundle = generate_ai_bundle(
            settings.OPENAI_API_KEY,
            settings.OPENAI_MODEL,
            payload.datasets,
            timeout=settings.OPENAI_REQUEST_TIMEOUT,
        )
    except Exception:
        return JSONResponse(
            {'ok': False, 'error': 'openai_failed', 'detail': traceback.format_exc()[-2000:]},
            status_code=502,
        )

    if username:
        try:
            _upsert_profile(db, username, build_profile_defaults(bundle, payload.datasets))
        except Exception as e:  # noqa: BLE001 — analysis already succeeded; don't lose it
            db.rollback()
            if settings.DEBUG:
                print('Failed to save profile:', e, flush=True)

    return JSONResponse({'ok': True, **bundle})


@router.post('/upload-bulk/')
async def upload_bulk(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """Ingest a CSV/JSON tweet export: group rows per user, then AI-score each user."""
    raw = await file.read()
    try:
        content = raw.decode('utf-8')
        if (file.filename or '').endswith('.csv'):
            data_list = list(csv.DictReader(io.StringIO(content)))
        else:
            data_list = json.loads(content)
        if not isinstance(data_list, list):
            return JSONResponse({'error': 'File must contain a list of user profiles.'}, status_code=400)
    except Exception as e:  # noqa: BLE001 — surface parse errors to the client
        return JSONResponse({'error': f'Invalid file format: {e}'}, status_code=400)

    if not settings.OPENAI_API_KEY:
        return JSONResponse({'error': 'OpenAI API key missing'}, status_code=503)

    user_tweets: dict[str, list] = {}
    skipped_count = 0
    for item in data_list:
        username = _extract_username(item)
        if not username:
            skipped_count += 1
            continue
        user_tweets.setdefault(username, []).append(item)

    unique_users = list(user_tweets)
    total = len(unique_users)
    print(f'[Upload] {len(data_list)} rows -> {total} unique users. Skipped: {skipped_count}.', flush=True)

    processed_count = 0
    for i, username in enumerate(unique_users, start=1):
        datasets = {'twitter': user_tweets[username]}
        try:
            print(f'[{i}/{total}] Analyzing @{username} ({len(datasets["twitter"])} tweets)...', flush=True)
            bundle = generate_ai_bundle(
                settings.OPENAI_API_KEY,
                settings.OPENAI_MODEL,
                datasets,
                timeout=settings.OPENAI_REQUEST_TIMEOUT,
            )
            _upsert_profile(db, username, build_profile_defaults(bundle, datasets))
            processed_count += 1
            print(f'[{i}/{total}] Saved @{username} — Risk: {bundle.get("executive", {}).get("riskScore", 0)}', flush=True)
        except Exception as e:  # noqa: BLE001 — one bad user must not abort the batch
            db.rollback()
            print(f'[{i}/{total}] Error analyzing @{username}: {e}', flush=True)

    return JSONResponse({
        'ok': True,
        'message': (
            f'{processed_count} profiles ingested from {len(data_list)} rows '
            f'({total} unique users). Skipped rows: {skipped_count}'
        ),
    })


@router.get('/analytics/')
def analytics(db: Session = Depends(get_db)) -> JSONResponse:
    profiles = list(db.scalars(select(TwitterProfile)))
    return JSONResponse(build_charts(profiles))


@router.get('/check-db/')
def check_db(db: Session = Depends(get_db)) -> JSONResponse:
    """Real data + AI risk signals for the 50 most recent profiles. No AI prose."""
    profiles = list(
        db.scalars(select(TwitterProfile).order_by(TwitterProfile.scanned_at.desc()).limit(50))
    )
    total = db.scalar(select(func.count()).select_from(TwitterProfile)) or 0

    data = [
        {
            'username': p.username,
            'scanned_at': _fmt(p.scanned_at),
            'bundle_generated_at': _fmt(p.bundle_generated_at),
            'risk_score': p.risk_score,
            'risk_label': p.risk_label,
            'tweet_stats': {
                'total_tweets':   p.total_tweets,
                'date_first':     p.date_first_tweet,
                'date_last':      p.date_last_tweet,
                'categories':     p.categories,
                'total_likes':    p.total_likes,
                'total_retweets': p.total_retweets,
                'total_replies':  p.total_replies,
                'total_quotes':   p.total_quotes,
                'total_views':    p.total_views,
                'avg_engagement': p.avg_engagement,
            },
            'risk_signals': p.inference_rows,
            'pattern_of_life': p.pattern_of_life,
        }
        for p in profiles
    ]
    return JSONResponse({'total_db_records': total, 'profiles': data})


@router.api_route('/clear-db/', methods=['GET', 'POST', 'DELETE'])
def clear_db(db: Session = Depends(get_db)) -> JSONResponse:
    """Wipe every stored profile so a dataset can be re-ingested from scratch."""
    count = db.query(TwitterProfile).delete()
    db.commit()
    return JSONResponse({
        'ok': True,
        'message': f'Successfully wiped {count} profiles. Database is now completely empty.',
    })


@router.get('/profile/{username}/')
def get_profile(username: str, db: Session = Depends(get_db)) -> JSONResponse:
    profile = db.scalar(
        select(TwitterProfile).where(TwitterProfile.username.ilike(username))
    )
    if not profile:
        return JSONResponse({'ok': False, 'error': 'Profile not found in database.'})

    return JSONResponse({
        'ok': True,
        'profile': {
            'username': profile.username,
            'risk_score': profile.risk_score,
            'risk_label': profile.risk_label,
            'total_tweets': profile.total_tweets,
            'avg_engagement': profile.avg_engagement,
            'categories': (profile.categories or [])[:5],
            'inference_rows': profile.inference_rows or [],
        },
    })


@router.get('/usernames/')
def usernames(db: Session = Depends(get_db)) -> JSONResponse:
    return JSONResponse({'ok': True, 'usernames': list(db.scalars(select(TwitterProfile.username)))})
