"""API routes. Paths match the Streamlit frontend in ../../frontend."""

from __future__ import annotations

import csv
import io
import json
import statistics
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from .models import LinkedInProfile, RiskSnapshot, TwitterProfile
from .schemas import CollectRequest, InsightsRequest
from .services.analytics import build_charts
from .services.apify_collect import collect_osint
from .services.local_ingest import ingest_linkedin_rows, ingest_twitter_rows
from .services.ml_models import clear_persisted_models, linkedin_ml_features, run_anomaly_and_clusters, twitter_ml_features
from .services.openai_insights import generate_ai_bundle
from .services.risk_engine import (
    MAX_CONTENT_SENSITIVITY,
    MAX_EXPOSURE_REACH,
    MAX_PII_EXPOSURE,
    MAX_PREDICTABILITY,
)
from .services.weight_analysis import analyse_weight_sensitivity
from .services.stats import build_profile_defaults

router = APIRouter()

PLATFORM_MODELS = {'twitter': TwitterProfile, 'linkedin': LinkedInProfile}


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
        body: dict[str, Any] = {'ok': False, 'error': 'openai_failed'}
        if settings.DEBUG:
            body['detail'] = traceback.format_exc()[-2000:]
        return JSONResponse(body, status_code=502)

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


@router.post('/ingest-local/')
async def ingest_local(
    file: UploadFile = File(...),
    platform: str = Form(...),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """
    CSV ingestion through the fully local pipeline: real stats + spaCy NER +
    deterministic rule-based risk scoring. No OpenAI key required.
    `platform` must be 'twitter' or 'linkedin'.
    """
    if platform not in PLATFORM_MODELS:
        return JSONResponse({'ok': False, 'error': "platform must be 'twitter' or 'linkedin'"}, status_code=400)

    raw = await file.read()
    try:
        content = raw.decode('utf-8-sig')
        rows = list(csv.DictReader(io.StringIO(content)))
    except Exception as e:  # noqa: BLE001 — surface parse errors to the client
        return JSONResponse({'ok': False, 'error': f'Invalid CSV: {e}'}, status_code=400)

    if not rows:
        return JSONResponse({'ok': False, 'error': 'CSV has no rows.'}, status_code=400)

    if platform == 'twitter':
        quality = ingest_twitter_rows(db, rows)
    else:
        quality = ingest_linkedin_rows(db, rows)

    count = quality['ingested']
    return JSONResponse({
        'ok': True,
        'platform': platform,
        'profiles_ingested': count,
        'data_quality': quality,
        'message': f'{count} {platform} profile(s) ingested via the local rule-based/NER pipeline (no AI key used).',
    })


@router.post('/recompute-models/')
def recompute_models(platform: str = 'twitter', force_retrain: bool = False, db: Session = Depends(get_db)) -> JSONResponse:
    """Refit Isolation Forest (anomaly) + KMeans (clustering) across every
    stored profile for a platform, and persist the results. By default,
    KMeans cluster IDs are matched back to the previous run's so persona
    identity survives the refit — pass force_retrain=true to reset that and
    get a fresh, unmatched clustering baseline instead."""
    if platform not in PLATFORM_MODELS:
        return JSONResponse({'ok': False, 'error': "platform must be 'twitter' or 'linkedin'"}, status_code=400)

    model = PLATFORM_MODELS[platform]
    feature_fn = twitter_ml_features if platform == 'twitter' else linkedin_ml_features
    profiles = list(db.scalars(select(model)))

    anomaly_by_id, cluster_by_id, diagnostics = run_anomaly_and_clusters(
        profiles, feature_fn, platform, force_retrain=force_retrain,
    )
    for p in profiles:
        p.anomaly_score = anomaly_by_id.get(p.id)
        p.cluster_id = cluster_by_id.get(p.id)
    db.commit()

    return JSONResponse({'ok': True, 'platform': platform, 'diagnostics': diagnostics})


@router.get('/bucket-contributions/')
def bucket_contributions(platform: str = 'twitter', db: Session = Depends(get_db)) -> JSONResponse:
    """
    Measured evidence for the scoring weights: how many points each bucket
    actually contributes across the stored population, versus the maximum it
    is allowed to contribute.

    This exists because "why 40/25/20/15?" is otherwise unanswerable. Rather
    than asserting the weights are right, it reports what they do in practice
    — including the uncomfortable parts, e.g. a bucket that is allocated a
    large share of the budget but is zero for most real profiles, which caps
    the score range below 100 and makes the top tier unreachable.
    """
    if platform not in PLATFORM_MODELS:
        return JSONResponse({'ok': False, 'error': "platform must be 'twitter' or 'linkedin'"}, status_code=400)

    profiles = list(db.scalars(select(PLATFORM_MODELS[platform])))
    if not profiles:
        return JSONResponse({'ok': False, 'error': 'No profiles stored for this platform.'})

    caps = {
        'pii_exposure': MAX_PII_EXPOSURE,
        'predictability': MAX_PREDICTABILITY,
        'content_sensitivity': MAX_CONTENT_SENSITIVITY,
        'exposure_reach': MAX_EXPOSURE_REACH,
    }
    per_bucket: dict[str, list[float]] = {c: [] for c in caps}
    for p in profiles:
        totals = dict.fromkeys(caps, 0.0)
        for factor in (p.risk_factors or []):
            cat = factor.get('category')
            if cat in totals:
                totals[cat] += float(factor.get('points') or 0)
        for cat, val in totals.items():
            per_bucket[cat].append(val)

    n = len(profiles)
    rows = []
    for cat, cap in caps.items():
        vals = per_bucket[cat]
        mean = statistics.mean(vals)
        rows.append({
            'bucket': cat,
            'max_points': cap,
            'weight_share_pct': round(cap / sum(caps.values()) * 100, 1),
            'mean_points': round(mean, 2),
            'stdev_points': round(statistics.pstdev(vals), 2) if n > 1 else 0.0,
            'min_points': round(min(vals), 1),
            'max_observed': round(max(vals), 1),
            'pct_scoring_zero': round(sum(1 for v in vals if v == 0) / n * 100, 1),
            'budget_utilisation_pct': round(mean / cap * 100, 1) if cap else 0.0,
        })

    observed_max = round(sum(r['max_observed'] for r in rows), 1)
    theoretical_max = float(sum(caps.values()))

    # A bucket that is almost never awarded still consumes its share of the
    # 0-100 budget, which lowers the score ceiling the population can actually
    # reach and can put a tier out of range entirely. Surfaced rather than left
    # for someone to discover from the fact that no profile is ever Critical.
    dormant = [
        {'bucket': r['bucket'], 'max_points': r['max_points'], 'pct_scoring_zero': r['pct_scoring_zero']}
        for r in rows if r['pct_scoring_zero'] >= 75
    ]
    unreachable_headroom = sum(d['max_points'] for d in dormant)

    return JSONResponse({
        'ok': True,
        'platform': platform,
        'n_profiles': n,
        'buckets': rows,
        'theoretical_max_score': theoretical_max,
        'observed_max_score': max((p.risk_score or 0) for p in profiles),
        'sum_of_per_bucket_maxima': observed_max,
        'dormant_buckets': dormant,
        'practical_ceiling': round(theoretical_max - unreachable_headroom, 1),
        'weight_sensitivity': analyse_weight_sensitivity(profiles),
        'note': (
            'Spread (stdev) is what moves a profile up or down the ranking — a bucket with '
            'a large cap but near-zero spread contributes little to how profiles are ordered, '
            'however much budget it is allocated.'
        ),
    })


@router.get('/profiles-summary/')
def profiles_summary(platform: str = 'twitter', db: Session = Depends(get_db)) -> JSONResponse:
    """Lightweight, uncapped list of numeric/risk fields for every stored
    profile on a platform — used by the frontend to build interactive
    charts client-side (Plotly) without pulling heavy JSON blob columns."""
    if platform == 'linkedin':
        rows = list(db.scalars(select(LinkedInProfile)))
        data = [
            {
                'username': p.public_identifier,
                'full_name': p.full_name,
                'risk_score': p.risk_score,
                'risk_label': p.risk_label,
                'anomaly_score': p.anomaly_score,
                'cluster_id': p.cluster_id,
                'location_city': p.location_city,
                'location_state': p.location_state,
                'location_country': p.location_country,
                'current_employer': p.current_employer,
                'employer_count': p.employer_count,
                'connections_count': p.connections_count,
                'follower_count': p.follower_count,
                'open_to_work': p.open_to_work,
                'is_hiring': p.is_hiring,
            }
            for p in rows
        ]
    else:
        rows = list(db.scalars(select(TwitterProfile)))
        data = [
            {
                'username': p.username,
                'risk_score': p.risk_score,
                'risk_label': p.risk_label,
                'anomaly_score': p.anomaly_score,
                'cluster_id': p.cluster_id,
                'total_tweets': p.total_tweets,
                'total_likes': p.total_likes,
                'total_views': p.total_views,
                'avg_engagement': p.avg_engagement,
                'followers_count': p.followers_count,
                'categories': p.categories,
            }
            for p in rows
        ]
    return JSONResponse({'ok': True, 'platform': platform, 'total': len(data), 'profiles': data})


@router.get('/analytics/')
def analytics(platform: str = 'twitter', db: Session = Depends(get_db)) -> JSONResponse:
    model = PLATFORM_MODELS.get(platform, TwitterProfile)
    profiles = list(db.scalars(select(model)))
    return JSONResponse(build_charts(profiles, platform=platform))


@router.get('/check-db/')
def check_db(platform: str = 'twitter', db: Session = Depends(get_db)) -> JSONResponse:
    """Real data + local risk-engine output for the 50 most recent profiles."""
    model = PLATFORM_MODELS.get(platform, TwitterProfile)
    profiles = list(
        db.scalars(select(model).order_by(model.scanned_at.desc()).limit(50))
    )
    total = db.scalar(select(func.count()).select_from(model)) or 0

    if platform == 'linkedin':
        data = [
            {
                'username': p.public_identifier,
                'full_name': p.full_name,
                'scanned_at': _fmt(p.scanned_at),
                'risk_score': p.risk_score,
                'risk_label': p.risk_label,
                'anomaly_score': p.anomaly_score,
                'cluster_id': p.cluster_id,
                'tweet_stats': {
                    'headline': p.headline,
                    'location': ', '.join(x for x in [p.location_city, p.location_state, p.location_country] if x),
                    'current_employer': p.current_employer,
                    'employer_count': p.employer_count,
                    'connections_count': p.connections_count,
                    'follower_count': p.follower_count,
                    'open_to_work': p.open_to_work,
                    'is_hiring': p.is_hiring,
                },
                'risk_signals': p.inference_rows,
                'pattern_of_life': [],
            }
            for p in profiles
        ]
    else:
        data = [
            {
                'username': p.username,
                'scanned_at': _fmt(p.scanned_at),
                'bundle_generated_at': _fmt(p.bundle_generated_at),
                'risk_score': p.risk_score,
                'risk_label': p.risk_label,
                'anomaly_score': p.anomaly_score,
                'cluster_id': p.cluster_id,
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


@router.get('/profile-detail/{platform}/{identifier}/')
def profile_detail(platform: str, identifier: str, db: Session = Depends(get_db)) -> JSONResponse:
    """Full risk_factors breakdown for one profile — the 'why this score' view."""
    model = PLATFORM_MODELS.get(platform)
    if model is None:
        return JSONResponse({'ok': False, 'error': "platform must be 'twitter' or 'linkedin'"}, status_code=400)

    key_col = model.username if platform == 'twitter' else model.public_identifier
    profile = db.scalar(select(model).where(key_col.ilike(identifier)))
    if not profile:
        return JSONResponse({'ok': False, 'error': 'Profile not found in database.'})

    return JSONResponse({
        'ok': True,
        'profile': {
            'identifier': identifier,
            'risk_score': profile.risk_score,
            'risk_label': profile.risk_label,
            'risk_factors': profile.risk_factors,
            'anomaly_score': profile.anomaly_score,
            'cluster_id': profile.cluster_id,
            'inference_rows': profile.inference_rows or [],
        },
    })


@router.get('/snapshots/{platform}/{identifier}/')
def snapshot_history(platform: str, identifier: str, db: Session = Depends(get_db)) -> JSONResponse:
    """Every past scoring event for one profile, oldest first — the audit
    trail behind 'has this profile's exposure score changed.' Real history,
    not a synthetic backfill: a new row is written every time the profile
    is (re-)scored, so this only has more than one entry once you've
    actually re-ingested (e.g. after a scoring-methodology change)."""
    if platform not in PLATFORM_MODELS:
        return JSONResponse({'ok': False, 'error': "platform must be 'twitter' or 'linkedin'"}, status_code=400)

    rows = list(db.scalars(
        select(RiskSnapshot)
        .where(RiskSnapshot.platform == platform, RiskSnapshot.profile_identifier.ilike(identifier))
        .order_by(RiskSnapshot.scanned_at.asc())
    ))
    return JSONResponse({
        'ok': True,
        'count': len(rows),
        'snapshots': [
            {
                'scanned_at': _fmt(r.scanned_at),
                'risk_score': r.risk_score,
                'risk_label': r.risk_label,
            }
            for r in rows
        ],
    })


@router.api_route('/clear-db/', methods=['GET', 'POST', 'DELETE'])
def clear_db(platform: str = 'twitter', db: Session = Depends(get_db)) -> JSONResponse:
    """Wipe every stored profile for a platform so it can be re-ingested from
    scratch. Also clears the persisted scaler/cluster-centroid state for that
    platform — a wiped-and-reingested population should get a fresh baseline
    rather than being scaled against a scaler fit on data that no longer exists."""
    model = PLATFORM_MODELS.get(platform, TwitterProfile)
    count = db.query(model).delete()
    db.commit()
    clear_persisted_models(platform)
    return JSONResponse({
        'ok': True,
        'message': f'Successfully wiped {count} {platform} profiles. Table is now empty.',
    })


@router.get('/profile/{username}/')
def get_profile(username: str, db: Session = Depends(get_db)) -> JSONResponse:
    """Twitter-only quick lookup, kept for backward compatibility with the
    Target Scan tab. Use /api/profile-detail/{platform}/{identifier}/ for
    the full risk-factors breakdown on either platform."""
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
def usernames(platform: str = 'twitter', db: Session = Depends(get_db)) -> JSONResponse:
    if platform == 'linkedin':
        return JSONResponse({'ok': True, 'usernames': list(db.scalars(select(LinkedInProfile.public_identifier)))})
    return JSONResponse({'ok': True, 'usernames': list(db.scalars(select(TwitterProfile.username)))})
