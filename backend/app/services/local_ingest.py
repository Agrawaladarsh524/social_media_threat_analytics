"""
Local (no-API) CSV ingestion pipelines for Twitter and LinkedIn exports.

Runs the full pipeline end to end: parse rows -> compute real stats ->
engineer features -> spaCy NER -> deterministic rule-based risk score ->
persist. No OpenAI key required — this is the primary ingestion path.
`/api/upload-bulk/` (the OpenAI path) still exists separately for users who
add a key later.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from ..models import LinkedInProfile, TwitterProfile
from .nlp_features import (
    extract_linkedin_features,
    extract_twitter_features,
    unpack_twitter_profile_meta,
)
from .risk_engine import score_linkedin_profile, score_twitter_profile
from .stats import compute_tweet_stats


def _to_int(v: Any) -> int:
    try:
        return int(float(v or 0))
    except (TypeError, ValueError):
        return 0


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, '') else None
    except (TypeError, ValueError):
        return None


def _to_bool(v: Any) -> bool:
    return str(v).strip().lower() in {'1', 'true', 'yes'}


def ingest_twitter_rows(db: Session, rows: list[dict]) -> int:
    """rows: raw dicts from a twitter_export-style CSV (one row per tweet)."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    meta_by_handle: dict[str, str] = {}

    for row in rows:
        handle = (row.get('twitter_handle') or '').strip().lstrip('@')
        if not handle:
            continue
        grouped[handle].append({
            'text': row.get('content', ''),
            'created_at': row.get('created_at', ''),
            'favorite_count': _to_int(row.get('favorite_count')),
            'retweet_count': _to_int(row.get('retweet_count')),
            'reply_count': _to_int(row.get('reply_count')),
            'quote_count': _to_int(row.get('quote_count')),
            'views_count': _to_int(row.get('views_count')),
            'category': row.get('category') or 'general',
        })
        meta_by_handle[handle] = row.get('people_profile_json')

    count = 0
    for handle, tweets in grouped.items():
        stats = compute_tweet_stats(tweets)
        profile_meta = unpack_twitter_profile_meta(meta_by_handle.get(handle))
        features, inference_rows = extract_twitter_features(tweets, profile_meta)
        result = score_twitter_profile(features)

        defaults = {
            'username': handle,
            **stats,
            'risk_score': result.score,
            'risk_label': f'{result.tier} risk — local rule engine ({result.confidence:.0%} confidence)',
            'risk_factors': result.factors_as_dicts(),
            'inference_rows': inference_rows,
            'has_email': features['has_email'],
            'has_linkedin': features['has_linkedin'],
            'has_org': features['has_org'],
            'organization': profile_meta.get('organization', ''),
            'followers_count': features['followers_count'],
            'following_count': features['following_count'],
            'posting_entropy': features['posting_entropy'],
            'twitter_raw_data': {'twitter': tweets},
        }

        profile = db.query(TwitterProfile).filter(TwitterProfile.username == handle).first()
        if profile is None:
            profile = TwitterProfile(username=handle)
            db.add(profile)
        for k, v in defaults.items():
            setattr(profile, k, v)
        count += 1

    db.commit()
    return count


def ingest_linkedin_rows(db: Session, rows: list[dict]) -> int:
    """rows: raw dicts from a linkedin_export-style CSV (one row per person)."""
    count = 0
    for row in rows:
        pub_id = (row.get('public_identifier') or '').strip()
        if not pub_id:
            continue

        features, inference_rows = extract_linkedin_features(row)
        result = score_linkedin_profile(features)

        defaults = {
            'public_identifier': pub_id,
            'full_name': f"{(row.get('first_name') or '').strip()} {(row.get('last_name') or '').strip()}".strip(),
            'headline': row.get('headline') or '',
            'about': (row.get('about') or '')[:2000],
            'location_city': row.get('location_city') or '',
            'location_state': row.get('location_state') or '',
            'location_country': row.get('location_country') or '',
            'current_employer': features['current_employer'],
            'employer_count': features['employer_count'],
            'education_disclosed': features['education_disclosed'],
            'skill_count': features['skill_count'],
            'connections_count': features['connections_count'],
            'follower_count': features['follower_count'],
            'open_to_work': features['open_to_work'],
            'is_hiring': features['is_hiring'],
            'is_premium': _to_bool(row.get('is_premium')),
            'is_influencer': _to_bool(row.get('is_influencer')),
            'is_verified': _to_bool(row.get('is_verified')),
            'source_background_score': _to_float(row.get('background_score')),
            'risk_score': result.score,
            'risk_label': f'{result.tier} risk — local rule engine',
            'risk_factors': result.factors_as_dicts(),
            'inference_rows': inference_rows,
            'linkedin_raw_data': row,
        }

        profile = db.query(LinkedInProfile).filter(LinkedInProfile.public_identifier == pub_id).first()
        if profile is None:
            profile = LinkedInProfile(public_identifier=pub_id)
            db.add(profile)
        for k, v in defaults.items():
            setattr(profile, k, v)
        count += 1

    db.commit()
    return count
