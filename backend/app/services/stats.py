"""Real metrics computed directly from raw tweet rows (no AI involved)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _to_int(val: Any) -> int:
    try:
        return int(float(val or 0))
    except (TypeError, ValueError):
        return 0


def compute_tweet_stats(tweets: list) -> dict[str, Any]:
    if not tweets:
        return {
            'total_tweets': 0, 'date_first_tweet': '', 'date_last_tweet': '',
            'categories': [], 'total_likes': 0, 'total_retweets': 0,
            'total_replies': 0, 'total_quotes': 0, 'total_views': 0,
            'avg_engagement': 0.0,
        }

    total_likes = total_retweets = total_replies = total_quotes = total_views = 0
    dates: list[str] = []
    categories: set[str] = set()

    for t in tweets:
        if not isinstance(t, dict):
            continue
        total_likes    += _to_int(t.get('favorite_count') or t.get('likeCount'))
        total_retweets += _to_int(t.get('retweet_count')  or t.get('retweetCount'))
        total_replies  += _to_int(t.get('reply_count')    or t.get('replyCount'))
        total_quotes   += _to_int(t.get('quote_count')    or t.get('quoteCount'))
        total_views    += _to_int(t.get('views_count')    or t.get('viewCount'))

        cat = t.get('category') or t.get('type')
        if cat:
            categories.add(str(cat).strip())

        date = t.get('created_at') or t.get('createdAt') or t.get('date')
        if date:
            dates.append(str(date))

    dates_sorted = sorted(dates)
    total = len(tweets)
    avg_eng = round((total_likes + total_retweets + total_replies) / total, 2) if total else 0.0

    return {
        'total_tweets':     total,
        'date_first_tweet': dates_sorted[0]  if dates_sorted else '',
        'date_last_tweet':  dates_sorted[-1] if dates_sorted else '',
        'categories':       sorted(categories),
        'total_likes':      total_likes,
        'total_retweets':   total_retweets,
        'total_replies':    total_replies,
        'total_quotes':     total_quotes,
        'total_views':      total_views,
        'avg_engagement':   avg_eng,
    }


def build_profile_defaults(bundle: dict, datasets: dict) -> dict[str, Any]:
    """Column values to persist: real tweet stats + AI risk signals (no prose)."""
    tweets = datasets.get('twitter', []) if isinstance(datasets, dict) else []
    if isinstance(tweets, dict):
        tweets = [tweets]

    executive = bundle.get('executive', {}) or {}

    return {
        'twitter_raw_data':    datasets,
        'ai_bundle':           bundle,
        'risk_score':          executive.get('riskScore', 0),
        'risk_label':          executive.get('riskLabel', ''),
        'inference_rows':      bundle.get('inferenceRows', []),
        'pattern_of_life':     bundle.get('patternOfLife', []),
        'bundle_generated_at': datetime.now(timezone.utc),
        **compute_tweet_stats(tweets),
    }
