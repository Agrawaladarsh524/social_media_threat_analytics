"""
Run Apify actors and normalize items to the JSON shapes expected by the API.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from urllib.parse import urlparse

from apify_client import ApifyClient


def twitter_profile_url(handle: str | None) -> str | None:
    if not handle or not str(handle).strip():
        return None
    h = re.sub(r'^@+', '', handle.strip())
    return f'https://x.com/{h}' if h else None


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_tweet(raw: dict[str, Any]) -> dict[str, Any]:
    if 'author.name' in raw and 'author.userName' in raw:
        return dict(raw)
    author = raw.get('author')
    if isinstance(author, dict):
        name = _pick(author, 'name', 'displayName', default='')
        un = _pick(author, 'userName', 'username', 'screenName', default='')
        out = {**raw, 'author.name': name, 'author.userName': un}
        return out
    return {
        **raw,
        'author.name': str(raw.get('author.name', '') or ''),
        'author.userName': str(raw.get('author.userName', '') or ''),
    }


def _run_actor(
    token: str,
    actor_id: str,
    run_input: dict[str, Any],
    normalize_item: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fresh ApifyClient per call — safe with ThreadPoolExecutor (no shared client races)."""
    client = ApifyClient(token)
    run = client.actor(actor_id).call(run_input=run_input)
    ds = run.get('defaultDatasetId')
    if not ds:
        return []
    out: list[dict[str, Any]] = []
    for item in client.dataset(ds).iterate_items():
        if isinstance(item, dict):
            out.append(normalize_item(item) if normalize_item else dict(item))
    return out


def collect_osint(
    token: str,
    *,
    twitter: str | None,
    actor_twitter: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """
    Returns (data, errors) where errors map platform keys to messages.
    Empty lists for skipped platforms.
    """
    data: dict[str, list[dict[str, Any]]] = {
        'twitter': [],
    }
    errors: dict[str, str] = {}

    tw_url = twitter_profile_url(twitter)

    tasks: list[tuple[str, Callable[[], list[dict[str, Any]]]]] = []

    if tw_url:
        tasks.append(
            (
                'twitter',
                lambda u=tw_url: _run_actor(
                    token,
                    actor_twitter,
                    {
                        'customMapFunction': '(object) => { return {...object} }',
                        'getAboutData': False,
                        'getReplies': True,
                        'includeNativeRetweets': True,
                        'maxItems': 1000,
                        'minReplyCount': 0,
                        'onlyImages': False,
                        'startUrls': [u],
                    },
                    normalize_tweet,
                ),
            )
        )

    if not tasks:
        return data, errors

    with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as pool:
        future_map = {pool.submit(fn): key for key, fn in tasks}
        for fut in as_completed(future_map):
            key = future_map[fut]
            try:
                data[key] = fut.result()
            except Exception as e:  # noqa: BLE001 — surface Apify errors to client
                errors[key] = str(e)

    return data, errors
