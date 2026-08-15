"""
Unsupervised ML layer: Isolation Forest (anomaly/authenticity) + KMeans
(persona clustering).

Isolation Forest refits fresh every call by design — its score is explicitly
population-relative (see the Model Insights caption), so there's no
"identity" for it to preserve across runs.

KMeans is different: cluster IDs are meant to represent stable personas, but
naively refitting from scratch every ingestion reassigns centroids (and
therefore IDs) with no relationship to the previous run. The StandardScaler
is persisted to disk (joblib) and reused across calls so the feature space
itself doesn't shift call to call, and new cluster centroids are matched back
to the previous run's centroids (via optimal assignment on centroid
distance) before their IDs are handed out — so cluster_id=1 today and
cluster_id=1 next week refer to the same persona, not a coincidence of
refit order. Pass force_retrain=True to reset this (e.g. after a scoring
methodology change, or a `clear-db` wipe) and get a fresh baseline.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_PROFILES_FOR_ML = 8  # below this, unsupervised models are unreliable/undefined
K_MIN, K_MAX = 2, 8  # k-sweep range for cluster-count selection
MIN_CLUSTER_SIZE = 3  # reject a k if it produces a cluster smaller than this

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / 'models'

# Feature schemas, declared explicitly so a persisted scaler can be checked
# against the extractor that will feed it. Order is significant: StandardScaler
# stores per-column mean/scale, so reusing an artifact after the extractor's
# columns are reordered or redefined applies the wrong statistics to every
# column. scikit-learn raises on a feature-COUNT mismatch but is silent on a
# same-count reorder — measured effect of that silent case on a fixed dataset
# was silhouette 0.576 / k=5 against a correct 0.311 / k=2. Bumping these lists
# (rename, reorder, add, remove) changes the fingerprint and forces a refit.
TWITTER_FEATURE_NAMES = [
    'log1p_followers',
    'followers_per_following',
    'avg_engagement',
    'likes_per_tweet',
]
LINKEDIN_FEATURE_NAMES = [
    'log1p_connections',
    'followers_per_connection',
    'employer_count',
    'status_flag_count',
]
FEATURE_NAMES = {'twitter': TWITTER_FEATURE_NAMES, 'linkedin': LINKEDIN_FEATURE_NAMES}


def schema_fingerprint(feature_names: list[str], feature_fn=None) -> str:
    """
    Short hash identifying the feature schema a scaler was fit against.

    Hashing the declared names alone is not enough: the first version of this
    guard did exactly that, and a test where only the extractor's column
    ORDER changed still slipped through (names unchanged -> fingerprint
    unchanged -> stale scaler reused -> silhouette 0.576/k=5 against a
    correct 0.311/k=2). The declaration and the implementation can drift
    apart silently.

    So the extractor itself is folded in: its bytecode (catches reordering
    and any logic change) plus the attribute names it reads (catches
    swapping one profile field for another). A Python version bump changes
    bytecode and causes one spurious refit, which is harmless — the failure
    direction is "refit unnecessarily", never "reuse a wrong scaler".
    """
    parts = ['|'.join(feature_names)]
    if feature_fn is not None:
        code = getattr(feature_fn, '__code__', None)
        if code is not None:
            parts.append(code.co_code.hex())
            parts.append(','.join(sorted(code.co_names)))
    return hashlib.sha256('||'.join(parts).encode()).hexdigest()[:16]


def _model_path(platform: str, name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f'{platform}_{name}.joblib'


def clear_persisted_models(platform: str) -> None:
    """Called on /api/clear-db/ so a wiped-and-reingested population always
    gets a fresh scaler baseline instead of being scaled against old data."""
    for name in ('scaler', 'centroids'):
        path = _model_path(platform, name)
        if path.exists():
            path.unlink()


def load_model_metadata(platform: str) -> dict[str, Any] | None:
    """Provenance for the persisted scaler: which schema it was fit against,
    when, and on how many profiles. Returns None if nothing is persisted."""
    path = _model_path(platform, 'scaler')
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception:  # noqa: BLE001 — unreadable artifact is reported as absent
        return None
    if not isinstance(bundle, dict):
        return None
    return {k: v for k, v in bundle.items() if k != 'scaler'}


def _get_scaler(
    platform: str, rows: list[list[float]], force_retrain: bool, feature_fn=None,
) -> tuple[StandardScaler, dict[str, Any]]:
    """Returns (scaler, provenance). Refits whenever the persisted artifact is
    missing, unreadable, or was fit against a different feature schema — the
    schema check is what stops a reordered/redefined extractor from silently
    reusing the wrong per-column statistics."""
    path = _model_path(platform, 'scaler')
    names = FEATURE_NAMES[platform]
    want = schema_fingerprint(names, feature_fn)
    refit_reason = 'forced fresh baseline' if force_retrain else None

    if not force_retrain and path.exists():
        try:
            bundle = joblib.load(path)
            if isinstance(bundle, dict) and bundle.get('schema_fingerprint') == want:
                meta = {k: v for k, v in bundle.items() if k != 'scaler'}
                meta['refit'] = False
                return bundle['scaler'], meta
            refit_reason = (
                'feature schema changed since the persisted scaler was fit'
                if isinstance(bundle, dict) else
                'persisted artifact predates schema tracking'
            )
        except Exception:  # noqa: BLE001 — corrupt artifact, refit instead of hard-failing
            refit_reason = 'persisted artifact unreadable'
    elif not path.exists():
        refit_reason = 'no persisted scaler yet'

    scaler = StandardScaler().fit(np.array(rows, dtype=float))
    meta = {
        'schema_fingerprint': want,
        'feature_names': names,
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'training_population': len(rows),
    }
    joblib.dump({'scaler': scaler, **meta}, path)
    return scaler, {**meta, 'refit': True, 'refit_reason': refit_reason}


def _remap_cluster_ids(platform: str, centroids: np.ndarray, labels: np.ndarray, force_retrain: bool) -> np.ndarray:
    """Matches new centroids to the previous run's centroids by nearest
    distance (optimal assignment, so it's correct even when k changed
    between runs) and relabels so persona identity survives the refit."""
    path = _model_path(platform, 'centroids')
    prev: dict[int, np.ndarray] | None = None
    if not force_retrain and path.exists():
        try:
            prev = joblib.load(path)
        except Exception:  # noqa: BLE001 — corrupt/incompatible artifact, treat as first run
            prev = None

    if not prev:
        id_map = {i: i for i in range(len(centroids))}
    else:
        old_ids = list(prev.keys())
        old_matrix = np.array([prev[i] for i in old_ids])
        cost = np.linalg.norm(centroids[:, None, :] - old_matrix[None, :, :], axis=2)
        new_idx, old_idx = linear_sum_assignment(cost)
        id_map = {int(n): old_ids[int(o)] for n, o in zip(new_idx, old_idx)}
        next_free_id = max(old_ids) + 1
        for n in range(len(centroids)):
            if n not in id_map:
                id_map[n] = next_free_id
                next_free_id += 1

    joblib.dump({id_map[i]: centroids[i] for i in range(len(centroids))}, path)
    return np.array([id_map[int(label)] for label in labels])


def _select_k(X: np.ndarray, n_profiles: int) -> tuple[int, "KMeans", np.ndarray, float | None]:
    """
    Sweeps k=K_MIN..K_MAX (bounded by population size) and picks the k with
    the highest silhouette score, rejecting any k that produces a cluster
    smaller than MIN_CLUSTER_SIZE (a fixed k=4 default previously had no
    justification beyond "seemed reasonable" — this replaces that guess with
    a measured choice).
    """
    max_k = min(K_MAX, n_profiles - 1)
    best: tuple[int, "KMeans", np.ndarray, float] | None = None

    for k in range(K_MIN, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if np.bincount(labels).min() < MIN_CLUSTER_SIZE:
            continue
        score = silhouette_score(X, labels)
        if best is None or score > best[3]:
            best = (k, km, labels, score)

    if best is not None:
        k, km, labels, score = best
        return k, km, labels, score

    # No k satisfied the minimum cluster-size floor (tiny/unbalanced population) —
    # fall back to k=2 regardless, so clustering never hard-fails.
    k = max(2, max_k)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels) if k < n_profiles else None
    return k, km, labels, score


def twitter_ml_features(p: Any) -> list[float]:
    followers = max(p.followers_count, 0)
    following = max(p.following_count, 1)
    tweets = max(p.total_tweets, 1)
    return [
        np.log1p(followers),
        followers / following,
        p.avg_engagement,
        p.total_likes / tweets,
    ]


def linkedin_ml_features(p: Any) -> list[float]:
    conn = max(p.connections_count, 1)
    followers = max(p.follower_count, 0)
    return [
        np.log1p(conn),
        followers / conn,
        p.employer_count,
        float(p.is_premium) + float(p.is_influencer) + float(p.is_verified),
    ]


def run_anomaly_and_clusters(
    profiles: list[Any],
    feature_fn,
    platform: str,
    contamination: float = 0.1,
    force_retrain: bool = False,
) -> tuple[dict[int, float], dict[int, int], dict[str, Any]]:
    """
    Returns (anomaly_scores_by_id, cluster_ids_by_id, diagnostics).
    diagnostics includes the silhouette score and the k selected by the
    k-sweep, for the Model Insights tab / evaluation section.
    """
    if len(profiles) < MIN_PROFILES_FOR_ML:
        return {}, {}, {'ok': False, 'reason': f'Need at least {MIN_PROFILES_FOR_ML} profiles, have {len(profiles)}.'}

    rows = [feature_fn(p) for p in profiles]
    expected = len(FEATURE_NAMES[platform])
    if rows and len(rows[0]) != expected:
        return {}, {}, {
            'ok': False,
            'reason': (
                f'Feature extractor returned {len(rows[0])} features but the declared '
                f'{platform} schema has {expected}. Update FEATURE_NAMES in ml_models.py '
                'so the persisted scaler is validated against the real schema.'
            ),
        }

    scaler, model_meta = _get_scaler(platform, rows, force_retrain, feature_fn)
    X = scaler.transform(np.array(rows, dtype=float))

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X)
    # decision_function: higher = more normal. Flip + rescale to 0-100 "anomaly score".
    raw = -iso.decision_function(X)
    lo, hi = raw.min(), raw.max()
    anomaly_scaled = (raw - lo) / (hi - lo) * 100 if hi > lo else np.zeros_like(raw)

    k, km, labels, sil = _select_k(X, len(profiles))
    labels = _remap_cluster_ids(platform, km.cluster_centers_, labels, force_retrain)

    anomaly_by_id = {p.id: round(float(s), 1) for p, s in zip(profiles, anomaly_scaled)}
    cluster_by_id = {p.id: int(c) for p, c in zip(profiles, labels)}
    diagnostics = {
        'ok': True,
        'n_profiles': len(profiles),
        'k': k,
        'k_selection': f'swept k={K_MIN}..{min(K_MAX, len(profiles) - 1)}, chose highest silhouette (min cluster size {MIN_CLUSTER_SIZE})',
        'silhouette_score': round(float(sil), 3) if sil is not None else None,
        'contamination': contamination,
        'flagged_anomalies': int((anomaly_scaled >= 70).sum()),
        'cluster_ids_stable_across_refits': not force_retrain,
        'model': model_meta,
    }
    return anomaly_by_id, cluster_by_id, diagnostics
