"""
Unsupervised ML layer: Isolation Forest (anomaly/authenticity) + KMeans
(persona clustering). Both fit fresh on whatever profiles currently exist for
a platform — no labeled data, no pre-trained model files. Small dataset sizes
here (tens to low thousands of profiles) make refitting on demand cheap; a
`joblib`-cached model is a one-line addition if the profile count grows large
enough for refit cost to matter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_PROFILES_FOR_ML = 8  # below this, unsupervised models are unreliable/undefined


def _feature_matrix(rows: list[list[float]]) -> np.ndarray:
    return StandardScaler().fit_transform(np.array(rows, dtype=float))


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
    n_clusters: int = 4,
    contamination: float = 0.1,
) -> tuple[dict[int, float], dict[int, int], dict[str, Any]]:
    """
    Returns (anomaly_scores_by_id, cluster_ids_by_id, diagnostics).
    diagnostics includes the silhouette score and k used, for the
    Model Insights tab / evaluation section.
    """
    if len(profiles) < MIN_PROFILES_FOR_ML:
        return {}, {}, {'ok': False, 'reason': f'Need at least {MIN_PROFILES_FOR_ML} profiles, have {len(profiles)}.'}

    rows = [feature_fn(p) for p in profiles]
    X = _feature_matrix(rows)

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X)
    # decision_function: higher = more normal. Flip + rescale to 0-100 "anomaly score".
    raw = -iso.decision_function(X)
    lo, hi = raw.min(), raw.max()
    anomaly_scaled = (raw - lo) / (hi - lo) * 100 if hi > lo else np.zeros_like(raw)

    k = max(2, min(n_clusters, len(profiles) - 1))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if k < len(profiles) else None

    anomaly_by_id = {p.id: round(float(s), 1) for p, s in zip(profiles, anomaly_scaled)}
    cluster_by_id = {p.id: int(c) for p, c in zip(profiles, labels)}
    diagnostics = {
        'ok': True,
        'n_profiles': len(profiles),
        'k': k,
        'silhouette_score': round(float(sil), 3) if sil is not None else None,
        'contamination': contamination,
        'flagged_anomalies': int((anomaly_scaled >= 70).sum()),
    }
    return anomaly_by_id, cluster_by_id, diagnostics
