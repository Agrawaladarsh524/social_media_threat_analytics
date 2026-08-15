"""
Direct unit tests for the unsupervised ML layer — previously only covered
indirectly through one route-level test. Uses a temp model directory per
test (via monkeypatching MODEL_DIR) so persisted scaler/centroid artifacts
never leak between tests or touch the real backend/models/ directory.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from app.services import ml_models


@pytest.fixture(autouse=True)
def _isolated_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_models, 'MODEL_DIR', tmp_path)


def _profile(pid, followers, following, avg_engagement, likes, tweets=10):
    return SimpleNamespace(
        id=pid, followers_count=followers, following_count=following,
        avg_engagement=avg_engagement, total_likes=likes, total_tweets=tweets,
    )


def _synthetic_population(n=30, seed=0):
    rng = np.random.default_rng(seed)
    profiles = []
    for i in range(n):
        followers = max(1, int(rng.normal(1000, 100)))
        following = max(1, int(rng.normal(200, 20)))
        avg_eng = float(rng.normal(50, 5))
        likes = max(0, int(rng.normal(500, 50)))
        profiles.append(_profile(i + 1, followers, following, avg_eng, likes))
    return profiles


def test_below_minimum_profiles_returns_not_ok():
    profiles = _synthetic_population(n=3)
    anomaly, cluster, diag = ml_models.run_anomaly_and_clusters(profiles, ml_models.twitter_ml_features, 'twitter')
    assert anomaly == {} and cluster == {}
    assert diag['ok'] is False


def test_obvious_outlier_gets_highest_anomaly_score():
    profiles = _synthetic_population(n=20)
    # profile 21 is wildly outside the normal population's distribution
    profiles.append(_profile(21, followers=10_000_000, following=1, avg_engagement=99999, likes=99999))

    anomaly, _cluster, diag = ml_models.run_anomaly_and_clusters(profiles, ml_models.twitter_ml_features, 'twitter')
    assert diag['ok'] is True
    outlier_score = anomaly[21]
    normal_scores = [s for pid, s in anomaly.items() if pid != 21]
    assert outlier_score == max(anomaly.values())
    assert outlier_score > max(normal_scores)


def test_k_sweep_selects_a_k_and_reports_selection_method():
    profiles = _synthetic_population(n=25)
    _anomaly, cluster, diag = ml_models.run_anomaly_and_clusters(profiles, ml_models.twitter_ml_features, 'twitter')
    assert diag['ok'] is True
    assert ml_models.K_MIN <= diag['k'] <= ml_models.K_MAX
    assert 'k_selection' in diag
    assert len(set(cluster.values())) == diag['k']


def _three_persona_population(seed):
    """Three genuinely separated behavioral personas (not one blob split
    arbitrarily) — the only population shape where 'did remapping preserve
    persona identity' is actually a meaningful question to ask."""
    rng = np.random.default_rng(seed)
    profiles = []
    pid = 1
    personas = [
        dict(followers=200, following=200, avg_engagement=2, likes=20),     # low-reach, low-engagement
        dict(followers=50_000, following=500, avg_engagement=80, likes=900),  # high-reach, high-engagement
        dict(followers=5_000, following=4_800, avg_engagement=5, likes=50),   # follow-for-follow pattern
    ]
    persona_by_pid = {}
    for persona in personas:
        for _ in range(15):
            followers = max(1, int(rng.normal(persona['followers'], persona['followers'] * 0.05 + 1)))
            following = max(1, int(rng.normal(persona['following'], persona['following'] * 0.05 + 1)))
            avg_eng = float(rng.normal(persona['avg_engagement'], 0.5))
            likes = max(0, int(rng.normal(persona['likes'], persona['likes'] * 0.05 + 1)))
            profiles.append(_profile(pid, followers, following, avg_eng, likes))
            persona_by_pid[pid] = personas.index(persona)
            pid += 1
    return profiles, persona_by_pid


def test_cluster_ids_stable_across_independent_refits_of_same_profiles():
    """The core Phase 2 fix: re-scoring the same profiles later must not
    reassign their persona cluster IDs just because KMeans refit from scratch."""
    profiles_run1, persona_by_pid = _three_persona_population(seed=1)
    _anomaly1, cluster1, diag1 = ml_models.run_anomaly_and_clusters(profiles_run1, ml_models.twitter_ml_features, 'twitter')
    assert diag1['cluster_ids_stable_across_refits'] is True

    # Same underlying personas, freshly regenerated (independent random draw) —
    # simulates re-ingesting and re-scoring a similar population later.
    profiles_run2, _ = _three_persona_population(seed=2)
    for p in profiles_run2:
        p.id += 1000  # distinct IDs, but same persona-index ordering as run 1
    _anomaly2, cluster2, diag2 = ml_models.run_anomaly_and_clusters(profiles_run2, ml_models.twitter_ml_features, 'twitter')

    # Each persona's representative members should land in the same cluster ID
    # in both runs, since the scaler was reused (not refit) and centroids were
    # matched back to the previous run's before IDs were assigned.
    for persona_idx in range(3):
        run1_id = next(cluster1[pid] for pid, pi in persona_by_pid.items() if pi == persona_idx)
        run2_id = cluster2[1000 + next(pid for pid, pi in persona_by_pid.items() if pi == persona_idx)]
        assert run1_id == run2_id, f'persona {persona_idx} cluster ID drifted between refits'


def test_force_retrain_resets_cluster_identity_tracking():
    profiles = _synthetic_population(n=25)
    _a, _c, diag = ml_models.run_anomaly_and_clusters(
        profiles, ml_models.twitter_ml_features, 'twitter', force_retrain=True,
    )
    assert diag['cluster_ids_stable_across_refits'] is False


def test_clear_persisted_models_removes_artifacts(tmp_path):
    profiles = _synthetic_population(n=25)
    ml_models.run_anomaly_and_clusters(profiles, ml_models.twitter_ml_features, 'twitter')
    assert any(tmp_path.iterdir())
    ml_models.clear_persisted_models('twitter')
    assert not any(tmp_path.iterdir())
