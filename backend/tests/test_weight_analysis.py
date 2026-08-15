"""
Tests for the weight-sensitivity analysis.

The load-bearing test here is the baseline self-consistency check: recomputing
scores from stored risk_factors under the shipping weights must reproduce the
stored scores exactly. If that drifts, every sensitivity number computed
against the baseline is meaningless, so it is checked directly rather than
inferred from the reported correlation.
"""

from types import SimpleNamespace

from app.services.risk_engine import score_twitter_profile
from app.services.weight_analysis import (
    BASELINE_WEIGHTS,
    WEIGHT_VARIANTS,
    analyse_weight_sensitivity,
    bucket_totals,
    rescore,
)


def _profile(pid, factors, username=None):
    return SimpleNamespace(id=pid, username=username or f'user{pid}', risk_factors=factors)


def _factors(pii=0.0, pred=0.0, content=0.0, reach=0.0):
    out = []
    for cat, pts in (('pii_exposure', pii), ('predictability', pred),
                     ('content_sensitivity', content), ('exposure_reach', reach)):
        if pts:
            out.append({'category': cat, 'label': cat, 'points': pts, 'max_points': BASELINE_WEIGHTS[cat],
                        'evidence': 'test'})
    return out


def test_bucket_totals_sums_by_category():
    factors = [
        {'category': 'pii_exposure', 'points': 10},
        {'category': 'pii_exposure', 'points': 10},
        {'category': 'exposure_reach', 'points': 7.5},
        {'category': 'unknown_bucket', 'points': 99},  # ignored
    ]
    totals = bucket_totals(factors)
    assert totals['pii_exposure'] == 20
    assert totals['exposure_reach'] == 7.5
    assert totals['content_sensitivity'] == 0
    assert 'unknown_bucket' not in totals


def test_bucket_totals_handles_empty():
    assert bucket_totals(None) == dict.fromkeys(BASELINE_WEIGHTS, 0.0)
    assert bucket_totals([]) == dict.fromkeys(BASELINE_WEIGHTS, 0.0)


def test_rescore_under_baseline_reproduces_the_real_engine_score():
    """Round-trip against the actual scoring engine, not a hand-written
    expectation: score a profile normally, then recompute from its emitted
    risk_factors. The two must agree, or the sensitivity analysis is invalid."""
    features = {
        'has_email': True, 'has_linkedin': True, 'has_org': True,
        'location_entity_count': 3, 'posting_entropy': 0.4,
        'sensitive_keyword_hits': 2, 'followers_count': 25_000, 'total_tweets': 40,
    }
    result = score_twitter_profile(features)
    recomputed = rescore(bucket_totals(result.factors_as_dicts()), BASELINE_WEIGHTS)
    assert recomputed == result.score


def test_rescore_absolute_bucket_is_clamped_not_scaled():
    """PII awards are fixed points clamped at the cap. Raising the cap must not
    inflate a profile whose awards already fit under the old cap."""
    totals = bucket_totals(_factors(pii=30))
    assert rescore(totals, {**BASELINE_WEIGHTS, 'pii_exposure': 40}) == 30
    assert rescore(totals, {**BASELINE_WEIGHTS, 'pii_exposure': 50}) == 30   # no inflation
    assert rescore(totals, {**BASELINE_WEIGHTS, 'pii_exposure': 25}) == 25   # lower cap clamps


def test_rescore_proportional_bucket_scales_with_cap():
    """Reach is computed as a fraction of its cap, so it scales when the cap does."""
    totals = bucket_totals(_factors(reach=10))  # 10 of a 15 cap
    assert rescore(totals, {**BASELINE_WEIGHTS, 'exposure_reach': 30}) == 20  # doubled cap -> doubled
    assert rescore(totals, {**BASELINE_WEIGHTS, 'exposure_reach': 15}) == 10


def test_rescore_is_bounded():
    totals = bucket_totals(_factors(pii=999, pred=999, content=999, reach=999))
    assert 0 <= rescore(totals, BASELINE_WEIGHTS) <= 100


def test_analysis_baseline_variant_is_perfectly_self_consistent():
    profiles = [
        _profile(1, _factors(pii=40, pred=20, reach=12)),
        _profile(2, _factors(pii=10, pred=5, reach=6)),
        _profile(3, _factors(pii=25, pred=12, content=4, reach=9)),
        _profile(4, _factors(pii=35, pred=18, reach=14)),
    ]
    out = analyse_weight_sensitivity(profiles)
    assert out['ok'] is True
    baseline = out['variants'][0]
    assert baseline['name'] == 'Baseline'
    assert baseline['rank_correlation'] == 1.0
    assert baseline['tier_change_pct'] == 0.0
    assert baseline['mean_abs_score_change'] == 0.0
    assert baseline['top_n_overlap_pct'] == 100.0


def test_analysis_variants_are_distinct():
    """A variant set where two rows produce identical metrics is measuring the
    same change twice — that bug shipped once and is guarded against here."""
    profiles = [
        _profile(i, _factors(pii=10 + i, pred=5 + i, content=(i % 3), reach=4 + (i % 5)))
        for i in range(30)
    ]
    out = analyse_weight_sensitivity(profiles)
    signatures = [
        (v['rank_correlation'], v['tier_change_pct'], v['mean_abs_score_change'])
        for v in out['variants']
    ]
    assert len(set(signatures)) == len(signatures), 'two weight variants produced identical results'


def test_analysis_metrics_are_in_valid_ranges():
    profiles = [
        _profile(i, _factors(pii=(i * 3) % 41, pred=(i * 2) % 26, content=i % 5, reach=i % 16))
        for i in range(25)
    ]
    out = analyse_weight_sensitivity(profiles)
    for v in out['variants']:
        assert -1.0 <= v['rank_correlation'] <= 1.0
        assert 0.0 <= v['top_n_overlap_pct'] <= 100.0
        assert 0.0 <= v['tier_change_pct'] <= 100.0
        assert v['mean_abs_score_change'] >= 0.0
    assert sum(w for w in WEIGHT_VARIANTS[0]['weights'].values()) == 100


def test_all_variants_keep_the_total_at_100():
    """Scores must stay on a 0-100 scale across variants or they aren't comparable."""
    for variant in WEIGHT_VARIANTS:
        assert sum(variant['weights'].values()) == 100, variant['name']


def test_analysis_needs_at_least_two_profiles():
    out = analyse_weight_sensitivity([_profile(1, _factors(pii=10))])
    assert out['ok'] is False
