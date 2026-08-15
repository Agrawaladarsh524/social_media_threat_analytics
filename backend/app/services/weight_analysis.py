"""
Weight-sensitivity analysis for the rule engine.

"Why 40/25/20/15?" is otherwise unanswerable except by assertion. This
recomputes every stored profile's score under several plausible alternative
weightings and reports how much the resulting *ranking* actually moves. The
honest claim it supports is not "these weights are optimal" — there is no
ground truth to optimise against — but "these are heuristic design weights,
and reasonable perturbations of them do / do not materially change the
ordering", which is a testable statement.

Recomputation is exact, not approximate, because the two bucket types behave
differently when their cap changes:

* ABSOLUTE buckets (PII, content sensitivity) award fixed points per finding
  — email +10, LinkedIn +10 — and are then clamped to the cap. Raising the
  cap cannot invent new findings, it only relaxes the clamp, so the correct
  recomputation is min(sum_of_awards, new_cap). The stored risk_factors hold
  each award individually, pre-clamp, so the true sum is recoverable.
* PROPORTIONAL buckets (predictability, reach) are computed as a fraction of
  their cap — (1 - entropy) * cap * confidence — so the correct recomputation
  scales by new_cap / old_cap.

Scaling everything uniformly would silently mis-handle the absolute buckets.
"""

from __future__ import annotations

from typing import Any

from .risk_engine import (
    MAX_CONTENT_SENSITIVITY,
    MAX_EXPOSURE_REACH,
    MAX_PII_EXPOSURE,
    MAX_PREDICTABILITY,
    _tier_for,
)

BASELINE_WEIGHTS: dict[str, int] = {
    'pii_exposure': MAX_PII_EXPOSURE,
    'predictability': MAX_PREDICTABILITY,
    'content_sensitivity': MAX_CONTENT_SENSITIVITY,
    'exposure_reach': MAX_EXPOSURE_REACH,
}

# Buckets whose points are fixed awards clamped at the cap, vs. buckets whose
# points are a fraction of the cap. Drives the recomputation branch above.
ABSOLUTE_BUCKETS = {'pii_exposure', 'content_sensitivity'}

# Variants all sum to 100 so scores stay on the same scale and stay comparable.
#
# Design note, learned by measuring rather than assuming: a first attempt used
# "+5 to bucket X" variants that each drew their 5 points from predictability.
# Two of them came back byte-identical, because RAISING the cap of an
# absolute-clamped bucket is a no-op unless profiles actually exceed the old
# cap — content sensitivity never approaches 20, so "content +5" measured
# nothing except the predictability change it was paired with. These variants
# instead each shift weight in a distinct direction, and include the naive
# equal-weight alternative as a genuine comparison point.
WEIGHT_VARIANTS: list[dict[str, Any]] = [
    {'name': 'Baseline', 'description': 'Shipping weights (40/25/20/15)',
     'weights': {'pii_exposure': 40, 'predictability': 25, 'content_sensitivity': 20, 'exposure_reach': 15}},
    {'name': 'PII-heavy', 'description': 'Disclosed identifiers dominate',
     'weights': {'pii_exposure': 50, 'predictability': 20, 'content_sensitivity': 15, 'exposure_reach': 15}},
    {'name': 'Behaviour-heavy', 'description': 'PII cap cut, routine weighted higher',
     'weights': {'pii_exposure': 30, 'predictability': 35, 'content_sensitivity': 20, 'exposure_reach': 15}},
    {'name': 'Reach-heavy', 'description': 'Audience size weighted higher',
     'weights': {'pii_exposure': 35, 'predictability': 20, 'content_sensitivity': 20, 'exposure_reach': 25}},
    {'name': 'Equal weights', 'description': 'Naive 25/25/25/25 — no bucket privileged',
     'weights': {'pii_exposure': 25, 'predictability': 25, 'content_sensitivity': 25, 'exposure_reach': 25}},
]


def bucket_totals(risk_factors: list[dict] | None) -> dict[str, float]:
    """Sums the stored per-factor points by bucket. For absolute buckets this
    is the pre-clamp total, which is what makes exact recomputation possible."""
    totals = dict.fromkeys(BASELINE_WEIGHTS, 0.0)
    for factor in (risk_factors or []):
        cat = factor.get('category')
        if cat in totals:
            totals[cat] += float(factor.get('points') or 0)
    return totals


def rescore(totals: dict[str, float], weights: dict[str, int]) -> int:
    """Recomputes a 0-100 score from stored bucket totals under new weights."""
    score = 0.0
    for bucket, baseline_cap in BASELINE_WEIGHTS.items():
        raw = totals.get(bucket, 0.0)
        new_cap = weights[bucket]
        if bucket in ABSOLUTE_BUCKETS:
            score += min(raw, new_cap)
        else:
            score += raw * (new_cap / baseline_cap) if baseline_cap else 0.0
    return max(0, min(round(score), 100))


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties. Implemented
    directly to avoid pulling scipy.stats in for one coefficient."""
    n = len(a)
    if n < 2:
        return 1.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg_rank
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den_a = sum((x - mean_a) ** 2 for x in ra) ** 0.5
    den_b = sum((y - mean_b) ** 2 for y in rb) ** 0.5
    if den_a == 0 or den_b == 0:
        return 1.0
    return num / (den_a * den_b)


def analyse_weight_sensitivity(profiles: list[Any], top_n: int = 10) -> dict[str, Any]:
    """
    Returns per-variant rank correlation, top-N overlap, tier-change rate and
    mean absolute score change, all measured against the baseline weighting.
    """
    if len(profiles) < 2:
        return {'ok': False, 'reason': 'Need at least 2 scored profiles.'}

    totals = [bucket_totals(p.risk_factors) for p in profiles]
    identifiers = [
        getattr(p, 'username', None) or getattr(p, 'public_identifier', '') for p in profiles
    ]

    baseline_weights = WEIGHT_VARIANTS[0]['weights']
    baseline_scores = [rescore(t, baseline_weights) for t in totals]
    baseline_tiers = [_tier_for(s) for s in baseline_scores]
    baseline_top = {
        identifiers[i] for i in sorted(range(len(profiles)), key=lambda i: -baseline_scores[i])[:top_n]
    }

    results = []
    for variant in WEIGHT_VARIANTS:
        scores = [rescore(t, variant['weights']) for t in totals]
        tiers = [_tier_for(s) for s in scores]
        top = {identifiers[i] for i in sorted(range(len(profiles)), key=lambda i: -scores[i])[:top_n]}

        results.append({
            'name': variant['name'],
            'description': variant['description'],
            'weights': variant['weights'],
            'rank_correlation': round(_spearman(baseline_scores, scores), 4),
            'top_n_overlap_pct': round(len(top & baseline_top) / max(len(baseline_top), 1) * 100, 1),
            'tier_change_pct': round(
                sum(1 for a, b in zip(baseline_tiers, tiers) if a != b) / len(profiles) * 100, 1
            ),
            'mean_abs_score_change': round(
                sum(abs(a - b) for a, b in zip(baseline_scores, scores)) / len(profiles), 2
            ),
        })

    non_baseline = results[1:]
    min_corr = min((r['rank_correlation'] for r in non_baseline), default=1.0)
    max_tier_change = max((r['tier_change_pct'] for r in non_baseline), default=0.0)

    return {
        'ok': True,
        'n_profiles': len(profiles),
        'top_n': top_n,
        'variants': results,
        'summary': {
            'min_rank_correlation': round(min_corr, 4),
            'max_tier_change_pct': max_tier_change,
            'ranking_is_robust': bool(min_corr >= 0.9),
        },
        'interpretation': (
            'Each variant shifts 5 points between buckets, holding the total at 100. '
            'High rank correlation means the ordering of profiles barely depends on the '
            'exact weights; a variant that changes almost nothing indicates its bucket '
            'contributes little signal in this population, not that the weighting is '
            'unimportant in principle. These are heuristic design weights tested for '
            'stability — not weights fitted to any ground truth, which does not exist here.'
        ),
    }
