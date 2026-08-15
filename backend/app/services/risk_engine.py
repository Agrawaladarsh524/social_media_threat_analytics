"""
Platform-agnostic weighted risk-scoring engine.

No AI, no external calls, no labeled training data. Every point awarded is a
deterministic function of a disclosed or inferred feature, and the full
breakdown is returned alongside the score so the dashboard can show exactly
why a profile scored the way it did.

Both `score_twitter_profile()` and `score_linkedin_profile()` normalize their
platform-specific inputs into the same four weighted buckets and hand off to
`_score()` — one scoring core, two feature extractors (see nlp_features.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Weighted buckets — must sum to 100.
MAX_PII_EXPOSURE = 40
MAX_PREDICTABILITY = 25
MAX_CONTENT_SENSITIVITY = 20
MAX_EXPOSURE_REACH = 15

TIER_BOUNDARIES = [(40, 'Low'), (70, 'Medium'), (90, 'High'), (101, 'Critical')]


@dataclass
class RiskFactor:
    category: str
    label: str
    points: float
    max_points: float
    evidence: str = ''


@dataclass
class RiskResult:
    score: int
    tier: str
    confidence: float = 1.0
    factors: list[RiskFactor] = field(default_factory=list)

    def factors_as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                'category': f.category,
                'label': f.label,
                'points': round(f.points, 1),
                'max_points': f.max_points,
                'evidence': f.evidence,
            }
            for f in self.factors
        ]


def _tier_for(score: int) -> str:
    for ceiling, name in TIER_BOUNDARIES:
        if score < ceiling:
            return name
    return 'Critical'


def _clamp(points: float, max_points: float) -> float:
    return max(0.0, min(points, max_points))


def posting_entropy(hour_counts: list[int]) -> float:
    """
    Shannon entropy (0-1, normalized) of a 24-bucket posting-hour histogram.
    Low entropy = posts cluster in a narrow, predictable time window
    (higher behavioral-predictability risk). High entropy = posting times
    are spread out (harder to infer a routine/timezone).
    """
    total = sum(hour_counts)
    if total == 0:
        return 1.0  # no data -> assume unpredictable (don't penalize)
    probs = [c / total for c in hour_counts if c > 0]
    raw_entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(24)
    return round(raw_entropy / max_entropy, 4) if max_entropy else 1.0


MIN_TWEETS_FOR_FULL_CONFIDENCE = 10


def score_twitter_profile(features: dict[str, Any]) -> RiskResult:
    """
    features keys: has_email, has_linkedin, has_org, location_entity_count,
    posting_entropy (0-1, low=predictable), sensitive_keyword_hits,
    followers_count, avg_engagement, total_tweets.

    Behavioral-pattern claims (posting-time predictability) require enough
    samples to be statistically meaningful: a profile with 1-2 tweets will
    trivially show near-zero entropy (all activity falls in one hour bucket),
    which is a sample-size artifact, not a real predictable routine. The
    predictability component is dampened by a confidence factor that ramps
    from 0 to 1 as tweet count approaches MIN_TWEETS_FOR_FULL_CONFIDENCE, so a
    single tweet can never masquerade as a proven behavioral pattern. PII and
    content-sensitivity findings are direct disclosures/keyword hits, not
    statistical inferences, so they are NOT dampened — one tweet revealing an
    email address is a real finding regardless of sample size.
    """
    factors: list[RiskFactor] = []
    total_tweets = int(features.get('total_tweets', 0))
    confidence = round(min(total_tweets / MIN_TWEETS_FOR_FULL_CONFIDENCE, 1.0), 2)

    # --- PII exposure (0-40) ---
    pii = 0.0
    if features.get('has_email'):
        pii += 10
        factors.append(RiskFactor('pii_exposure', 'Email address disclosed', 10, 10, 'people_profile_json.email'))
    if features.get('has_linkedin'):
        pii += 10
        factors.append(RiskFactor('pii_exposure', 'LinkedIn profile linked', 10, 10, 'people_profile_json.linkedin_url'))
    if features.get('has_org'):
        pii += 10
        factors.append(RiskFactor('pii_exposure', 'Employer/organization disclosed', 10, 10, 'people_profile_json.organization'))
    loc_hits = min(int(features.get('location_entity_count', 0)), 5)
    if loc_hits:
        pts = round(loc_hits * 2, 1)
        pii += pts
        factors.append(RiskFactor('pii_exposure', f'{loc_hits} location mention(s) in tweets (NER)', pts, 10, 'spaCy GPE/LOC entities'))
    pii = _clamp(pii, MAX_PII_EXPOSURE)

    # --- Behavioral predictability (0-25), confidence-weighted by sample size ---
    entropy = features.get('posting_entropy', 1.0)
    raw_predictability = (1 - entropy) * MAX_PREDICTABILITY
    predictability = round(raw_predictability * confidence, 1)
    if predictability > 0.5:
        factors.append(RiskFactor(
            'predictability', 'Predictable posting-time pattern', predictability, MAX_PREDICTABILITY,
            f'posting-hour entropy={entropy:.2f}, confidence={confidence:.0%} ({total_tweets} tweets sampled)',
        ))
    predictability = _clamp(predictability, MAX_PREDICTABILITY)

    # --- Content sensitivity (0-20) ---
    hits = min(int(features.get('sensitive_keyword_hits', 0)), 10)
    content = round(hits * 2, 1)
    if hits:
        factors.append(RiskFactor('content_sensitivity', f'{hits} sensitive-topic mention(s) (travel/home/family/finance)', content, MAX_CONTENT_SENSITIVITY, 'keyword dictionary match on tweet text'))
    content = _clamp(content, MAX_CONTENT_SENSITIVITY)

    # --- Exposure reach (0-15) ---
    followers = features.get('followers_count', 0) or 0
    reach = round(min(math.log10(followers + 1) / 6, 1) * MAX_EXPOSURE_REACH, 1)
    if reach > 0.5:
        factors.append(RiskFactor('exposure_reach', f'{followers:,} followers (audience size)', reach, MAX_EXPOSURE_REACH, 'people_profile_json.followers_count'))
    reach = _clamp(reach, MAX_EXPOSURE_REACH)

    total = round(pii + predictability + content + reach)
    total = max(0, min(total, 100))
    return RiskResult(score=total, tier=_tier_for(total), confidence=confidence, factors=factors)


def score_linkedin_profile(features: dict[str, Any]) -> RiskResult:
    """
    features keys: location_disclosed (0/1/2/3 = none/country/state/city),
    current_employer (str), employer_count, education_disclosed,
    open_to_work, is_hiring, connections_count, follower_count.
    """
    factors: list[RiskFactor] = []

    # --- PII exposure (0-40) --- explicit disclosure is worth more than inference.
    pii = 0.0
    granularity = int(features.get('location_disclosed', 0))
    if granularity == 3:
        pii += 15
        factors.append(RiskFactor('pii_exposure', 'Exact city disclosed', 15, 15, 'location_city'))
    elif granularity == 2:
        pii += 10
        factors.append(RiskFactor('pii_exposure', 'State/region disclosed', 10, 15, 'location_state'))
    elif granularity == 1:
        pii += 5
        factors.append(RiskFactor('pii_exposure', 'Country disclosed', 5, 15, 'location_country'))
    if features.get('current_employer'):
        pii += 10
        factors.append(RiskFactor('pii_exposure', f"Current employer disclosed: {features['current_employer']}", 10, 10, 'experiences_json[0].company_name'))
    if features.get('education_disclosed'):
        pii += 5
        factors.append(RiskFactor('pii_exposure', 'Education history disclosed', 5, 5, 'educations_json'))
    entity_hits = min(int(features.get('bio_entity_count', 0)), 5)
    if entity_hits:
        pts = round(entity_hits * 2, 1)
        pii += pts
        factors.append(RiskFactor('pii_exposure', f'{entity_hits} entity mention(s) in headline/about (NER)', pts, 10, 'spaCy entities on headline/about'))
    pii = _clamp(pii, MAX_PII_EXPOSURE)

    # --- Behavioral predictability / career timeline (0-25) ---
    employer_count = int(features.get('employer_count', 0))
    timeline = round(min(employer_count / 5, 1) * MAX_PREDICTABILITY, 1)
    if timeline > 0.5:
        factors.append(RiskFactor('predictability', f'{employer_count}-employer career timeline reconstructable', timeline, MAX_PREDICTABILITY, 'experiences_json (dated entries)'))
    predictability = _clamp(timeline, MAX_PREDICTABILITY)

    # --- Content/status sensitivity (0-20) ---
    content = 0.0
    if features.get('open_to_work'):
        content += 10
        factors.append(RiskFactor('content_sensitivity', 'Actively job-seeking ("Open to Work")', 10, 10, 'open_to_work=true — recruitment/phishing target'))
    if features.get('is_hiring'):
        content += 10
        factors.append(RiskFactor('content_sensitivity', 'Actively hiring', 10, 10, 'is_hiring=true — BEC/fake-candidate target'))
    content = _clamp(content, MAX_CONTENT_SENSITIVITY)

    # --- Exposure reach (0-15) ---
    reach_n = (features.get('connections_count', 0) or 0) + (features.get('follower_count', 0) or 0)
    reach = round(min(math.log10(reach_n + 1) / 5, 1) * MAX_EXPOSURE_REACH, 1)
    if reach > 0.5:
        factors.append(RiskFactor('exposure_reach', f'{reach_n:,} combined connections/followers', reach, MAX_EXPOSURE_REACH, 'connections_count + follower_count'))
    reach = _clamp(reach, MAX_EXPOSURE_REACH)

    total = round(pii + predictability + content + reach)
    total = max(0, min(total, 100))
    return RiskResult(score=total, tier=_tier_for(total), factors=factors)
