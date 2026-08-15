"""
Sanity-check tests for the deterministic rule-based risk engine. There is no
labeled ground truth for "true" risk, so these are face-validity checks: a
synthetic high-exposure profile must score meaningfully higher than a
synthetic low-exposure one, scores must stay in [0, 100], and the
sample-size confidence dampening on Twitter's predictability component must
behave as designed.
"""

from app.services.risk_engine import (
    posting_entropy,
    score_linkedin_profile,
    score_twitter_profile,
)


def test_twitter_high_exposure_scores_higher_than_low_exposure():
    high = score_twitter_profile({
        'has_email': True, 'has_linkedin': True, 'has_org': True,
        'location_entity_count': 5, 'posting_entropy': 0.1,
        'sensitive_keyword_hits': 5, 'followers_count': 500_000,
        'total_tweets': 50,
    })
    low = score_twitter_profile({
        'has_email': False, 'has_linkedin': False, 'has_org': False,
        'location_entity_count': 0, 'posting_entropy': 1.0,
        'sensitive_keyword_hits': 0, 'followers_count': 10,
        'total_tweets': 50,
    })
    assert high.score > low.score
    assert low.score < 40  # low-exposure profile should land in the Low tier


def test_twitter_score_and_confidence_are_bounded():
    result = score_twitter_profile({
        'has_email': True, 'has_linkedin': True, 'has_org': True,
        'location_entity_count': 999, 'posting_entropy': 0.0,
        'sensitive_keyword_hits': 999, 'followers_count': 10**9,
        'total_tweets': 1000,
    })
    assert 0 <= result.score <= 100
    assert 0 <= result.confidence <= 1


def test_low_tweet_count_dampens_predictability_not_pii():
    """A single tweet should never be treated as a proven behavioral routine,
    but a disclosed email in that one tweet is still a real finding."""
    one_tweet = score_twitter_profile({
        'has_email': True, 'posting_entropy': 0.0,  # would-be "perfectly predictable"
        'followers_count': 100, 'total_tweets': 1,
    })
    many_tweets_same_entropy = score_twitter_profile({
        'has_email': True, 'posting_entropy': 0.0,
        'followers_count': 100, 'total_tweets': 50,
    })
    assert one_tweet.confidence < many_tweets_same_entropy.confidence
    assert one_tweet.score < many_tweets_same_entropy.score
    pii_points = next(f.points for f in one_tweet.factors if f.label == 'Email address disclosed')
    assert pii_points == 10  # undamped regardless of sample size


def test_linkedin_explicit_location_worth_more_than_country_only():
    city = score_linkedin_profile({'location_disclosed': 3})
    country = score_linkedin_profile({'location_disclosed': 1})
    none_ = score_linkedin_profile({'location_disclosed': 0})
    assert city.score > country.score > none_.score


def test_linkedin_job_status_flags_add_content_sensitivity_points():
    baseline = score_linkedin_profile({})
    seeking = score_linkedin_profile({'open_to_work': True})
    assert seeking.score > baseline.score


def test_posting_entropy_uniform_vs_concentrated():
    uniform = posting_entropy([1] * 24)
    concentrated = posting_entropy([100] + [0] * 23)
    assert uniform > concentrated
    assert uniform == 1.0
    assert concentrated == 0.0


def test_posting_entropy_no_data_defaults_to_unpredictable():
    assert posting_entropy([0] * 24) == 1.0
