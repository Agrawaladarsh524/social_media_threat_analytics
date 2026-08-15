"""Tests for local spaCy NER extraction and feature engineering — no network
calls, no OpenAI key. spaCy's en_core_web_sm must be installed
(`python -m spacy download en_core_web_sm`)."""

from app.services.nlp_features import (
    count_sensitive_keyword_hits,
    extract_entities,
    extract_linkedin_features,
    extract_twitter_features,
    posting_hour_histogram,
    unpack_twitter_profile_meta,
)


def test_extract_entities_finds_known_location():
    entities = extract_entities('Just landed in San Francisco for the week.')
    labels_and_text = [(label, text) for label, text in entities]
    assert any('San Francisco' in text for _label, text in labels_and_text)


def test_extract_entities_empty_text_returns_empty():
    assert extract_entities('') == []
    assert extract_entities(None) == []


def test_sensitive_keyword_hits_detects_travel_and_family():
    count, hits = count_sensitive_keyword_hits([
        'Just landed in Tokyo for vacation with my family',
        'Nothing interesting here',
    ])
    assert count >= 1
    assert any('landed in' in kw or 'vacation' in kw or 'my family' in kw for kw, _ in hits)


def test_posting_hour_histogram_buckets_correctly():
    dates = ['2026-01-01T09:00:00', '2026-01-02T09:30:00', '2026-01-03T22:00:00']
    hist = posting_hour_histogram(dates)
    assert len(hist) == 24
    assert hist[9] == 2
    assert hist[22] == 1
    assert sum(hist) == 3


def test_posting_hour_histogram_ignores_malformed_dates():
    hist = posting_hour_histogram(['', None, 'not-a-date', '2026-01-01T05:00:00'])
    assert sum(hist) == 1
    assert hist[5] == 1


def test_unpack_twitter_profile_meta_from_json_string():
    meta = unpack_twitter_profile_meta(
        '{"organization": "Acme", "email": "a@example.com", "followers_count": 100}'
    )
    assert meta['organization'] == 'Acme'
    assert meta['email'] == 'a@example.com'
    assert meta['followers_count'] == 100


def test_extract_twitter_features_shape():
    tweets = [
        {'text': 'Flying to New York tomorrow', 'created_at': '2026-01-01T10:00:00'},
        {'text': 'Great coffee this morning', 'created_at': '2026-01-02T10:00:00'},
    ]
    meta = {'organization': 'Acme', 'email': None, 'linkedin_url': None, 'followers_count': 50, 'following_count': 10}
    features, inference_rows = extract_twitter_features(tweets, meta)
    assert features['has_org'] is True
    assert features['has_email'] is False
    assert features['total_tweets'] == 2
    assert isinstance(inference_rows, list)


def test_extract_linkedin_features_location_granularity():
    row_with_city = {'location_city': 'Austin', 'location_state': 'Texas', 'location_country': 'USA'}
    features, _ = extract_linkedin_features(row_with_city)
    assert features['location_disclosed'] == 3

    row_country_only = {'location_city': '', 'location_state': '', 'location_country': 'USA'}
    features2, _ = extract_linkedin_features(row_country_only)
    assert features2['location_disclosed'] == 1

    row_none = {'location_city': '', 'location_state': '', 'location_country': ''}
    features3, _ = extract_linkedin_features(row_none)
    assert features3['location_disclosed'] == 0


def test_extract_linkedin_features_employer_and_flags():
    row = {
        'experiences_json': '[{"company_name": "Acme Corp"}, {"company_name": "Old Co"}]',
        'open_to_work': '1',
        'is_hiring': '0',
    }
    features, rows = extract_linkedin_features(row)
    assert features['current_employer'] == 'Acme Corp'
    assert features['employer_count'] == 2
    assert features['open_to_work'] is True
    assert features['is_hiring'] is False
    assert any(r['targetInfo'] == 'Current employer' for r in rows)
