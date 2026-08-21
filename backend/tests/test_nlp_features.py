"""Tests for local spaCy NER extraction and feature engineering — no network
calls, no OpenAI key. spaCy's en_core_web_sm must be available. Semantic
content matching uses scikit-learn's TfidfVectorizer (no external model
download needed)."""

from app.services.nlp_features import (
    count_sensitive_keyword_hits,
    extract_entities,
    extract_linkedin_features,
    extract_twitter_features,
    posting_hour_histogram,
    semantic_content_hits,
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
    # combined keyword+semantic hit count must be at least the semantic-only count
    # (union, not sum) — the "flying to New York tomorrow" tweet should register
    # as a semantic hit even though it doesn't match the keyword dictionary verbatim.
    assert features['sensitive_keyword_hits'] >= features['semantic_only_hits']


def test_semantic_content_hits_distinguishes_near_term_from_past():
    """The canonical example this feature exists for: same topic (travel/home),
    but only one of these is a real near-term, specific exposure."""
    near_term_count, near_term_hits = semantic_content_hits(['I am flying home tomorrow'])
    past_count, _past_hits = semantic_content_hits(['I visited my parents last year'])

    assert near_term_count >= 1
    assert any(h['category'] == 'travel' for h in near_term_hits)
    # Not a hard guarantee for every possible sentence pair, but for this
    # specific canonical example the near-term disclosure must score a
    # closer match than the past-tense one.
    if near_term_hits:
        near_term_sim = near_term_hits[0]['similarity']
        _past_count2, past_hits2 = semantic_content_hits(['I visited my parents last year', 'I am flying home tomorrow'])
        past_only_sim = next((h['similarity'] for h in past_hits2 if 'visited' in h['text']), 0.0)
        assert near_term_sim >= past_only_sim


def test_semantic_content_hits_empty_input():
    assert semantic_content_hits([]) == (0, [])
    assert semantic_content_hits(['', None]) == (0, [])


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


def test_extract_linkedin_features_field_completeness():
    full = {
        'experiences_json': '[{"company_name": "Acme"}]',
        'educations_json': '[{"school_name": "MIT"}]',
        'profile_skills_json': '[{"skill_name": "python"}]',
    }
    features_full, _ = extract_linkedin_features(full)
    assert features_full['field_completeness'] == 1.0

    empty = {'experiences_json': '[]', 'educations_json': '[]', 'profile_skills_json': '[]'}
    features_empty, _ = extract_linkedin_features(empty)
    assert features_empty['field_completeness'] == 0.0
