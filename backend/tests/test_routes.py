"""FastAPI TestClient smoke tests for the local ingestion pipeline and its
downstream endpoints. Uses a temp SQLite DB (see conftest.py) — never touches
the real dev database."""

import io

TWITTER_CSV = (
    'id,source_type,person_name,twitter_handle,person_twitter_id,tweet_id,content,tweet_url,'
    'created_at,category,favorite_count,retweet_count,reply_count,quote_count,views_count,'
    'bookmark_count,people_profile_json\n'
    '1,FOUNDER,Jane Doe,janedoe,123,111,"Excited to announce our Series A!",https://x.com/1,'
    '2026-01-01T10:00:00,funding,50,10,5,2,1000,3,'
    '"{""organization"": ""Acme Inc"", ""email"": ""jane@acme.com"", ""followers_count"": 5000, '
    '""following_count"": 200}"\n'
    '2,FOUNDER,Jane Doe,janedoe,123,112,"Great coffee this morning",https://x.com/2,'
    '2026-01-02T10:15:00,general,5,1,0,0,100,0,'
    '"{""organization"": ""Acme Inc"", ""email"": ""jane@acme.com"", ""followers_count"": 5000, '
    '""following_count"": 200}"\n'
)

LINKEDIN_CSV = (
    'id,linkedin_profile_id,public_identifier,linkedin_url,first_name,last_name,headline,about,'
    'photo_url,location_text,location_city,location_state,location_country,connections_count,'
    'follower_count,top_skills_raw,open_to_work,is_hiring,is_premium,is_influencer,is_verified,'
    'registered_at,created_at,updated_at,background_score,score_computed_at,experiences_json,'
    'educations_json,certifications_json,projects_json,publications_json,languages_json,'
    'volunteering_json,honors_and_awards_json,courses_json,profile_skills_json,profile_labels_json,'
    'experience_skills_association_json,education_skills_association_json\n'
    'abc,xyz,jane-smith-1,https://linkedin.com/in/jane-smith-1,Jane,Smith,'
    '"Founder at Acme",About me,,'
    '"Austin, Texas",Austin,Texas,United States,3000,4000,,1,0,0,0,0,'
    '2020-01-01,2026-01-01,2026-01-01,10.0,2026-01-01,'
    '"[{""company_name"": ""Acme Inc""}]","[{""school_name"": ""MIT""}]",[],[],[],[],[],[],[],[],[],[],[]\n'
)


def _ingest(client, csv_text: str, platform: str, filename: str):
    return client.post(
        '/api/ingest-local/',
        files={'file': (filename, io.BytesIO(csv_text.encode('utf-8')), 'text/csv')},
        data={'platform': platform},
    )


def test_health(client):
    r = client.get('/api/health/')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_ingest_local_rejects_bad_platform(client):
    r = _ingest(client, TWITTER_CSV, 'myspace', 'x.csv')
    assert r.status_code == 400


def test_ingest_local_twitter_end_to_end(client):
    r = _ingest(client, TWITTER_CSV, 'twitter', 'twitter.csv')
    assert r.status_code == 200
    body = r.json()
    assert body['ok'] is True
    assert body['profiles_ingested'] == 1  # both rows are the same handle

    r2 = client.get('/api/usernames/', params={'platform': 'twitter'})
    assert r2.json()['usernames'] == ['janedoe']

    r3 = client.get('/api/check-db/', params={'platform': 'twitter'})
    profile = r3.json()['profiles'][0]
    assert profile['risk_score'] is not None
    assert 0 <= profile['risk_score'] <= 100
    assert profile['tweet_stats']['total_tweets'] == 2

    r4 = client.get('/api/profile-detail/twitter/janedoe/')
    assert r4.json()['ok'] is True
    assert len(r4.json()['profile']['risk_factors']) > 0


def test_ingest_local_linkedin_end_to_end(client):
    r = _ingest(client, LINKEDIN_CSV, 'linkedin', 'linkedin.csv')
    assert r.status_code == 200
    assert r.json()['profiles_ingested'] == 1

    r2 = client.get('/api/check-db/', params={'platform': 'linkedin'})
    profile = r2.json()['profiles'][0]
    assert profile['username'] == 'jane-smith-1'
    assert profile['risk_score'] is not None
    assert profile['tweet_stats']['current_employer'] == 'Acme Inc'


def test_analytics_empty_db_returns_ok_false(client):
    r = client.get('/api/analytics/', params={'platform': 'twitter'})
    body = r.json()
    assert body['ok'] is False


def test_analytics_after_ingest_returns_charts(client):
    _ingest(client, TWITTER_CSV, 'twitter', 'twitter.csv')
    r = client.get('/api/analytics/', params={'platform': 'twitter'})
    body = r.json()
    assert body['ok'] is True
    assert 'risk_distribution' in body['charts']


def test_recompute_models_requires_minimum_profiles(client):
    _ingest(client, TWITTER_CSV, 'twitter', 'twitter.csv')  # only 1 profile
    r = client.post('/api/recompute-models/', params={'platform': 'twitter'})
    body = r.json()
    assert body['ok'] is True
    assert body['diagnostics']['ok'] is False  # below MIN_PROFILES_FOR_ML


def test_clear_db_only_affects_specified_platform(client):
    _ingest(client, TWITTER_CSV, 'twitter', 'twitter.csv')
    _ingest(client, LINKEDIN_CSV, 'linkedin', 'linkedin.csv')

    client.post('/api/clear-db/', params={'platform': 'twitter'})

    assert client.get('/api/usernames/', params={'platform': 'twitter'}).json()['usernames'] == []
    assert client.get('/api/usernames/', params={'platform': 'linkedin'}).json()['usernames'] == ['jane-smith-1']


def test_ingest_reports_data_quality(client):
    r = _ingest(client, TWITTER_CSV, 'twitter', 'twitter.csv')
    dq = r.json()['data_quality']
    assert dq['rows_received'] == 2
    assert dq['rows_skipped_no_handle'] == 0
    assert dq['profiles_with_malformed_profile_json'] == 0

    bad_csv = TWITTER_CSV.replace('twitter_handle', 'handle_typo')  # drops every handle
    r2 = _ingest(client, bad_csv, 'twitter', 'bad.csv')
    assert r2.json()['data_quality']['rows_skipped_no_handle'] == 2
    assert r2.json()['profiles_ingested'] == 0


def test_snapshot_history_grows_on_reingestion(client):
    _ingest(client, TWITTER_CSV, 'twitter', 'twitter.csv')
    first = client.get('/api/snapshots/twitter/janedoe/').json()
    assert first['count'] == 1

    _ingest(client, TWITTER_CSV, 'twitter', 'twitter.csv')  # re-score the same profile
    second = client.get('/api/snapshots/twitter/janedoe/').json()
    assert second['count'] == 2
    assert second['snapshots'][0]['scanned_at'] <= second['snapshots'][1]['scanned_at']


def test_recompute_models_force_retrain_param(client):
    # Need >= MIN_PROFILES_FOR_ML distinct profiles for clustering to run at all.
    rows_header = TWITTER_CSV.split('\n')[0]
    csv_rows = [rows_header]
    for i in range(10):
        csv_rows.append(
            f'{i},FOUNDER,Person {i},user{i},{i},{i}00,"hello world",https://x.com/{i},'
            f'2026-01-0{(i % 9) + 1}T10:00:00,general,{i * 2},1,0,0,{i * 50},0,'
            f'"{{""organization"": """", ""email"": null, ""followers_count"": {i * 100}, ""following_count"": 10}}"'
        )
    many_profiles_csv = '\n'.join(csv_rows) + '\n'
    _ingest(client, many_profiles_csv, 'twitter', 'many.csv')

    r1 = client.post('/api/recompute-models/', params={'platform': 'twitter'})
    assert r1.json()['diagnostics']['cluster_ids_stable_across_refits'] is True

    r2 = client.post('/api/recompute-models/', params={'platform': 'twitter', 'force_retrain': True})
    assert r2.json()['diagnostics']['cluster_ids_stable_across_refits'] is False
