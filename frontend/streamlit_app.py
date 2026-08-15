"""
OSINT-Guard — Streamlit frontend.

Pure Python UI for the FastAPI backend in ../backend. No HTML/CSS/JS.
Run:  streamlit run streamlit_app.py
"""

import base64
import io
import json
import os

import pandas as pd
import requests
import streamlit as st

DEFAULT_API = os.environ.get('OSINT_API_BASE', 'http://127.0.0.1:8000')

# Bulk ingestion runs one AI analysis per user, so it can take many minutes.
TIMEOUT_QUICK = 30
TIMEOUT_SCAN = 600
TIMEOUT_UPLOAD = 3600

CHART_TITLES = {
    'risk_distribution':  'Risk Score Distribution',
    'risk_tiers':         'Risk Tier Breakdown',
    'top_risk_users':     'Top Users by Risk Score',
    'engagement':         'Avg Engagement per User',
    'risk_vs_engagement': 'Risk Score vs Avg Engagement',
    'total_views':        'Total Tweet Views per User',
    'tweet_volume':       'Tweet Volume per User',
    'categories':         'Tweet Category Distribution',
}

st.set_page_config(page_title='OSINT-Guard', page_icon='🛡️', layout='wide')


# ── API helpers ─────────────────────────────────────────────────────────────
def api_url(path: str) -> str:
    return f"{st.session_state.api_base.rstrip('/')}{path}"


def api_get(path: str, timeout: int = TIMEOUT_QUICK) -> dict:
    r = requests.get(api_url(path), timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict, timeout: int = TIMEOUT_SCAN) -> dict:
    r = requests.post(api_url(path), json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def load_usernames(api_base: str) -> list:
    """Cached on api_base so switching hosts refetches. Never raises."""
    try:
        r = requests.get(f"{api_base.rstrip('/')}/api/usernames/", timeout=TIMEOUT_QUICK)
        data = r.json()
        return sorted(data.get('usernames') or []) if data.get('ok') else []
    except Exception:
        return []


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.session_state.setdefault('api_base', DEFAULT_API)

with st.sidebar:
    st.header('🛡️ OSINT-Guard')
    st.caption('Twitter risk analytics')
    st.session_state.api_base = st.text_input('Backend API', st.session_state.api_base)

    try:
        health = api_get('/api/health/', timeout=5)
        st.success(f"Connected — {health.get('service', 'api')}")
    except Exception as e:
        st.error('Backend unreachable')
        st.caption(str(e)[:200])
        st.caption('Start it with: `uvicorn app.main:app --reload` in ../backend')

    if st.button('Refresh username cache', width='stretch'):
        load_usernames.clear()
        st.rerun()

st.title('OSINT-Guard')
st.caption('Enterprise Twitter risk analytics platform')

tab_scan, tab_upload, tab_analytics, tab_db = st.tabs(
    ['🔍 Target Scan', '📥 Bulk Ingestion', '📊 Analytics', '🗄️ Database']
)


# ── 1. Target scan ──────────────────────────────────────────────────────────
with tab_scan:
    st.subheader('Live Target Scan')
    st.caption('Looks the handle up in the database first, then falls back to a live Apify scrape + AI analysis.')

    usernames = load_usernames(st.session_state.api_base)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        if usernames:
            picked = st.selectbox(
                'Known handle', ['— type a new handle below —'] + usernames, index=0
            )
        else:
            picked = None
            st.info('No profiles stored yet. Ingest a dataset or scan a live handle.')
        typed = st.text_input('Twitter handle', placeholder='elonmusk')
    with col_b:
        st.write('')
        st.write('')
        force_live = st.checkbox('Force live scan', help='Skip the database cache and re-scrape via Apify.')
        run = st.button('Run scan', type='primary', width='stretch')

    handle = (typed or '').strip().lstrip('@')
    if not handle and picked and not picked.startswith('—'):
        handle = picked

    if run:
        if not handle:
            st.warning('Pick or type a handle first.')
        else:
            profile = None
            if not force_live:
                with st.spinner(f'Checking database for @{handle}…'):
                    try:
                        res = api_get(f'/api/profile/{handle}/')
                        if res.get('ok'):
                            profile = res['profile']
                    except Exception as e:
                        st.error(f'Lookup failed: {e}')

            if profile:
                st.success(f"@{profile['username']} — served from database cache")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric('Risk score', f"{profile.get('risk_score', '?')}/100")
                m2.metric('Tweets', profile.get('total_tweets', 0))
                m3.metric('Avg engagement', profile.get('avg_engagement', 0))
                m4.metric('Categories', len(profile.get('categories') or []))
                if profile.get('risk_label'):
                    st.info(profile['risk_label'])
                if profile.get('categories'):
                    st.write('**Categories:** ' + ', '.join(map(str, profile['categories'])))
                rows = profile.get('inference_rows') or []
                if rows:
                    st.markdown('**Key PII extractions**')
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            else:
                with st.spinner(f'Scraping @{handle} via Apify — this can take a few minutes…'):
                    try:
                        collected = api_post('/api/collect/', {'twitter': handle}, TIMEOUT_SCAN)
                    except Exception as e:
                        collected = {'ok': False, 'error': str(e)}

                if not collected.get('ok'):
                    st.error(collected.get('error', 'Collection failed.'))
                else:
                    tweets = (collected.get('data') or {}).get('twitter') or []
                    st.write(f'Collected **{len(tweets)}** tweets.')
                    for platform, msg in (collected.get('errors') or {}).items():
                        st.warning(f'{platform}: {msg}')

                    with st.spinner('Running AI risk analysis…'):
                        try:
                            bundle = api_post(
                                '/api/insights/',
                                {'username': handle, 'datasets': collected['data']},
                                TIMEOUT_SCAN,
                            )
                        except Exception as e:
                            bundle = {'ok': False, 'error': str(e)}

                    if not bundle.get('ok'):
                        st.error(bundle.get('error', 'AI analysis failed.'))
                        if bundle.get('detail'):
                            st.code(bundle['detail'])
                    else:
                        load_usernames.clear()
                        executive = bundle.get('executive') or {}
                        st.metric('Risk score', f"{executive.get('riskScore', '?')}/{executive.get('riskMax', 100)}")
                        st.info(executive.get('riskLabel', ''))
                        if executive.get('summary'):
                            st.write(executive['summary'])

                        rows = bundle.get('inferenceRows') or []
                        if rows:
                            st.markdown('**Inferred information**')
                            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

                        pol = bundle.get('patternOfLife') or []
                        if pol:
                            st.markdown('**Pattern of life**')
                            st.dataframe(pd.DataFrame(pol), width='stretch', hide_index=True)
                        if bundle.get('polInference'):
                            st.caption(bundle['polInference'])

                        sims = bundle.get('phishingSims') or []
                        if sims:
                            st.markdown('**Phishing simulations**')
                            for sim in sims:
                                with st.expander(sim.get('title', 'Simulation')):
                                    st.write(f"**From:** {sim.get('from', '')}")
                                    st.write(f"**To:** {sim.get('to', '')}")
                                    st.write(f"**Subject:** {sim.get('subject', '')}")
                                    st.text(sim.get('body', ''))
                                    st.caption(sim.get('why', ''))

                        method = bundle.get('methodology') or {}
                        if method.get('pillars'):
                            with st.expander(method.get('headline', 'Methodology')):
                                st.write(method.get('intro', ''))
                                for pillar in method['pillars']:
                                    st.markdown(f"**{pillar.get('title', '')}** — {pillar.get('subtitle', '')}")
                                    st.write(pillar.get('text', ''))


# ── 2. Bulk ingestion ───────────────────────────────────────────────────────
with tab_upload:
    st.subheader('Bulk Ingestion')
    st.caption('Upload a CSV or JSON export of tweets. Rows are grouped by username and each user is scored by the AI.')

    uploaded = st.file_uploader('Dataset', type=['csv', 'json'])

    if uploaded is not None:
        raw = uploaded.getvalue()
        try:
            if uploaded.name.endswith('.csv'):
                preview = pd.read_csv(io.BytesIO(raw))
            else:
                preview = pd.DataFrame(json.loads(raw.decode('utf-8')))
            st.write(f'**{len(preview)}** rows, **{len(preview.columns)}** columns')
            st.dataframe(preview.head(20), width='stretch')
        except Exception as e:
            st.warning(f'Could not preview file locally ({e}); it can still be uploaded.')

        if st.button('Ingest and analyze', type='primary'):
            with st.spinner('Uploading and running AI analysis per user — this can take a long time…'):
                try:
                    r = requests.post(
                        api_url('/api/upload-bulk/'),
                        files={'file': (uploaded.name, raw)},
                        timeout=TIMEOUT_UPLOAD,
                    )
                    result = r.json()
                except Exception as e:
                    result = {'error': str(e)}

            if result.get('error'):
                st.error(result['error'])
            else:
                st.success(result.get('message', 'Upload complete.'))
                load_usernames.clear()


# ── 3. Analytics ────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader('Platform Analytics')
    st.caption('Seaborn/matplotlib charts rendered by the backend across every stored profile.')

    if st.button('Generate all charts', type='primary'):
        with st.spinner('Building charts…'):
            try:
                st.session_state.analytics = api_get('/api/analytics/', timeout=TIMEOUT_SCAN)
            except Exception as e:
                st.session_state.analytics = {'ok': False, 'error': str(e)}

    analytics = st.session_state.get('analytics')
    if analytics:
        if not analytics.get('ok'):
            st.error(analytics.get('error', 'Analytics failed.'))
        else:
            st.metric('Users in database', analytics.get('total_users', 0))
            charts = analytics.get('charts') or {}
            keys = [k for k in CHART_TITLES if k in charts]
            for i in range(0, len(keys), 2):
                for col, key in zip(st.columns(2), keys[i:i + 2]):
                    with col:
                        st.markdown(f'**{CHART_TITLES[key]}**')
                        st.image(base64.b64decode(charts[key]), width='stretch')


# ── 4. Database ─────────────────────────────────────────────────────────────
with tab_db:
    st.subheader('Stored Profiles')

    col_load, col_clear = st.columns([1, 1])
    if col_load.button('Load profiles', width='stretch'):
        with st.spinner('Fetching…'):
            try:
                st.session_state.db = api_get('/api/check-db/', timeout=TIMEOUT_SCAN)
            except Exception as e:
                st.session_state.db = {'error': str(e)}

    with col_clear.popover('Wipe database', width='stretch'):
        st.warning('This permanently deletes every stored profile.')
        if st.button('Yes, delete everything', type='primary'):
            try:
                res = api_post('/api/clear-db/', {}, TIMEOUT_QUICK)
                st.success(res.get('message', 'Cleared.'))
                st.session_state.pop('db', None)
                load_usernames.clear()
            except Exception as e:
                st.error(f'Clear failed: {e}')

    db = st.session_state.get('db')
    if db:
        if db.get('error'):
            st.error(db['error'])
        else:
            profiles = db.get('profiles') or []
            st.caption(f"{db.get('total_db_records', 0)} records total — showing {len(profiles)} most recent")

            flat = [
                {
                    'username': p['username'],
                    'risk_score': p.get('risk_score'),
                    'risk_label': p.get('risk_label'),
                    **{k: v for k, v in (p.get('tweet_stats') or {}).items() if k != 'categories'},
                    'categories': ', '.join(map(str, (p.get('tweet_stats') or {}).get('categories') or [])),
                    'scanned_at': p.get('scanned_at'),
                }
                for p in profiles
            ]
            if flat:
                df = pd.DataFrame(flat)
                st.dataframe(df, width='stretch', hide_index=True)
                st.download_button(
                    'Download as CSV',
                    df.to_csv(index=False).encode('utf-8'),
                    'osint_profiles.csv',
                    'text/csv',
                )

                pick = st.selectbox('Inspect a profile', [p['username'] for p in profiles])
                detail = next(p for p in profiles if p['username'] == pick)
                if detail.get('risk_signals'):
                    st.markdown('**Risk signals**')
                    st.dataframe(pd.DataFrame(detail['risk_signals']), width='stretch', hide_index=True)
                if detail.get('pattern_of_life'):
                    st.markdown('**Pattern of life**')
                    st.dataframe(pd.DataFrame(detail['pattern_of_life']), width='stretch', hide_index=True)
            else:
                st.info('No profiles stored yet.')
