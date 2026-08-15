"""
OSINT-Guard — Streamlit frontend.

Pure Python UI for the FastAPI backend in ../backend. No HTML/CSS/JS authored
here — theming comes from .streamlit/config.toml, charts from Plotly's
native Streamlit integration.

Primary intelligence path is fully local (rule-based risk engine + spaCy NER
+ Isolation Forest/KMeans, see backend/app/services/): no OpenAI key needed.
The original OpenAI-powered scan/upload flow is kept as an optional,
clearly-labeled "advanced" path for users who add a key later.

Run:  streamlit run streamlit_app.py
"""

import io
import json
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DEFAULT_API = os.environ.get('OSINT_API_BASE', 'http://127.0.0.1:8000')

TIMEOUT_QUICK = 30
TIMEOUT_SCAN = 600
TIMEOUT_UPLOAD = 3600

TIER_COLORS = {'Low': '#4ade80', 'Medium': '#fbbf24', 'High': '#f87171', 'Critical': '#991b1b'}

st.set_page_config(page_title='OSINT-Guard', page_icon='\U0001F6E1️', layout='wide')


# ── API helpers ─────────────────────────────────────────────────────────────
def api_url(path: str) -> str:
    return f"{st.session_state.api_base.rstrip('/')}{path}"


def api_get(path: str, params: dict | None = None, timeout: int = TIMEOUT_QUICK) -> dict:
    r = requests.get(api_url(path), params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict, timeout: int = TIMEOUT_SCAN) -> dict:
    r = requests.post(api_url(path), json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post_file(path: str, file_bytes: bytes, filename: str, form: dict, timeout: int = TIMEOUT_UPLOAD) -> dict:
    r = requests.post(
        api_url(path), files={'file': (filename, file_bytes)}, data=form, timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def load_usernames(api_base: str, platform: str) -> list:
    try:
        r = requests.get(f"{api_base.rstrip('/')}/api/usernames/", params={'platform': platform}, timeout=TIMEOUT_QUICK)
        data = r.json()
        return sorted(data.get('usernames') or []) if data.get('ok') else []
    except Exception:
        return []


@st.cache_data(ttl=30, show_spinner=False)
def load_profiles_summary(api_base: str, platform: str) -> pd.DataFrame:
    try:
        r = requests.get(f"{api_base.rstrip('/')}/api/profiles-summary/", params={'platform': platform}, timeout=TIMEOUT_SCAN)
        data = r.json()
        return pd.DataFrame(data.get('profiles') or [])
    except Exception:
        return pd.DataFrame()


def tier_for(score) -> str:
    if score is None:
        return 'Unknown'
    if score < 40:
        return 'Low'
    if score < 70:
        return 'Medium'
    if score < 90:
        return 'High'
    return 'Critical'


def clear_summary_cache():
    load_profiles_summary.clear()
    load_usernames.clear()


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.session_state.setdefault('api_base', DEFAULT_API)
st.session_state.setdefault('platform', 'twitter')

with st.sidebar:
    st.header('\U0001F6E1️ OSINT-Guard')
    st.caption('Local, explainable OSINT risk analytics — no LLM required')
    st.session_state.api_base = st.text_input('Backend API', st.session_state.api_base)

    try:
        health = api_get('/api/health/', timeout=5)
        st.success(f"Connected — {health.get('service', 'api')}")
    except Exception as e:
        st.error('Backend unreachable')
        st.caption(str(e)[:200])
        st.caption('Start it with: `uvicorn app.main:app --reload` in ../backend')

    if st.button('Refresh cached data', width='stretch'):
        clear_summary_cache()
        st.rerun()

    st.divider()
    st.caption(
        'Risk scores come from a deterministic, weighted rule engine + local '
        'spaCy NER + Isolation Forest/KMeans — every point is traceable to a '
        'signal. See the **Model Insights** tab for the full methodology.'
    )


st.title('OSINT-Guard')
st.caption('Local intelligence platform — Twitter engagement + LinkedIn exposure risk, zero external API calls')

try:
    platform = st.segmented_control(
        'Platform', options=['twitter', 'linkedin'],
        format_func=lambda p: 'Twitter / X' if p == 'twitter' else 'LinkedIn',
        key='platform',
    )
except Exception:
    platform = st.radio(
        'Platform', options=['twitter', 'linkedin'],
        format_func=lambda p: 'Twitter / X' if p == 'twitter' else 'LinkedIn',
        horizontal=True, key='platform',
    )
platform = platform or 'twitter'

# ── KPI header (always visible, current platform) ───────────────────────────
summary_df = load_profiles_summary(st.session_state.api_base, platform)

k1, k2, k3, k4 = st.columns(4)
with k1, st.container(border=True):
    st.metric('Profiles ingested', len(summary_df))
with k2, st.container(border=True):
    avg_risk = round(summary_df['risk_score'].dropna().mean(), 1) if not summary_df.empty and summary_df['risk_score'].notna().any() else 0
    st.metric('Avg risk score', f'{avg_risk}/100')
with k3, st.container(border=True):
    high_risk = int((summary_df['risk_score'] >= 70).sum()) if not summary_df.empty else 0
    st.metric('High/Critical profiles', high_risk)
with k4, st.container(border=True):
    flagged = int((summary_df['anomaly_score'] >= 70).sum()) if not summary_df.empty and 'anomaly_score' in summary_df else 0
    st.metric('Anomaly-flagged', flagged, help='Isolation Forest score >= 70/100')

st.divider()

tab_lookup, tab_upload, tab_analytics, tab_insights, tab_db = st.tabs(
    ['\U0001F50D Target Lookup', '\U0001F4E5 Bulk Ingestion', '\U0001F4CA Analytics', '\U0001F9E0 Model Insights', '\U0001F5C4️ Database']
)


# ── 1. Target Lookup ─────────────────────────────────────────────────────────
with tab_lookup:
    st.subheader('Target Lookup')
    st.caption('Looks up an already-ingested profile and shows the full itemized risk breakdown — every point traceable to a signal.')

    usernames = load_usernames(st.session_state.api_base, platform)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        picked = st.selectbox('Known profile', ['— type below —'] + usernames, index=0) if usernames else None
        typed = st.text_input('Handle / public identifier', placeholder='jxnlco' if platform == 'twitter' else 'jane-smith-123')
    with col_b:
        st.write('')
        st.write('')
        run = st.button('Look up', type='primary', width='stretch')

    identifier = (typed or '').strip().lstrip('@')
    if not identifier and picked and not picked.startswith('—'):
        identifier = picked

    if run and identifier:
        with st.spinner(f'Looking up {identifier}…'):
            try:
                res = api_get(f'/api/profile-detail/{platform}/{identifier}/')
            except Exception as e:
                res = {'ok': False, 'error': str(e)}

        if not res.get('ok'):
            st.warning(res.get('error', 'Not found. Ingest a dataset first in the Bulk Ingestion tab.'))
        else:
            p = res['profile']
            tier = tier_for(p['risk_score'])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Risk score', f"{p['risk_score']}/100")
            m2.metric('Tier', tier)
            m3.metric('Anomaly score', p.get('anomaly_score') if p.get('anomaly_score') is not None else '—')
            m4.metric('Cluster', p.get('cluster_id') if p.get('cluster_id') is not None else '—')
            st.progress(min(int(p['risk_score']), 100) / 100)
            st.info(p.get('risk_label', ''))

            factors = p.get('risk_factors') or []
            if factors:
                st.markdown('**Why this score — itemized factors**')
                fdf = pd.DataFrame(factors)[['category', 'label', 'points', 'max_points', 'evidence']]
                st.dataframe(fdf, width='stretch', hide_index=True)

            rows = p.get('inference_rows') or []
            if rows:
                st.markdown('**Extracted entities / signals (local NER)**')
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    elif run:
        st.warning('Pick or type an identifier first.')

    if platform == 'twitter':
        with st.expander('Advanced: live scan via Apify + OpenAI (optional, requires API keys)'):
            st.caption('Only needed if you want to scrape a brand-new handle and score it with an LLM instead of the local engine.')
            live_handle = st.text_input('Twitter handle to scrape live', key='live_handle')
            if st.button('Run live scan'):
                if not live_handle:
                    st.warning('Enter a handle first.')
                else:
                    with st.spinner(f'Scraping @{live_handle} via Apify…'):
                        try:
                            collected = api_post('/api/collect/', {'twitter': live_handle}, TIMEOUT_SCAN)
                        except Exception as e:
                            collected = {'ok': False, 'error': str(e)}
                    if not collected.get('ok'):
                        st.error(collected.get('error', 'Collection failed — check APIFY_API_TOKEN in backend/.env.'))
                    else:
                        tweets = (collected.get('data') or {}).get('twitter') or []
                        st.write(f'Collected **{len(tweets)}** tweets.')
                        with st.spinner('Running AI risk analysis…'):
                            try:
                                bundle = api_post('/api/insights/', {'username': live_handle, 'datasets': collected['data']}, TIMEOUT_SCAN)
                            except Exception as e:
                                bundle = {'ok': False, 'error': str(e)}
                        if not bundle.get('ok'):
                            st.error(bundle.get('error', 'AI analysis failed — check OPENAI_API_KEY in backend/.env.'))
                        else:
                            clear_summary_cache()
                            executive = bundle.get('executive') or {}
                            st.metric('AI risk score', f"{executive.get('riskScore', '?')}/{executive.get('riskMax', 100)}")
                            st.write(executive.get('summary', ''))


# ── 2. Bulk Ingestion ────────────────────────────────────────────────────────
with tab_upload:
    st.subheader(f'Bulk Ingestion — {"Twitter/X" if platform == "twitter" else "LinkedIn"}')
    st.caption(
        'Primary path runs entirely locally: real engagement/profile stats + spaCy NER + '
        'the weighted rule-based risk engine. No API key required.'
    )

    expected_cols = (
        'id, source_type, person_name, twitter_handle, ..., content, created_at, category, '
        'favorite_count, ..., people_profile_json'
        if platform == 'twitter' else
        'id, public_identifier, first_name, last_name, headline, about, location_city/state/country, '
        'experiences_json, connections_count, follower_count, open_to_work, is_hiring, ...'
    )
    st.caption(f'Expected columns ({platform}): {expected_cols}')

    uploaded = st.file_uploader('Dataset CSV', type=['csv'], key=f'uploader_{platform}')

    if uploaded is not None:
        raw = uploaded.getvalue()
        try:
            preview = pd.read_csv(io.BytesIO(raw))
            st.write(f'**{len(preview)}** rows, **{len(preview.columns)}** columns')
            st.dataframe(preview.head(20), width='stretch')
        except Exception as e:
            st.warning(f'Could not preview file locally ({e}); it can still be uploaded.')

        if st.button('\U0001F680 Ingest locally (no AI, no API key)', type='primary'):
            with st.spinner('Parsing rows, computing stats, running NER + risk scoring…'):
                try:
                    result = api_post_file('/api/ingest-local/', raw, uploaded.name, {'platform': platform})
                except Exception as e:
                    result = {'ok': False, 'error': str(e)}

            if not result.get('ok'):
                st.error(result.get('error', 'Ingestion failed.'))
            else:
                st.success(result.get('message'))
                clear_summary_cache()
                with st.spinner('Fitting Isolation Forest + KMeans across the updated population…'):
                    try:
                        r = requests.post(api_url('/api/recompute-models/'), params={'platform': platform}, timeout=TIMEOUT_SCAN)
                        recompute = r.json()
                    except Exception as e:
                        recompute = {'ok': False, 'error': str(e)}
                if recompute.get('ok'):
                    diag = recompute['diagnostics']
                    if diag.get('ok'):
                        st.success(f"Models refit: k={diag['k']} clusters, silhouette={diag['silhouette_score']}, {diag['flagged_anomalies']} anomalies flagged.")
                    else:
                        st.info(diag.get('reason', 'Not enough profiles yet for anomaly/cluster models.'))
                clear_summary_cache()

    with st.expander('Advanced: AI-powered ingestion (Twitter only, requires OPENAI_API_KEY)'):
        st.caption('Legacy path — scores every user with 5 chained OpenAI calls instead of the local engine.')
        ai_uploaded = st.file_uploader('Dataset (CSV/JSON)', type=['csv', 'json'], key='ai_uploader')
        if ai_uploaded is not None and st.button('Ingest and analyze with AI'):
            with st.spinner('Uploading and running AI analysis per user — this can take a long time…'):
                try:
                    r = requests.post(api_url('/api/upload-bulk/'), files={'file': (ai_uploaded.name, ai_uploaded.getvalue())}, timeout=TIMEOUT_UPLOAD)
                    result = r.json()
                except Exception as e:
                    result = {'error': str(e)}
            if result.get('error'):
                st.error(result['error'])
            else:
                st.success(result.get('message', 'Upload complete.'))
                clear_summary_cache()


# ── 3. Analytics ─────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader(f'Analytics — {"Twitter/X" if platform == "twitter" else "LinkedIn"}')
    st.caption('Interactive charts built from the local risk engine + Isolation Forest/KMeans output — every chart reflects a real, computed signal.')

    df = summary_df
    if df.empty:
        st.info('No profiles ingested yet for this platform. Use the Bulk Ingestion tab.')
    else:
        df = df.copy()
        df['tier'] = df['risk_score'].apply(tier_for)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x='risk_score', nbins=15, title='Risk Score Distribution', color_discrete_sequence=['#38bdf8'])
            st.plotly_chart(fig, width='stretch')
        with c2:
            tier_counts = df['tier'].value_counts().reindex(['Low', 'Medium', 'High', 'Critical']).dropna()
            fig = px.pie(names=tier_counts.index, values=tier_counts.values, title='Risk Tier Breakdown',
                         color=tier_counts.index, color_discrete_map=TIER_COLORS)
            st.plotly_chart(fig, width='stretch')

        top15 = df.sort_values('risk_score', ascending=False).head(15)
        name_col = 'username' if platform == 'twitter' else 'full_name'
        fig = px.bar(top15.sort_values('risk_score'), x='risk_score', y=name_col, orientation='h',
                     title='Top 15 Profiles by Risk Score', color='risk_score', color_continuous_scale='Reds')
        st.plotly_chart(fig, width='stretch')

        c3, c4 = st.columns(2)
        if platform == 'twitter':
            with c3:
                fig = px.bar(top15.sort_values('avg_engagement'), x='avg_engagement', y='username', orientation='h',
                             title='Avg Engagement per Profile (Top 15 by risk)', color_discrete_sequence=['#38bdf8'])
                st.plotly_chart(fig, width='stretch')
            with c4:
                fig = px.scatter(df, x='avg_engagement', y='risk_score', hover_name='username', color='tier',
                                  color_discrete_map=TIER_COLORS, title='Risk Score vs Avg Engagement')
                st.plotly_chart(fig, width='stretch')

            cat_rows = []
            for _, row in df.iterrows():
                for cat in (row.get('categories') or []):
                    cat_rows.append(cat)
            if cat_rows:
                cat_df = pd.Series(cat_rows).value_counts().reset_index()
                cat_df.columns = ['category', 'count']
                fig = px.bar(cat_df, x='count', y='category', orientation='h', title='Tweet Category Distribution',
                             color_discrete_sequence=['#34d399'])
                st.plotly_chart(fig, width='stretch')
        else:
            with c3:
                top15 = top15.assign(reach=top15['connections_count'].fillna(0) + top15['follower_count'].fillna(0))
                fig = px.bar(top15.sort_values('reach'), x='reach', y='full_name', orientation='h',
                             title='Professional Reach (Top 15 by risk)', color_discrete_sequence=['#38bdf8'])
                st.plotly_chart(fig, width='stretch')
            with c4:
                df_reach = df.assign(reach=df['connections_count'].fillna(0) + df['follower_count'].fillna(0))
                fig = px.scatter(df_reach, x='reach', y='risk_score', hover_name='full_name', color='tier',
                                  color_discrete_map=TIER_COLORS, title='Risk Score vs Professional Reach')
                st.plotly_chart(fig, width='stretch')

            c5, c6 = st.columns(2)
            with c5:
                loc = df['location_city'].replace('', pd.NA).fillna(df['location_state']).fillna(df['location_country'])
                loc_counts = loc.dropna().value_counts().head(15).reset_index()
                loc_counts.columns = ['location', 'count']
                if not loc_counts.empty:
                    fig = px.bar(loc_counts, x='count', y='location', orientation='h',
                                 title='Most-Disclosed Locations (Direct PII Exposure)', color_discrete_sequence=['#34d399'])
                    st.plotly_chart(fig, width='stretch')
            with c6:
                status = pd.Series({
                    'Open to Work': int(df['open_to_work'].sum()),
                    'Hiring': int(df['is_hiring'].sum()),
                    'Neither flag set': int(((~df['open_to_work']) & (~df['is_hiring'])).sum()),
                })
                status = status[status > 0]
                if not status.empty:
                    fig = px.pie(names=status.index, values=status.values, title='Job-Status Exposure (Targeting Risk)',
                                 color_discrete_sequence=['#fbbf24', '#f87171', '#4ade80'])
                    st.plotly_chart(fig, width='stretch')

        if 'anomaly_score' in df.columns and df['anomaly_score'].notna().any():
            c7, c8 = st.columns(2)
            with c7:
                fig = px.histogram(df.dropna(subset=['anomaly_score']), x='anomaly_score', nbins=15,
                                    title='Anomaly Score Distribution (Isolation Forest)', color_discrete_sequence=['#a78bfa'])
                fig.add_vline(x=70, line_dash='dash', line_color='#991b1b', annotation_text='Flag threshold')
                st.plotly_chart(fig, width='stretch')
            with c8:
                if df['cluster_id'].notna().any():
                    cl = df.dropna(subset=['cluster_id'])['cluster_id'].astype(int).value_counts().sort_index()
                    fig = px.bar(x=[f'Cluster {i}' for i in cl.index], y=cl.values,
                                 title='Persona Cluster Sizes (KMeans)', color_discrete_sequence=['#38bdf8'])
                    st.plotly_chart(fig, width='stretch')
        else:
            st.info('Run "Recompute anomaly/cluster models" (Bulk Ingestion tab or Model Insights tab) to unlock anomaly/cluster charts.')

    with st.expander('Export static PNG charts (matplotlib, for reports/slides)'):
        if st.button('Generate PNG charts'):
            with st.spinner('Rendering…'):
                try:
                    st.session_state.png_charts = api_get('/api/analytics/', params={'platform': platform}, timeout=TIMEOUT_SCAN)
                except Exception as e:
                    st.session_state.png_charts = {'ok': False, 'error': str(e)}
        png = st.session_state.get('png_charts')
        if png:
            if not png.get('ok'):
                st.error(png.get('error', 'Failed.'))
            else:
                import base64
                charts = png.get('charts') or {}
                keys = list(charts)
                for i in range(0, len(keys), 2):
                    for col, key in zip(st.columns(2), keys[i:i + 2]):
                        with col:
                            st.markdown(f'**{key.replace("_", " ").title()}**')
                            st.image(base64.b64decode(charts[key]), width='stretch')


# ── 4. Model Insights ────────────────────────────────────────────────────────
with tab_insights:
    st.subheader('Model Insights')
    st.caption('Diagnostics for the unsupervised layer (Isolation Forest + KMeans), and how the risk score is computed.')

    if st.button('Recompute anomaly/cluster models', type='primary'):
        with st.spinner('Fitting Isolation Forest + KMeans…'):
            try:
                r = requests.post(api_url('/api/recompute-models/'), params={'platform': platform}, timeout=TIMEOUT_SCAN)
                st.session_state.model_diag = r.json()
            except Exception as e:
                st.session_state.model_diag = {'ok': False, 'error': str(e)}
        clear_summary_cache()

    diag = (st.session_state.get('model_diag') or {}).get('diagnostics')
    if diag:
        if diag.get('ok'):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric('Profiles used', diag['n_profiles'])
            d2.metric('Clusters (k)', diag['k'])
            d3.metric('Silhouette score', diag['silhouette_score'])
            d4.metric('Anomalies flagged', diag['flagged_anomalies'])
            st.caption(
                'Silhouette score ranges -1 to 1; above ~0.3 indicates reasonably '
                'separated clusters given real, unlabeled behavioral data. k was '
                'chosen via a fixed default (4) balanced against sample size — see README for the elbow-method extension.'
            )
        else:
            st.warning(diag.get('reason'))

    df = summary_df
    if not df.empty and 'anomaly_score' in df.columns and df['anomaly_score'].notna().any():
        st.markdown('#### Flagged anomalies (score ≥ 70)')
        flagged_df = df[df['anomaly_score'] >= 70].sort_values('anomaly_score', ascending=False)
        name_col = 'username' if platform == 'twitter' else 'full_name'
        cols = [name_col, 'risk_score', 'anomaly_score', 'cluster_id']
        st.dataframe(flagged_df[cols], width='stretch', hide_index=True)

        st.markdown('#### Cluster profiles')
        cl_summary = df.dropna(subset=['cluster_id']).groupby('cluster_id').agg(
            profiles=('risk_score', 'count'), avg_risk_score=('risk_score', 'mean'),
        ).round(1).reset_index()
        st.dataframe(cl_summary, width='stretch', hide_index=True)

    st.divider()
    st.markdown('#### How the risk score is computed')
    if platform == 'twitter':
        st.markdown(
            """
The score is a **deterministic, weighted sum** across four buckets (no black box, no training data):

| Bucket | Max points | Driven by |
| --- | --- | --- |
| PII exposure | 40 | Email/LinkedIn/employer disclosed in profile metadata + locations mentioned in tweets (spaCy NER) |
| Behavioral predictability | 25 | Entropy of posting-hour histogram, **confidence-weighted by tweet count** so a single tweet can't fake a "routine" |
| Content sensitivity | 20 | Keyword-dictionary hits for travel/home/family/finance topics in tweet text |
| Exposure reach | 15 | Follower count (log-scaled) |

Separately, an **Isolation Forest** flags statistically anomalous accounts (unusual followers/following/engagement ratios) — an authenticity signal, not folded into the risk score itself. **KMeans** groups accounts into behavioral personas.
            """
        )
    else:
        st.markdown(
            """
The score is a **deterministic, weighted sum** across four buckets (no black box, no training data):

| Bucket | Max points | Driven by |
| --- | --- | --- |
| PII exposure | 40 | Explicit city/state/country disclosure (city worth more than country — it's exact, not inferred), employer, education, NER entities in headline/about |
| Career-timeline predictability | 25 | Number of distinct employers in `experiences_json` (a longer dated history is more reconstructable) |
| Content/status sensitivity | 20 | `open_to_work` / `is_hiring` flags — direct recruitment-scam/BEC targeting signals |
| Exposure reach | 15 | Connections + followers (log-scaled) |

Same Isolation Forest / KMeans layer as Twitter, fit on LinkedIn-specific features (connection ratios, employer count, premium/verified/influencer flags).
            """
        )


# ── 5. Database ──────────────────────────────────────────────────────────────
with tab_db:
    st.subheader(f'Stored Profiles — {"Twitter/X" if platform == "twitter" else "LinkedIn"}')

    col_load, col_clear = st.columns([1, 1])
    if col_load.button('Load profiles', width='stretch'):
        with st.spinner('Fetching…'):
            try:
                st.session_state.db = api_get('/api/check-db/', params={'platform': platform}, timeout=TIMEOUT_SCAN)
            except Exception as e:
                st.session_state.db = {'error': str(e)}

    with col_clear.popover('Wipe database', width='stretch'):
        st.warning(f'This permanently deletes every stored {platform} profile.')
        if st.button('Yes, delete everything', type='primary'):
            try:
                r = requests.post(api_url('/api/clear-db/'), params={'platform': platform}, timeout=TIMEOUT_QUICK)
                res = r.json()
                st.success(res.get('message', 'Cleared.'))
                st.session_state.pop('db', None)
                clear_summary_cache()
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
                    'anomaly_score': p.get('anomaly_score'),
                    'cluster_id': p.get('cluster_id'),
                    **{k: v for k, v in (p.get('tweet_stats') or {}).items() if not isinstance(v, (list, dict))},
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
                    f'osint_{platform}_profiles.csv',
                    'text/csv',
                )

                pick = st.selectbox('Inspect a profile', [p['username'] for p in profiles])
                with st.spinner('Loading risk breakdown…'):
                    try:
                        detail = api_get(f'/api/profile-detail/{platform}/{pick}/')
                    except Exception as e:
                        detail = {'ok': False, 'error': str(e)}
                if detail.get('ok'):
                    prof = detail['profile']
                    st.markdown(f"**Risk factors for `{pick}`**")
                    factors = prof.get('risk_factors') or []
                    if factors:
                        st.dataframe(pd.DataFrame(factors), width='stretch', hide_index=True)
                    rows = prof.get('inference_rows') or []
                    if rows:
                        st.markdown('**Extracted entities / signals**')
                        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            else:
                st.info('No profiles stored yet.')
