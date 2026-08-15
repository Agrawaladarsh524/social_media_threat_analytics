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
import plotly.io as pio
import requests
import streamlit as st

DEFAULT_API = os.environ.get('OSINT_API_BASE', 'http://127.0.0.1:8000')

TIMEOUT_QUICK = 30
TIMEOUT_SCAN = 600
TIMEOUT_UPLOAD = 3600

# Severity palette — semantic, deliberately distinct from the accent color so
# "this needs attention" never reads as decoration.
TIER_COLORS = {'Low': '#4ade80', 'Medium': '#fbbf24', 'High': '#f87171', 'Critical': '#991b1b'}
TIER_ORDER = ['Low', 'Medium', 'High', 'Critical']
TIER_ICONS = {'Low': ':material/check_circle:', 'Medium': ':material/info:',
              'High': ':material/warning:', 'Critical': ':material/dangerous:'}
ACCENT = '#38bdf8'
CHART_H = 320

st.set_page_config(
    page_title='OSINT-Guard',
    page_icon='\U0001F6E1️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# One chart identity across every figure — transparent plot area so charts sit
# on the app background instead of in grey boxes, muted grid, consistent font.
pio.templates['osint'] = pio.templates['plotly_dark']
pio.templates['osint'].layout.update(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='sans-serif', size=12, color='#cbd5e1'),
    title=dict(font=dict(size=14, color='#e2e8f0'), x=0, xanchor='left', pad=dict(b=12)),
    margin=dict(l=8, r=8, t=44, b=8),
    xaxis=dict(gridcolor='rgba(148,163,184,0.12)', zerolinecolor='rgba(148,163,184,0.2)'),
    yaxis=dict(gridcolor='rgba(148,163,184,0.12)', zerolinecolor='rgba(148,163,184,0.2)'),
    hoverlabel=dict(font_size=12),
    legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1, title_text=''),
)
pio.templates.default = 'osint'


def styled(fig, height: int = CHART_H, legend: bool = False):
    """Applies the shared chart identity: fixed height so rows align, no
    redundant axis titles (the chart title already says it), legend off by
    default since most charts here encode one series."""
    fig.update_layout(height=height, showlegend=legend, xaxis_title=None, yaxis_title=None)
    return fig


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
    st.title('\U0001F6E1️ OSINT-Guard')
    st.caption('Local, explainable OSINT exposure analytics')

    try:
        health = api_get('/api/health/', timeout=5)
        st.success(f"Connected — {health.get('service', 'api')}", icon=':material/check_circle:')
        backend_up = True
    except Exception as e:
        st.error('Backend unreachable', icon=':material/error:')
        st.caption(str(e)[:160])
        st.code('uvicorn app.main:app --reload', language='bash')
        backend_up = False

    st.subheader('Data source', divider='gray')
    platform = st.segmented_control(
        'Platform', options=['twitter', 'linkedin'],
        format_func=lambda p: 'Twitter / X' if p == 'twitter' else 'LinkedIn',
        key='platform', label_visibility='collapsed', width='stretch',
    ) or 'twitter'
    st.caption('Scopes every tab below to this platform.')

    with st.expander('Connection & cache'):
        st.session_state.api_base = st.text_input('Backend API', st.session_state.api_base)
        if st.button('Refresh cached data', width='stretch', icon=':material/refresh:'):
            clear_summary_cache()
            st.rerun()

    st.subheader('How scoring works', divider='gray')
    st.caption(
        'A deterministic weighted rule engine + local spaCy NER + sentence-embedding '
        'similarity, with Isolation Forest / KMeans on top. Every point is traceable '
        'to a specific signal — see **Model Insights** for the full methodology.'
    )
    st.info(
        '**Exposure risk** measures how much can be inferred from public data. '
        'It is not a prediction of malicious intent or victimhood.',
        icon=':material/policy:',
    )


st.title('Exposure Risk Intelligence')
st.caption(
    f'{"Twitter / X" if platform == "twitter" else "LinkedIn"} · '
    'deterministic scoring, local NLP, unsupervised anomaly detection — no external API calls'
)

# ── KPI header (always visible, scoped to the selected platform) ────────────
summary_df = load_profiles_summary(st.session_state.api_base, platform)

_scores = summary_df['risk_score'].dropna() if not summary_df.empty else pd.Series(dtype=float)
avg_risk = round(_scores.mean(), 1) if not _scores.empty else 0
high_risk = int((_scores >= 70).sum()) if not _scores.empty else 0
flagged = int((summary_df['anomaly_score'] >= 70).sum()) if not summary_df.empty and 'anomaly_score' in summary_df else 0
# Sorted scores make the metric sparkline read as a distribution curve rather
# than arbitrary row-order noise.
_spark = sorted(_scores.tolist()) if not _scores.empty else None

k1, k2, k3, k4 = st.columns(4, gap='medium')
k1.metric(
    'Profiles ingested', f'{len(summary_df):,}', border=True, icon=':material/database:',
    help='Unique profiles scored for this platform.',
)
k2.metric(
    'Avg exposure risk', f'{avg_risk}', delta=f'{tier_for(avg_risk)} tier', delta_color='off',
    border=True, icon=':material/speed:', chart_data=_spark, chart_type='area',
    help='Mean 0-100 score. Sparkline shows the sorted distribution across all profiles.',
)
k3.metric(
    'High / Critical', f'{high_risk:,}',
    delta=f'{high_risk / len(summary_df) * 100:.0f}% of population' if len(summary_df) else None,
    delta_color='off', border=True, icon=':material/warning:',
    help='Profiles scoring 70 or above.',
)
k4.metric(
    'Anomaly-flagged', f'{flagged:,}', border=True, icon=':material/scatter_plot:',
    help='Isolation Forest score >= 70. Relative to the profiles currently in the database, not an absolute probability.',
)

if summary_df.empty:
    st.warning(
        f'No {platform} profiles ingested yet — start in the **Bulk Ingestion** tab.',
        icon=':material/upload_file:',
    )

st.divider()

tab_lookup, tab_upload, tab_analytics, tab_insights, tab_db = st.tabs([
    ':material/search: Target Lookup',
    ':material/upload_file: Bulk Ingestion',
    ':material/bar_chart: Analytics',
    ':material/neurology: Model Insights',
    ':material/table: Database',
])


# ── 1. Target Lookup ─────────────────────────────────────────────────────────
with tab_lookup:
    st.subheader('Target Lookup')
    st.caption('Look up an ingested profile and see its full itemized exposure breakdown — every point traceable to a specific signal.')

    usernames = load_usernames(st.session_state.api_base, platform)
    with st.container(border=True):
        col_a, col_b, col_c = st.columns([2, 2, 1], vertical_alignment='bottom')
        picked = col_a.selectbox(
            'Pick a known profile', ['— type instead —'] + usernames, index=0,
            help='Every profile already ingested for this platform.',
        ) if usernames else None
        typed = col_b.text_input(
            'Or type an identifier',
            placeholder='jxnlco' if platform == 'twitter' else 'jane-smith-123',
            help='Twitter handle, or LinkedIn public identifier from the profile URL.',
        )
        run = col_c.button('Look up', type='primary', width='stretch', icon=':material/search:')

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
            st.warning(
                res.get('error', 'Not found. Ingest a dataset first in the Bulk Ingestion tab.'),
                icon=':material/search_off:',
            )
        else:
            p = res['profile']
            score = int(p['risk_score'])
            tier = tier_for(score)

            st.subheader(f'`{identifier}`', divider='gray')
            m1, m2, m3, m4 = st.columns(4, gap='medium')
            m1.metric('Exposure risk', f'{score}/100', border=True, icon=':material/speed:')
            m2.metric('Tier', tier, border=True, icon=TIER_ICONS.get(tier, ':material/help:'))
            m3.metric(
                'Anomaly score',
                p.get('anomaly_score') if p.get('anomaly_score') is not None else '—',
                border=True, icon=':material/scatter_plot:',
                help='Relative to the current population, not an absolute probability.',
            )
            m4.metric(
                'Persona cluster',
                p.get('cluster_id') if p.get('cluster_id') is not None else '—',
                border=True, icon=':material/group_work:',
                help='KMeans behavioral cluster. Stable across model refits.',
            )
            st.progress(min(score, 100) / 100, text=p.get('risk_label', ''))

            factors = p.get('risk_factors') or []
            if factors:
                st.markdown('#### Why this score')
                st.caption('Every point traces to one signal. Nothing here is inferred by a model.')
                fdf = pd.DataFrame(factors)
                bucket_totals = fdf.groupby('category', as_index=False)['points'].sum()
                st.plotly_chart(
                    styled(px.bar(
                        bucket_totals.sort_values('points'), x='points', y='category',
                        orientation='h', title='Points contributed by bucket',
                        color_discrete_sequence=[ACCENT], text='points',
                    ), height=220),
                    width='stretch',
                )
                st.dataframe(
                    fdf[['category', 'label', 'points', 'max_points', 'evidence']],
                    width='stretch', hide_index=True,
                    column_config={
                        'category': st.column_config.TextColumn('Bucket', width='small'),
                        'label': st.column_config.TextColumn('Signal', width='medium'),
                        'points': st.column_config.ProgressColumn(
                            'Points', min_value=0, max_value=float(fdf['max_points'].max()), format='%.1f',
                        ),
                        'max_points': st.column_config.NumberColumn('Max', width='small'),
                        'evidence': st.column_config.TextColumn('Evidence', width='large'),
                    },
                )

            rows = p.get('inference_rows') or []
            if rows:
                st.markdown('#### Extracted signals')
                st.caption('Real entities pulled from public text by local spaCy NER and sentence-embedding similarity.')
                st.dataframe(
                    pd.DataFrame(rows), width='stretch', hide_index=True,
                    column_config={
                        'targetInfo': st.column_config.TextColumn('Signal type', width='medium'),
                        'inferredValue': st.column_config.TextColumn('Value', width='medium'),
                        'evidence': st.column_config.TextColumn('Source evidence', width='large'),
                    },
                )

            try:
                hist = api_get(f'/api/snapshots/{platform}/{identifier}/')
            except Exception:
                hist = {'ok': False}
            snapshots = hist.get('snapshots') or []
            if len(snapshots) > 1:
                st.markdown('#### Score history')
                st.caption(
                    f'{len(snapshots)} scoring events. A row is appended every time this profile is '
                    're-ingested — real recorded history, never a backfilled projection.'
                )
                hdf = pd.DataFrame(snapshots)
                st.plotly_chart(
                    styled(px.line(
                        hdf, x='scanned_at', y='risk_score', markers=True,
                        title='Exposure risk over time', color_discrete_sequence=[ACCENT],
                    ), height=240),
                    width='stretch',
                )
                st.dataframe(hdf, width='stretch', hide_index=True)

    elif run:
        st.warning('Pick or type an identifier first.', icon=':material/edit:')

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
        '`twitter_handle`, `content`, `created_at`, `category`, `favorite_count`, '
        '`retweet_count`, `views_count`, `people_profile_json`'
        if platform == 'twitter' else
        '`public_identifier`, `first_name`, `last_name`, `headline`, `about`, '
        '`location_city` / `location_state` / `location_country`, `experiences_json`, '
        '`educations_json`, `connections_count`, `follower_count`, `open_to_work`, `is_hiring`'
    )
    with st.expander(f'Expected {platform} CSV columns'):
        st.markdown(expected_cols)
        st.caption('Extra columns are ignored. Missing ones are reported in the data-quality panel after ingestion.')

    uploaded = st.file_uploader('Dataset CSV', type=['csv'], key=f'uploader_{platform}')

    if uploaded is not None:
        raw = uploaded.getvalue()
        try:
            preview = pd.read_csv(io.BytesIO(raw))
            c1, c2 = st.columns(2, gap='medium')
            c1.metric('Rows in file', f'{len(preview):,}', border=True, icon=':material/table_rows:')
            c2.metric('Columns', len(preview.columns), border=True, icon=':material/view_column:')
            with st.expander('Preview first 20 rows'):
                st.dataframe(preview.head(20), width='stretch')
        except Exception as e:
            st.warning(f'Could not preview file locally ({e}); it can still be uploaded.', icon=':material/warning:')

        if st.button('Ingest locally — no AI, no API key', type='primary', icon=':material/rocket_launch:'):
            with st.status('Running the local pipeline…', expanded=True) as status:
                st.write('Parsing rows and computing engagement stats…')
                st.write('Running spaCy NER + semantic similarity…')
                st.write('Scoring with the weighted rule engine…')
                try:
                    result = api_post_file('/api/ingest-local/', raw, uploaded.name, {'platform': platform})
                except Exception as e:
                    result = {'ok': False, 'error': str(e)}

                if not result.get('ok'):
                    status.update(label='Ingestion failed', state='error')
                    st.error(result.get('error', 'Ingestion failed.'), icon=':material/error:')
                else:
                    clear_summary_cache()
                    st.write('Fitting Isolation Forest + KMeans…')
                    try:
                        r = requests.post(api_url('/api/recompute-models/'), params={'platform': platform}, timeout=TIMEOUT_SCAN)
                        recompute = r.json()
                    except Exception as e:
                        recompute = {'ok': False, 'error': str(e)}
                    clear_summary_cache()
                    status.update(label=result.get('message', 'Ingestion complete'), state='complete', expanded=False)

            if result.get('ok'):
                st.success(result.get('message'), icon=':material/check_circle:')

                dq = result.get('data_quality') or {}
                if dq:
                    st.markdown('#### Data quality')
                    st.caption('Reported rather than silently zeroed — anything skipped or malformed shows up here.')
                    dq_labels = {
                        'rows_received': ('Rows received', ':material/input:'),
                        'unique_profiles_found': ('Unique profiles', ':material/person_search:'),
                        'ingested': ('Profiles scored', ':material/check_circle:'),
                        'rows_skipped_no_handle': ('Skipped — no handle', ':material/block:'),
                        'rows_skipped_no_identifier': ('Skipped — no identifier', ':material/block:'),
                        'tweets_missing_date': ('Missing a date', ':material/event_busy:'),
                        'profiles_with_malformed_profile_json': ('Malformed profile JSON', ':material/data_object:'),
                        'profiles_with_malformed_experiences_json': ('Malformed experiences JSON', ':material/data_object:'),
                        'profiles_with_malformed_educations_json': ('Malformed educations JSON', ':material/data_object:'),
                    }
                    present = [(k, v) for k, v in dq_labels.items() if k in dq]
                    dq_cols = st.columns(3, gap='medium')
                    for i, (key, (label, icon)) in enumerate(present):
                        dq_cols[i % 3].metric(label, f'{dq[key]:,}', border=True, icon=icon)

                if recompute.get('ok'):
                    diag = recompute.get('diagnostics', {})
                    if diag.get('ok'):
                        st.markdown('#### Model refit')
                        r1, r2, r3 = st.columns(3, gap='medium')
                        r1.metric('Clusters (k)', diag['k'], border=True, icon=':material/group_work:',
                                  help=diag.get('k_selection', ''))
                        r2.metric('Silhouette', diag['silhouette_score'], border=True, icon=':material/straighten:',
                                  help='Cluster separation quality, -1 to 1. Above ~0.3 is reasonable.')
                        r3.metric('Anomalies flagged', diag['flagged_anomalies'], border=True, icon=':material/warning:')
                    else:
                        st.info(diag.get('reason', 'Not enough profiles yet for anomaly/cluster models.'),
                                icon=':material/info:')

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
        st.info('No profiles ingested yet for this platform. Use the Bulk Ingestion tab.', icon=':material/upload_file:')
    else:
        df = df.copy()
        df['tier'] = df['risk_score'].apply(tier_for)
        name_col = 'username' if platform == 'twitter' else 'full_name'

        st.markdown('#### Exposure distribution')
        c1, c2 = st.columns(2, gap='medium')
        with c1:
            fig = px.histogram(df, x='risk_score', nbins=20, title='Score distribution',
                               color_discrete_sequence=[ACCENT])
            fig.add_vline(x=df['risk_score'].mean(), line_dash='dash', line_color='#94a3b8',
                          annotation_text='mean', annotation_position='top')
            st.plotly_chart(styled(fig), width='stretch')
        with c2:
            tier_counts = df['tier'].value_counts().reindex(TIER_ORDER).dropna()
            fig = px.pie(names=tier_counts.index, values=tier_counts.values, title='Tier breakdown',
                         color=tier_counts.index, color_discrete_map=TIER_COLORS, hole=0.55)
            fig.update_traces(textposition='outside', textinfo='label+percent')
            st.plotly_chart(styled(fig), width='stretch')

        st.markdown('#### Highest-exposure profiles')
        top15 = df.sort_values('risk_score', ascending=False).head(15)
        fig = px.bar(top15.sort_values('risk_score'), x='risk_score', y=name_col, orientation='h',
                     title='Top 15 by exposure risk', color='tier', color_discrete_map=TIER_COLORS,
                     text='risk_score')
        fig.update_traces(textposition='outside', cliponaxis=False)
        fig.update_xaxes(range=[0, 105])
        st.plotly_chart(styled(fig, height=460, legend=True), width='stretch')

        st.markdown('#### Reach and behavior')
        c3, c4 = st.columns(2, gap='medium')
        if platform == 'twitter':
            with c3:
                fig = px.bar(top15.sort_values('avg_engagement'), x='avg_engagement', y='username',
                             orientation='h', title='Avg engagement (top 15 by exposure)',
                             color_discrete_sequence=[ACCENT])
                st.plotly_chart(styled(fig, height=420), width='stretch')
            with c4:
                fig = px.scatter(df, x='avg_engagement', y='risk_score', hover_name='username',
                                 color='tier', color_discrete_map=TIER_COLORS,
                                 size='total_views', size_max=26, opacity=0.75,
                                 title='Exposure vs engagement (bubble = total views)')
                st.plotly_chart(styled(fig, height=420, legend=True), width='stretch')

            cat_rows = [c for _, row in df.iterrows() for c in (row.get('categories') or [])]
            if cat_rows:
                cat_df = pd.Series(cat_rows).value_counts().reset_index()
                cat_df.columns = ['category', 'count']
                fig = px.bar(cat_df.sort_values('count'), x='count', y='category', orientation='h',
                             title='Tweet category distribution', color_discrete_sequence=['#34d399'])
                st.plotly_chart(styled(fig, height=280), width='stretch')
        else:
            df_reach = df.assign(reach=df['connections_count'].fillna(0) + df['follower_count'].fillna(0))
            with c3:
                t15 = top15.assign(reach=top15['connections_count'].fillna(0) + top15['follower_count'].fillna(0))
                fig = px.bar(t15.sort_values('reach'), x='reach', y='full_name', orientation='h',
                             title='Professional reach (top 15 by exposure)', color_discrete_sequence=[ACCENT])
                st.plotly_chart(styled(fig, height=420), width='stretch')
            with c4:
                fig = px.scatter(df_reach, x='reach', y='risk_score', hover_name='full_name',
                                 color='tier', color_discrete_map=TIER_COLORS, opacity=0.7,
                                 title='Exposure vs professional reach')
                st.plotly_chart(styled(fig, height=420, legend=True), width='stretch')

            st.markdown('#### Disclosed PII')
            c5, c6 = st.columns(2, gap='medium')
            with c5:
                loc = df['location_city'].replace('', pd.NA).fillna(df['location_state']).fillna(df['location_country'])
                loc_counts = loc.dropna().value_counts().head(15).reset_index()
                loc_counts.columns = ['location', 'count']
                if not loc_counts.empty:
                    fig = px.bar(loc_counts.sort_values('count'), x='count', y='location', orientation='h',
                                 title='Most-disclosed locations', color_discrete_sequence=['#34d399'])
                    st.plotly_chart(styled(fig, height=420), width='stretch')
            with c6:
                status = pd.Series({
                    'Open to Work': int(df['open_to_work'].sum()),
                    'Hiring': int(df['is_hiring'].sum()),
                    'Neither flag set': int(((~df['open_to_work']) & (~df['is_hiring'])).sum()),
                })
                status = status[status > 0]
                if not status.empty:
                    fig = px.pie(names=status.index, values=status.values, hole=0.55,
                                 title='Job-status exposure (targeting risk)',
                                 color_discrete_sequence=['#fbbf24', '#f87171', '#4ade80'])
                    fig.update_traces(textposition='outside', textinfo='label+percent')
                    st.plotly_chart(styled(fig, height=420), width='stretch')

        st.markdown('#### Unsupervised layer')
        if 'anomaly_score' in df.columns and df['anomaly_score'].notna().any():
            c7, c8 = st.columns(2, gap='medium')
            with c7:
                fig = px.histogram(df.dropna(subset=['anomaly_score']), x='anomaly_score', nbins=20,
                                   title='Anomaly score distribution (Isolation Forest)',
                                   color_discrete_sequence=['#a78bfa'])
                fig.add_vline(x=70, line_dash='dash', line_color='#f87171',
                              annotation_text='flag threshold', annotation_position='top')
                st.plotly_chart(styled(fig), width='stretch')
            with c8:
                if df['cluster_id'].notna().any():
                    cl = df.dropna(subset=['cluster_id'])['cluster_id'].astype(int).value_counts().sort_index()
                    fig = px.bar(x=[f'Cluster {i}' for i in cl.index], y=cl.values,
                                 title='Persona cluster sizes (KMeans)', color_discrete_sequence=[ACCENT],
                                 text=cl.values)
                    fig.update_traces(textposition='outside', cliponaxis=False)
                    st.plotly_chart(styled(fig), width='stretch')
            st.caption(
                'Anomaly scores are relative to the profiles currently in the database — a rank within '
                'this population, not a calibrated probability.'
            )
        else:
            st.info(
                'Recompute the anomaly/cluster models (Bulk Ingestion or Model Insights tab) to populate these charts.',
                icon=':material/model_training:',
            )

    with st.expander('Export static PNG charts (matplotlib — for slides and reports)'):
        st.caption('Server-rendered PNGs of the same data, for pasting into documents.')
        if st.button('Generate PNG charts', icon=':material/image:'):
            with st.spinner('Rendering…'):
                try:
                    st.session_state.png_charts = api_get('/api/analytics/', params={'platform': platform}, timeout=TIMEOUT_SCAN)
                except Exception as e:
                    st.session_state.png_charts = {'ok': False, 'error': str(e)}
        png = st.session_state.get('png_charts')
        if png:
            if not png.get('ok'):
                st.error(png.get('error', 'Failed.'), icon=':material/error:')
            else:
                import base64
                charts = png.get('charts') or {}
                keys = list(charts)
                for i in range(0, len(keys), 2):
                    for col, key in zip(st.columns(2, gap='medium'), keys[i:i + 2]):
                        with col:
                            st.markdown(f'**{key.replace("_", " ").title()}**')
                            st.image(base64.b64decode(charts[key]), width='stretch')


# ── 4. Model Insights ────────────────────────────────────────────────────────
with tab_insights:
    st.subheader('Model Insights')
    st.caption('Diagnostics for the unsupervised layer (Isolation Forest + KMeans), and how the exposure risk score is computed.')

    with st.container(border=True):
        rc1, rc2 = st.columns([3, 2], vertical_alignment='center')
        with rc1:
            st.markdown('**Refit the unsupervised models**')
            st.caption(
                'Cluster IDs are matched back to the previous run so a persona keeps its identity. '
                'Force a fresh baseline only after a scoring-methodology change.'
            )
        force = rc2.toggle('Force fresh baseline', help='Discards previous centroid mapping — cluster IDs may be reassigned.')
        if rc2.button('Recompute models', type='primary', width='stretch', icon=':material/model_training:'):
            with st.spinner('Fitting Isolation Forest + KMeans…'):
                try:
                    r = requests.post(
                        api_url('/api/recompute-models/'),
                        params={'platform': platform, 'force_retrain': force},
                        timeout=TIMEOUT_SCAN,
                    )
                    st.session_state.model_diag = r.json()
                except Exception as e:
                    st.session_state.model_diag = {'ok': False, 'error': str(e)}
            clear_summary_cache()

    diag = (st.session_state.get('model_diag') or {}).get('diagnostics')
    if diag:
        if diag.get('ok'):
            d1, d2, d3, d4 = st.columns(4, gap='medium')
            d1.metric('Profiles used', f"{diag['n_profiles']:,}", border=True, icon=':material/database:')
            d2.metric('Clusters (k)', diag['k'], border=True, icon=':material/group_work:',
                      help=diag.get('k_selection', ''))
            d3.metric('Silhouette', diag['silhouette_score'], border=True, icon=':material/straighten:',
                      delta='well separated' if (diag['silhouette_score'] or 0) >= 0.5 else 'moderate',
                      delta_color='off')
            d4.metric('Anomalies flagged', diag['flagged_anomalies'], border=True, icon=':material/warning:')

            st.caption(f"**k selection:** {diag.get('k_selection', 'n/a')}")
            if diag.get('cluster_ids_stable_across_refits'):
                st.success('Cluster IDs were matched to the previous run — persona identity preserved.',
                           icon=':material/link:')
            else:
                st.warning('Fresh baseline: cluster IDs are newly assigned and may not match earlier runs.',
                           icon=':material/link_off:')
        else:
            st.warning(diag.get('reason'), icon=':material/info:')

    df = summary_df
    if not df.empty and 'anomaly_score' in df.columns and df['anomaly_score'].notna().any():
        name_col = 'username' if platform == 'twitter' else 'full_name'

        st.markdown('#### Flagged anomalies')
        st.caption('Isolation Forest score at or above 70 — statistical outliers within this population.')
        flagged_df = df[df['anomaly_score'] >= 70].sort_values('anomaly_score', ascending=False)
        if flagged_df.empty:
            st.caption('No profiles cross the threshold in the current population.')
        else:
            st.dataframe(
                flagged_df[[name_col, 'risk_score', 'anomaly_score', 'cluster_id']],
                width='stretch', hide_index=True,
                column_config={
                    name_col: st.column_config.TextColumn('Profile', width='medium'),
                    'risk_score': st.column_config.ProgressColumn('Exposure risk', min_value=0, max_value=100, format='%d'),
                    'anomaly_score': st.column_config.ProgressColumn('Anomaly', min_value=0, max_value=100, format='%.1f'),
                    'cluster_id': st.column_config.NumberColumn('Cluster', width='small'),
                },
            )

        st.markdown('#### Cluster composition')
        cl_summary = df.dropna(subset=['cluster_id']).groupby('cluster_id').agg(
            profiles=('risk_score', 'count'),
            avg_risk_score=('risk_score', 'mean'),
            avg_anomaly=('anomaly_score', 'mean'),
        ).round(1).reset_index()
        cl_summary['cluster_id'] = cl_summary['cluster_id'].astype(int)
        st.dataframe(
            cl_summary, width='stretch', hide_index=True,
            column_config={
                'cluster_id': st.column_config.NumberColumn('Cluster', width='small'),
                'profiles': st.column_config.NumberColumn('Profiles'),
                'avg_risk_score': st.column_config.ProgressColumn('Avg exposure risk', min_value=0, max_value=100, format='%.1f'),
                'avg_anomaly': st.column_config.NumberColumn('Avg anomaly', format='%.1f'),
            },
        )

    st.divider()
    st.markdown('#### How the exposure risk score is computed')
    st.caption('"Exposure risk" measures how much can be inferred from public data — not a probability of malicious intent or victimhood.')
    if platform == 'twitter':
        st.markdown(
            """
The score is a **deterministic, weighted sum** across four buckets (no black box, no training data):

| Bucket | Max points | Driven by |
| --- | --- | --- |
| PII exposure | 40 | Email/LinkedIn/employer disclosed in profile metadata + locations mentioned in tweets (spaCy NER) |
| Behavioral predictability | 25 | Entropy of posting-hour histogram, **confidence-weighted by tweet count** so a single tweet can't fake a "routine" |
| Content sensitivity | 20 | Two signals, **unioned not summed** (a tweet caught by both counts once): a keyword dictionary, plus local sentence-embedding similarity to hand-written example disclosures. The second catches what the first structurally can't — "I've moved to San Francisco", "Woke up at 4:30am, drove to JFK" — and every hit carries its similarity score as evidence so you can judge it |
| Exposure reach | 15 | Follower count (log-scaled) |

Separately, an **Isolation Forest** flags statistically anomalous accounts (unusual followers/following/engagement ratios) — an authenticity signal, not folded into the exposure risk score itself. **KMeans** groups accounts into behavioral personas, with cluster IDs matched back to the previous run's so a persona's identity survives a refit.

*Measured on this dataset (140 profiles / 1000 tweets): semantic similarity raised content-sensitivity detections from 5 to 17, affecting 8 profiles. Tuned for recall over precision — roughly 2 in 3 hits are clearly meaningful, which is why each one shows its similarity score rather than being silently folded in.*
            """
        )
    else:
        st.markdown(
            """
The score is a **deterministic, weighted sum** across four buckets (no black box, no training data):

| Bucket | Max points | Driven by |
| --- | --- | --- |
| PII exposure | 40 | Explicit city/state/country disclosure (city worth more than country — it's exact, not inferred), employer, education, NER entities in headline/about |
| Career-timeline predictability | 25 | Number of distinct employers in `experiences_json`, **confidence-weighted by how many career fields are actually populated** (employer/education/skills) so a sparse export isn't scored with the same certainty as a fully detailed profile |
| Content/status sensitivity | 20 | `open_to_work` / `is_hiring` flags — direct recruitment-scam/BEC targeting signals |
| Exposure reach | 15 | Connections + followers (log-scaled) |

Same Isolation Forest / KMeans layer as Twitter, fit on LinkedIn-specific features (connection ratios, employer count, premium/verified/influencer flags).
            """
        )


# ── 5. Database ──────────────────────────────────────────────────────────────
with tab_db:
    st.subheader(f'Stored Profiles — {"Twitter/X" if platform == "twitter" else "LinkedIn"}')

    st.caption('Browse, inspect, and export everything scored for this platform.')

    col_load, col_clear, _sp = st.columns([1, 1, 2], vertical_alignment='center')
    if col_load.button('Load profiles', width='stretch', type='primary', icon=':material/download:'):
        with st.spinner('Fetching…'):
            try:
                st.session_state.db = api_get('/api/check-db/', params={'platform': platform}, timeout=TIMEOUT_SCAN)
            except Exception as e:
                st.session_state.db = {'error': str(e)}

    with col_clear.popover('Wipe data', width='stretch', icon=':material/delete_forever:'):
        st.warning(
            f'Permanently deletes every stored {platform} profile and resets the persisted '
            'scaler/cluster state for this platform.',
            icon=':material/warning:',
        )
        if st.button('Yes, delete everything', type='primary'):
            try:
                r = requests.post(api_url('/api/clear-db/'), params={'platform': platform}, timeout=TIMEOUT_QUICK)
                res = r.json()
                st.success(res.get('message', 'Cleared.'), icon=':material/check_circle:')
                st.session_state.pop('db', None)
                clear_summary_cache()
            except Exception as e:
                st.error(f'Clear failed: {e}', icon=':material/error:')

    db = st.session_state.get('db')
    if db:
        if db.get('error'):
            st.error(db['error'], icon=':material/error:')
        else:
            profiles = db.get('profiles') or []
            st.caption(f"{db.get('total_db_records', 0):,} records total — showing the {len(profiles)} most recent")

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
                st.dataframe(
                    df, width='stretch', hide_index=True,
                    column_config={
                        'username': st.column_config.TextColumn('Profile', width='medium', pinned=True),
                        'risk_score': st.column_config.ProgressColumn('Exposure risk', min_value=0, max_value=100, format='%d'),
                        'risk_label': st.column_config.TextColumn('Assessment', width='medium'),
                        'anomaly_score': st.column_config.NumberColumn('Anomaly', format='%.1f'),
                        'cluster_id': st.column_config.NumberColumn('Cluster', width='small'),
                        'scanned_at': st.column_config.TextColumn('Scored at', width='medium'),
                    },
                )
                st.download_button(
                    'Download as CSV',
                    df.to_csv(index=False).encode('utf-8'),
                    f'osint_{platform}_profiles.csv',
                    'text/csv',
                    icon=':material/file_download:',
                )

                st.subheader('Inspect a profile', divider='gray')
                pick = st.selectbox('Profile', [p['username'] for p in profiles], label_visibility='collapsed')
                with st.spinner('Loading risk breakdown…'):
                    try:
                        detail = api_get(f'/api/profile-detail/{platform}/{pick}/')
                    except Exception as e:
                        detail = {'ok': False, 'error': str(e)}
                if detail.get('ok'):
                    prof = detail['profile']
                    factors = prof.get('risk_factors') or []
                    if factors:
                        st.markdown('**Risk factors**')
                        fdf = pd.DataFrame(factors)
                        st.dataframe(
                            fdf, width='stretch', hide_index=True,
                            column_config={
                                'category': st.column_config.TextColumn('Bucket', width='small'),
                                'label': st.column_config.TextColumn('Signal', width='medium'),
                                'points': st.column_config.ProgressColumn(
                                    'Points', min_value=0, max_value=float(fdf['max_points'].max()), format='%.1f',
                                ),
                                'max_points': st.column_config.NumberColumn('Max', width='small'),
                                'evidence': st.column_config.TextColumn('Evidence', width='large'),
                            },
                        )
                    rows = prof.get('inference_rows') or []
                    if rows:
                        st.markdown('**Extracted signals**')
                        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            else:
                st.info('No profiles stored yet.', icon=':material/inbox:')
    else:
        st.info('Click **Load profiles** to fetch stored records.', icon=':material/touch_app:')
