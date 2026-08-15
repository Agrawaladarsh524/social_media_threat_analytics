"""Seaborn / matplotlib charts rendered to base64 PNGs.

Every chart here is fed by the local risk_engine/nlp_features/ml_models
pipeline (backend/app/services/{risk_engine,nlp_features,ml_models}.py) —
no chart depends on OpenAI. `build_charts(profiles, platform)` branches on
platform because Twitter and LinkedIn profiles carry different native
metrics (tweet engagement vs. professional reach/status), but risk_score,
anomaly_score, and cluster_id are shared fields on both models, so those
three charts use one shared code path.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use('Agg')  # Required for headless environments

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def _fig_to_base64() -> str:
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')


def _risk_shared_charts(profiles: list, charts: dict) -> list:
    """Risk distribution / tier / top-15 charts — identical logic regardless
    of platform, since risk_score is a shared 0-100 field on both models."""
    scored = [p for p in profiles if p.risk_score is not None]
    if not scored:
        return []

    scored_sorted = sorted(scored, key=lambda p: p.risk_score, reverse=True)
    top15 = scored_sorted[:15]
    all_scores = [p.risk_score for p in scored]

    plt.figure(figsize=(9, 4))
    sns.histplot(all_scores, bins=min(len(all_scores), 15), kde=len(all_scores) > 3, color='#38bdf8')
    plt.title('Exposure Risk Score Distribution Across All Profiles')
    plt.xlabel('Exposure Risk Score (0-100)')
    plt.ylabel('Number of Profiles')
    plt.tight_layout()
    charts['risk_distribution'] = _fig_to_base64()

    tier_counts = [
        sum(1 for s in all_scores if s < 40),
        sum(1 for s in all_scores if 40 <= s < 70),
        sum(1 for s in all_scores if 70 <= s < 90),
        sum(1 for s in all_scores if s >= 90),
    ]
    tier_labels = ['Low (<40)', 'Medium (40-69)', 'High (70-89)', 'Critical (>=90)']
    tier_colors = ['#4ade80', '#fbbf24', '#f87171', '#991b1b']
    nz = [(s, l, c) for s, l, c in zip(tier_counts, tier_labels, tier_colors) if s > 0]
    plt.figure(figsize=(6, 6))
    if nz:
        sz, lb, cl = zip(*nz)
        plt.pie(sz, labels=lb, colors=cl, autopct='%1.1f%%', startangle=140)
    plt.title('Risk Tier Breakdown')
    plt.tight_layout()
    charts['risk_tiers'] = _fig_to_base64()

    return top15


def _anomaly_cluster_charts(profiles: list, charts: dict) -> None:
    """Isolation Forest anomaly-score histogram + KMeans cluster sizes —
    shared across platforms since both write anomaly_score/cluster_id."""
    scored = [p for p in profiles if p.anomaly_score is not None]
    if scored:
        plt.figure(figsize=(9, 4))
        sns.histplot([p.anomaly_score for p in scored], bins=15, color='#a78bfa')
        plt.axvline(70, color='#991b1b', linestyle='--', label='Flag threshold (70)')
        plt.title('Anomaly Score Distribution (Isolation Forest)')
        plt.xlabel('Anomaly Score (0-100, higher = more outlier-like)')
        plt.ylabel('Number of Profiles')
        plt.legend()
        plt.tight_layout()
        charts['anomaly_distribution'] = _fig_to_base64()

    clustered = [p for p in profiles if p.cluster_id is not None]
    if clustered:
        counts: dict[int, int] = {}
        for p in clustered:
            counts[p.cluster_id] = counts.get(p.cluster_id, 0) + 1
        labels = [f'Cluster {k}' for k in sorted(counts)]
        vals = [counts[k] for k in sorted(counts)]
        plt.figure(figsize=(7, 4))
        sns.barplot(x=labels, y=vals, hue=labels, palette='viridis', legend=False)
        plt.title('Persona Cluster Sizes (KMeans)')
        plt.ylabel('Number of Profiles')
        plt.tight_layout()
        charts['cluster_sizes'] = _fig_to_base64()


def _build_twitter_charts(profiles: list) -> dict:
    charts: dict[str, str] = {}
    top15 = _risk_shared_charts(profiles, charts)
    if not top15:
        return {'ok': False, 'error': 'No exposure risk scores calculated yet. Run local ingestion first.'}

    usernames    = [p.username for p in top15]
    risk_scores  = [p.risk_score for p in top15]
    avg_engs     = [p.avg_engagement for p in top15]
    total_views  = [p.total_views for p in top15]
    total_tweets = [p.total_tweets for p in top15]
    scored = [p for p in profiles if p.risk_score is not None]

    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    bars = ax.barh(usernames[::-1], risk_scores[::-1], color='#f87171')
    ax.set_xlabel('Exposure Risk Score (0-100)')
    ax.set_title('Top Users by Exposure Risk Score')
    ax.set_xlim(0, 100)
    for bar, score in zip(bars, risk_scores[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, str(score), va='center', fontsize=8)
    plt.tight_layout()
    charts['top_risk_users'] = _fig_to_base64()

    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    ax.barh(usernames[::-1], avg_engs[::-1], color='#38bdf8')
    ax.set_xlabel('Avg Engagement (likes + retweets + replies) / tweet')
    ax.set_title('Average Engagement per User (Risk-Ordered)')
    plt.tight_layout()
    charts['engagement'] = _fig_to_base64()

    cat_counter: dict[str, int] = {}
    for p in profiles:
        for cat in (p.categories or []):
            cat_counter[cat] = cat_counter.get(cat, 0) + 1
    if cat_counter:
        cats_sorted = sorted(cat_counter.items(), key=lambda x: x[1], reverse=True)[:20]
        cat_names, cat_vals = zip(*cats_sorted)
        plt.figure(figsize=(10, max(4, len(cat_names) * 0.4)))
        sns.barplot(x=list(cat_vals), y=list(cat_names), hue=list(cat_names), palette='Blues_r', legend=False)
        plt.xlabel('Number of Tweets')
        plt.title('Tweet Category Distribution (All Users)')
        plt.tight_layout()
        charts['categories'] = _fig_to_base64()

    plt.figure(figsize=(8, 5))
    sc_x = [p.avg_engagement for p in scored]
    sc_y = [p.risk_score for p in scored]
    plt.scatter(sc_x, sc_y, color='#a78bfa', alpha=0.7, edgecolors='white', linewidths=0.4, s=60)
    for x, y, label in zip(sc_x, sc_y, [p.username for p in scored]):
        plt.annotate(f'@{label}', (x, y), textcoords='offset points', xytext=(5, 3), fontsize=7, color='#64748b')
    plt.xlabel('Avg Engagement per Tweet')
    plt.ylabel('Exposure Risk Score')
    plt.title('Exposure Risk Score vs Avg Engagement')
    plt.tight_layout()
    charts['risk_vs_engagement'] = _fig_to_base64()

    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    ax.barh(usernames[::-1], total_views[::-1], color='#34d399')
    ax.set_xlabel('Total Views')
    ax.set_title('Total Tweet Views per User (Risk-Ordered)')
    plt.tight_layout()
    charts['total_views'] = _fig_to_base64()

    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    ax.barh(usernames[::-1], total_tweets[::-1], color='#fbbf24')
    ax.set_xlabel('Number of Tweets in Dataset')
    ax.set_title('Tweet Volume per User (Risk-Ordered)')
    plt.tight_layout()
    charts['tweet_volume'] = _fig_to_base64()

    _anomaly_cluster_charts(profiles, charts)
    return {'ok': True, 'total_users': len(profiles), 'charts': charts}


def _build_linkedin_charts(profiles: list) -> dict:
    charts: dict[str, str] = {}
    top15 = _risk_shared_charts(profiles, charts)
    if not top15:
        return {'ok': False, 'error': 'No exposure risk scores calculated yet. Run local ingestion first.'}

    names = [p.full_name or p.public_identifier for p in top15]
    risk_scores = [p.risk_score for p in top15]
    reach = [(p.connections_count or 0) + (p.follower_count or 0) for p in top15]
    scored = [p for p in profiles if p.risk_score is not None]

    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    bars = ax.barh(names[::-1], risk_scores[::-1], color='#f87171')
    ax.set_xlabel('Exposure Risk Score (0-100)')
    ax.set_title('Top LinkedIn Profiles by Exposure Risk Score')
    ax.set_xlim(0, 100)
    for bar, score in zip(bars, risk_scores[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, str(score), va='center', fontsize=8)
    plt.tight_layout()
    charts['top_risk_users'] = _fig_to_base64()

    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    ax.barh(names[::-1], reach[::-1], color='#38bdf8')
    ax.set_xlabel('Connections + Followers')
    ax.set_title('Professional Reach per Profile (Risk-Ordered)')
    plt.tight_layout()
    charts['reach'] = _fig_to_base64()

    loc_counter: dict[str, int] = {}
    for p in profiles:
        loc = p.location_city or p.location_state or p.location_country
        if loc:
            loc_counter[loc] = loc_counter.get(loc, 0) + 1
    if loc_counter:
        loc_sorted = sorted(loc_counter.items(), key=lambda x: x[1], reverse=True)[:15]
        loc_names, loc_vals = zip(*loc_sorted)
        plt.figure(figsize=(10, max(4, len(loc_names) * 0.4)))
        sns.barplot(x=list(loc_vals), y=list(loc_names), hue=list(loc_names), palette='Greens_r', legend=False)
        plt.xlabel('Number of Profiles')
        plt.title('Most-Disclosed Locations (Direct PII Exposure)')
        plt.tight_layout()
        charts['top_locations'] = _fig_to_base64()

    status_counts = {
        'Open to Work': sum(1 for p in profiles if p.open_to_work),
        'Hiring': sum(1 for p in profiles if p.is_hiring),
        'Neither flag set': sum(1 for p in profiles if not p.open_to_work and not p.is_hiring),
    }
    nz = {k: v for k, v in status_counts.items() if v > 0}
    if nz:
        plt.figure(figsize=(6, 6))
        plt.pie(nz.values(), labels=nz.keys(), autopct='%1.1f%%', startangle=140,
                colors=['#fbbf24', '#f87171', '#4ade80'][:len(nz)])
        plt.title('Job-Status Exposure (Social-Engineering Targeting Risk)')
        plt.tight_layout()
        charts['job_status'] = _fig_to_base64()

    plt.figure(figsize=(8, 5))
    sc_x = [(p.connections_count or 0) + (p.follower_count or 0) for p in scored]
    sc_y = [p.risk_score for p in scored]
    plt.scatter(sc_x, sc_y, color='#a78bfa', alpha=0.7, edgecolors='white', linewidths=0.4, s=60)
    plt.xlabel('Connections + Followers')
    plt.ylabel('Exposure Risk Score')
    plt.title('Exposure Risk Score vs Professional Reach')
    plt.tight_layout()
    charts['risk_vs_reach'] = _fig_to_base64()

    _anomaly_cluster_charts(profiles, charts)
    return {'ok': True, 'total_users': len(profiles), 'charts': charts}


def build_charts(profiles: list, platform: str = 'twitter') -> dict:
    """Returns {'ok': False, 'error': ...} or {'ok': True, 'total_users', 'charts'}."""
    if not profiles:
        return {'ok': False, 'error': 'No profiles found in database.'}
    if platform == 'linkedin':
        return _build_linkedin_charts(profiles)
    return _build_twitter_charts(profiles)
