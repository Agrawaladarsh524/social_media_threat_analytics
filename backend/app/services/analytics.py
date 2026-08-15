"""Seaborn / matplotlib charts rendered to base64 PNGs."""

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


def build_charts(profiles: list) -> dict:
    """Returns {'ok': False, 'error': ...} or {'ok': True, 'total_users', 'charts'}."""
    if not profiles:
        return {'ok': False, 'error': 'No profiles found in database.'}

    scored = [p for p in profiles if p.risk_score is not None]
    if not scored:
        return {'ok': False, 'error': 'No risk scores calculated yet.'}

    scored_sorted = sorted(scored, key=lambda p: p.risk_score, reverse=True)
    top15 = scored_sorted[:15]

    usernames    = [p.username for p in top15]
    risk_scores  = [p.risk_score for p in top15]
    avg_engs     = [p.avg_engagement for p in top15]
    total_views  = [p.total_views for p in top15]
    total_tweets = [p.total_tweets for p in top15]

    charts: dict[str, str] = {}

    # ── Chart 1: Risk score distribution histogram ──────────────────────────
    all_scores = [p.risk_score for p in scored]
    plt.figure(figsize=(9, 4))
    sns.histplot(all_scores, bins=min(len(all_scores), 15), kde=len(all_scores) > 3, color='#38bdf8')
    plt.title('Risk Score Distribution Across All Users')
    plt.xlabel('Risk Score (0-100)')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    charts['risk_distribution'] = _fig_to_base64()

    # ── Chart 2: Risk tier pie chart ────────────────────────────────────────
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

    # ── Chart 3: Top 15 users by risk score ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    bars = ax.barh(usernames[::-1], risk_scores[::-1], color='#f87171')
    ax.set_xlabel('Risk Score (0-100)')
    ax.set_title('Top Users by Risk Score')
    ax.set_xlim(0, 100)
    for bar, score in zip(bars, risk_scores[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(score), va='center', fontsize=8)
    plt.tight_layout()
    charts['top_risk_users'] = _fig_to_base64()

    # ── Chart 4: Avg engagement per user ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    ax.barh(usernames[::-1], avg_engs[::-1], color='#38bdf8')
    ax.set_xlabel('Avg Engagement (likes + retweets + replies) / tweet')
    ax.set_title('Average Engagement per User (Risk-Ordered)')
    plt.tight_layout()
    charts['engagement'] = _fig_to_base64()

    # ── Chart 5: Category distribution across all users ─────────────────────
    cat_counter: dict[str, int] = {}
    for p in profiles:
        for cat in (p.categories or []):
            cat_counter[cat] = cat_counter.get(cat, 0) + 1
    if cat_counter:
        cats_sorted = sorted(cat_counter.items(), key=lambda x: x[1], reverse=True)[:20]
        cat_names, cat_vals = zip(*cats_sorted)
        plt.figure(figsize=(10, max(4, len(cat_names) * 0.4)))
        sns.barplot(x=list(cat_vals), y=list(cat_names), hue=list(cat_names),
                    palette='Blues_r', legend=False)
        plt.xlabel('Number of Tweets')
        plt.title('Tweet Category Distribution (All Users)')
        plt.tight_layout()
        charts['categories'] = _fig_to_base64()

    # ── Chart 6: Scatter — risk score vs avg engagement ─────────────────────
    plt.figure(figsize=(8, 5))
    sc_x = [p.avg_engagement for p in scored]
    sc_y = [p.risk_score for p in scored]
    plt.scatter(sc_x, sc_y, color='#a78bfa', alpha=0.7, edgecolors='white', linewidths=0.4, s=60)
    for x, y, label in zip(sc_x, sc_y, [p.username for p in scored]):
        plt.annotate(f'@{label}', (x, y), textcoords='offset points',
                     xytext=(5, 3), fontsize=7, color='#64748b')
    plt.xlabel('Avg Engagement per Tweet')
    plt.ylabel('Risk Score')
    plt.title('Risk Score vs Avg Engagement')
    plt.tight_layout()
    charts['risk_vs_engagement'] = _fig_to_base64()

    # ── Chart 7: Total views per user ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    ax.barh(usernames[::-1], total_views[::-1], color='#34d399')
    ax.set_xlabel('Total Views')
    ax.set_title('Total Tweet Views per User (Risk-Ordered)')
    plt.tight_layout()
    charts['total_views'] = _fig_to_base64()

    # ── Chart 8: Tweet volume per user ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(4, len(top15) * 0.45)))
    ax.barh(usernames[::-1], total_tweets[::-1], color='#fbbf24')
    ax.set_xlabel('Number of Tweets in Dataset')
    ax.set_title('Tweet Volume per User (Risk-Ordered)')
    plt.tight_layout()
    charts['tweet_volume'] = _fig_to_base64()

    return {'ok': True, 'total_users': len(profiles), 'charts': charts}
