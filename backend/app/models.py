"""Database models."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TwitterProfile(Base):
    """
    Stores Twitter OSINT profiles.
    Schema = real data only. No AI-generated prose.
    """

    __tablename__ = 'twitter_profiles'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    # --- Real tweet stats computed from raw data ---
    total_tweets: Mapped[int] = mapped_column(Integer, default=0)
    date_first_tweet: Mapped[str] = mapped_column(String(50), default='')
    date_last_tweet: Mapped[str] = mapped_column(String(50), default='')
    categories: Mapped[list[Any]] = mapped_column(JSON, default=list)
    total_likes: Mapped[int] = mapped_column(Integer, default=0)
    total_retweets: Mapped[int] = mapped_column(Integer, default=0)
    total_replies: Mapped[int] = mapped_column(Integer, default=0)
    total_quotes: Mapped[int] = mapped_column(Integer, default=0)
    total_views: Mapped[int] = mapped_column(Integer, default=0)
    avg_engagement: Mapped[float] = mapped_column(Float, default=0.0)

    # --- Risk decision (score + what signals drove it) ---
    risk_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    risk_label: Mapped[str] = mapped_column(String(255), default='')
    inference_rows: Mapped[list[Any]] = mapped_column(JSON, default=list)
    pattern_of_life: Mapped[list[Any]] = mapped_column(JSON, default=list)

    # --- Local intelligence pipeline outputs (risk_engine / nlp_features / ml_models) ---
    has_email: Mapped[bool] = mapped_column(default=False)
    has_linkedin: Mapped[bool] = mapped_column(default=False)
    has_org: Mapped[bool] = mapped_column(default=False)
    organization: Mapped[str] = mapped_column(String(255), default='')
    followers_count: Mapped[int] = mapped_column(Integer, default=0)
    following_count: Mapped[int] = mapped_column(Integer, default=0)
    posting_entropy: Mapped[float] = mapped_column(Float, default=0.0)
    risk_factors: Mapped[list[Any]] = mapped_column(JSON, default=list)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- Full AI bundle + raw tweet data ---
    ai_bundle: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    twitter_raw_data: Mapped[Any] = mapped_column(JSON, default=list)

    # --- Metadata ---
    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    bundle_generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f'<TwitterProfile @{self.username} (Risk: {self.risk_score})>'


class LinkedInProfile(Base):
    """
    Stores LinkedIn OSINT profiles, scored by the same local risk engine as
    TwitterProfile but through a LinkedIn-specific feature extractor.
    """

    __tablename__ = 'linkedin_profiles'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_identifier: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255), default='')
    headline: Mapped[str] = mapped_column(String(500), default='')
    about: Mapped[str] = mapped_column(String(2000), default='')

    # --- Disclosed identity / location (real data, no inference) ---
    location_city: Mapped[str] = mapped_column(String(255), default='')
    location_state: Mapped[str] = mapped_column(String(255), default='')
    location_country: Mapped[str] = mapped_column(String(255), default='')
    current_employer: Mapped[str] = mapped_column(String(255), default='')
    employer_count: Mapped[int] = mapped_column(Integer, default=0)
    education_disclosed: Mapped[bool] = mapped_column(default=False)
    skill_count: Mapped[int] = mapped_column(Integer, default=0)

    # --- Reach / status flags ---
    connections_count: Mapped[int] = mapped_column(Integer, default=0)
    follower_count: Mapped[int] = mapped_column(Integer, default=0)
    open_to_work: Mapped[bool] = mapped_column(default=False)
    is_hiring: Mapped[bool] = mapped_column(default=False)
    is_premium: Mapped[bool] = mapped_column(default=False)
    is_influencer: Mapped[bool] = mapped_column(default=False)
    is_verified: Mapped[bool] = mapped_column(default=False)

    # --- Source-provided score (never reinterpreted as our own) ---
    source_background_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- Local intelligence pipeline outputs (shared shape with TwitterProfile) ---
    risk_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    risk_label: Mapped[str] = mapped_column(String(255), default='')
    risk_factors: Mapped[list[Any]] = mapped_column(JSON, default=list)
    inference_rows: Mapped[list[Any]] = mapped_column(JSON, default=list)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- Raw source row ---
    linkedin_raw_data: Mapped[Any] = mapped_column(JSON, default=dict)

    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def __repr__(self) -> str:
        return f'<LinkedInProfile @{self.public_identifier} (Risk: {self.risk_score})>'


class RiskSnapshot(Base):
    """
    Append-only audit trail: one row per scoring event. Never overwritten —
    local_ingest.py inserts a new row every time a profile is (re-)scored,
    instead of only updating the profile's "latest" columns in place. This
    is what makes "how has this profile's exposure changed" answerable at
    all: a re-ingestion after a scoring-methodology change produces a real,
    honest second data point, not a synthetic backfill.
    """

    __tablename__ = 'risk_snapshots'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    platform: Mapped[str] = mapped_column(String(20), index=True)
    profile_identifier: Mapped[str] = mapped_column(String(255), index=True)  # username or public_identifier
    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    risk_score: Mapped[int] = mapped_column(Integer)
    risk_label: Mapped[str] = mapped_column(String(255), default='')
    risk_factors: Mapped[list[Any]] = mapped_column(JSON, default=list)

    def __repr__(self) -> str:
        return f'<RiskSnapshot {self.platform}:{self.profile_identifier} @{self.scanned_at} (Risk: {self.risk_score})>'
