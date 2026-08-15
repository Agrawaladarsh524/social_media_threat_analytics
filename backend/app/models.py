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

    # --- AI risk decision (score + what signals drove it) ---
    risk_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    risk_label: Mapped[str] = mapped_column(String(255), default='')
    inference_rows: Mapped[list[Any]] = mapped_column(JSON, default=list)
    pattern_of_life: Mapped[list[Any]] = mapped_column(JSON, default=list)

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
