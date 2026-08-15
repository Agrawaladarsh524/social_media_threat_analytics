"""Request/response models for the API."""

from typing import Any

from pydantic import BaseModel, Field


class CollectRequest(BaseModel):
    twitter: str = Field(..., min_length=1, description='Twitter/X handle, with or without @')


class InsightsRequest(BaseModel):
    datasets: dict[str, Any] = Field(default_factory=dict)
    username: str | None = None
