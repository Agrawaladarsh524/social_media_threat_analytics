"""Application settings, loaded from environment variables / .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    DEBUG: bool = True

    # Origins allowed to call the API from a browser. Streamlit talks to the API
    # server-side, so this only matters if you add a browser client later.
    CORS_ALLOWED_ORIGINS: list[str] = ['http://localhost:8501', 'http://127.0.0.1:8501']

    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'osint.db'}"

    # --- Apify (live OSINT collection) ---
    APIFY_API_TOKEN: str = ''
    APIFY_ACTOR_TWITTER: str = 'dy7gIgPRMhrOrfW0f'

    # --- OpenAI (AI risk analysis) ---
    OPENAI_API_KEY: str = ''
    OPENAI_MODEL: str = 'gpt-4o-mini'
    OPENAI_REQUEST_TIMEOUT: int = 30  # seconds per request
    AI_CACHE_TTL_HOURS: int = 24      # hours before a cached AI bundle is regenerated


settings = Settings()
