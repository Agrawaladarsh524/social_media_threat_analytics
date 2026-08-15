"""Test fixtures — an isolated temp-file SQLite DB *and* an isolated model
directory per test, never the real backend/osint.db or backend/models/ used
by the running dev server."""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.services import ml_models


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # ml_models persists scaler/centroid artifacts to a module-level MODEL_DIR —
    # without this, route tests that call /api/recompute-models/ would write
    # into the real backend/models/ and leak state between test runs.
    monkeypatch.setattr(ml_models, 'MODEL_DIR', tmp_path)

    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    engine = create_engine(f'sqlite:///{path}', connect_args={'check_same_thread': False})
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    from app import models  # noqa: F401 — register tables on Base.metadata
    Base.metadata.create_all(bind=engine)

    def _override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    engine.dispose()
    os.remove(path)
