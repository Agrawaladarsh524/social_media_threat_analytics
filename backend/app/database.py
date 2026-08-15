"""SQLAlchemy engine, session factory and declarative base."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

# check_same_thread is a SQLite-only knob; FastAPI serves requests on a threadpool.
connect_args = {'check_same_thread': False} if settings.DATABASE_URL.startswith('sqlite') else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models  # noqa: F401 — registers the tables on Base.metadata

    Base.metadata.create_all(bind=engine)
