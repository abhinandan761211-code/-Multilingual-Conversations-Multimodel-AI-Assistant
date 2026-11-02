"""
database.py
===========
Engine, session factory and declarative base.

SQLite is the default (one file under ``data/assistant.db``) because it needs no
server and can be inspected with any SQLite browser. The connection string is
read from ``config.DATABASE_URL``, so pointing the app at Postgres is an
environment-variable change rather than a code change.

SQLite specifics handled here
-----------------------------
* ``check_same_thread=False`` — FastAPI may run a request on a different thread
  than the one that created the connection; SQLAlchemy still serialises access
  through its session pool.
* Foreign keys are OFF by default in SQLite. The ``PRAGMA`` listener below turns
  them on so ``ON DELETE CASCADE`` on messages actually works.
* WAL mode lets the Streamlit front-end read while the API writes.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core import config


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


def _build_engine(url: str):
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        url,
        echo=config.SQL_ECHO,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


engine = _build_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    # Import models so they are registered on Base.metadata before create_all.
    from backend.db import models  # noqa: F401

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """For services and scripts: commit on success, roll back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
