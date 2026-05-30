"""
database.py
===========
SQLAlchemy engine/session setup for the Urdu OCR Suite.

Uses SQLite by default (a single self-contained file, ideal for Replit). Override
with DATABASE_URL if you ever move to Postgres etc.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ocr_suite.db")

# check_same_thread=False lets the background OCR task use the connection from a
# worker thread; SQLite is otherwise single-thread by default.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


def get_db():
    """FastAPI dependency yielding a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables. Imports models so they register on Base.metadata."""
    import models  # noqa: F401  (registers ORM classes)
    Base.metadata.create_all(bind=engine)
    _ensure_columns()


# Tiny additive "migration": create_all won't add columns to pre-existing tables,
# so add any new ones here. (For anything more complex, use Alembic.)
_EXPECTED_COLUMNS = {
    "pages": {
        "verified": "BOOLEAN NOT NULL DEFAULT 0",
        "tokens_json": "TEXT NOT NULL DEFAULT '[]'",
        "agent_logs": "TEXT NOT NULL DEFAULT '{}'"
    },
    "documents": {
        "model": "VARCHAR NOT NULL DEFAULT 'gemini-2.5-flash'",
        "selected_agents": "TEXT NOT NULL DEFAULT '[]'"
    },
}


def _ensure_columns() -> None:
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table, columns in _EXPECTED_COLUMNS.items():
            if table not in tables:
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for col, ddl in columns.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))
