from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(value, minimum)


_BUSY_TIMEOUT_MS = _int_env("SQLITE_BUSY_TIMEOUT_MS", 60_000, minimum=1)
_MAX_CONCURRENT_PER_DB = _int_env("SQLITE_MAX_CONCURRENT_PER_DB", 5, minimum=1)
_JOURNAL_MODE = os.getenv("SQLITE_JOURNAL_MODE", "WAL")
_SYNCHRONOUS = os.getenv("SQLITE_SYNCHRONOUS", "NORMAL")

_DB_SEMAPHORES: Dict[str, asyncio.Semaphore] = {}


def create_sqlite_engine(db_path: str, *, echo: bool = False) -> AsyncEngine:
    """
    Create an async SQLite engine tuned for concurrent access.
    - WAL mode for concurrent readers
    - Busy timeout to reduce "database is locked" errors
    - Bounded pool to avoid spawning too many connections per DB
    """
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=echo,
        connect_args={"check_same_thread": False, "timeout": _BUSY_TIMEOUT_MS / 1000},
        pool_pre_ping=True,
    )
    _configure_sqlite_pragmas(engine)
    return engine


def _configure_sqlite_pragmas(engine: AsyncEngine) -> None:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[unused-argument]  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA journal_mode={_JOURNAL_MODE};")
            cursor.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS};")
            cursor.execute(f"PRAGMA synchronous={_SYNCHRONOUS};")
        finally:
            cursor.close()


def _get_db_semaphore(key: str) -> asyncio.Semaphore:
    semaphore = _DB_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_PER_DB)
        _DB_SEMAPHORES[key] = semaphore
    return semaphore


@asynccontextmanager
async def db_semaphore(key: str):
    """
    Limit concurrent access to a single SQLite database per process.
    This queues callers instead of letting them race into busy timeouts.
    """
    semaphore = _get_db_semaphore(key)
    async with semaphore:
        yield
