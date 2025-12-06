import os
import random

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

from db_engine import create_sqlite_engine, db_semaphore

BaseAuth = declarative_base()
_ENGINE: AsyncEngine | None = None
_SESSION_FACTORY: sessionmaker | None = None
_DB_PATH: str | None = None


def get_ulid() -> str:
    """Return a fake ULID using random digits."""
    # NOTE: This is a placeholder implementation
    return "".join(str(random.randint(0, 9)) for _ in range(16))


class User(BaseAuth):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    handle_id = Column(String, unique=True)
    password_hash = Column(String)
    lock_until = Column(DateTime, nullable=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    login_fail_count = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False)


class RefreshToken(BaseAuth):
    __tablename__ = "refresh_tokens"
    id = Column(String, primary_key=True, default=get_ulid)
    exp = Column(Integer, index=True)
    user_id = Column(
        String,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )
    scopes = Column("scopes", String)


async def get_session():
    """
    Yield an AsyncSession backed by a cached engine to avoid creating
    a new SQLite connection pool on every request.
    """
    db_path = _get_db_path()
    session_factory = _get_session_factory()
    async with db_semaphore(db_path):
        async with session_factory() as session:
            yield session


def _get_session_factory() -> sessionmaker:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is not None:
        return _SESSION_FACTORY

    engine = get_engine()
    _SESSION_FACTORY = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return _SESSION_FACTORY


def get_engine() -> AsyncEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_sqlite_engine(_get_db_path(), echo=False)
    return _ENGINE


def _get_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        db_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(db_dir, exist_ok=True)
        _DB_PATH = os.path.join(db_dir, "users.db")
    return _DB_PATH


async def dispose_auth_engine() -> None:
    """
    Dispose the cached OAuth2 engine (intended for FastAPI shutdown hook).
    """
    global _ENGINE, _SESSION_FACTORY
    engine = _ENGINE
    _ENGINE = None
    _SESSION_FACTORY = None
    if engine is not None:
        await engine.dispose()
