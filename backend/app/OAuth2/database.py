from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
import os
import random

BaseAuth = declarative_base()
_ENGINE: AsyncEngine | None = None
_SESSION_FACTORY: sessionmaker | None = None


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
    session_factory = _get_session_factory()
    async with session_factory() as session:
        yield session


def _get_session_factory() -> sessionmaker:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is not None:
        return _SESSION_FACTORY

    engine = _get_engine()
    _SESSION_FACTORY = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return _SESSION_FACTORY


def _get_engine() -> AsyncEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    dbname = "users.db"
    db_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, dbname)
    _ENGINE = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}?timeout=30", echo=False
    )
    return _ENGINE


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
