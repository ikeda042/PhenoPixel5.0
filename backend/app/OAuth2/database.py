from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os
import random

BaseAuth = declarative_base()


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
    dbname = "users.db"
    db_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, dbname)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}?timeout=30", echo=False
    )
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session
