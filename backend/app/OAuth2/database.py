from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os
import ulid

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    handle_id = Column(String, unique=True)
    password_hash = Column(String)
    lock_until = Column(DateTime, nullable=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    login_fail_count = Column(Integer, default=0, nullable=False)


def get_ulid() -> str:
    return ulid.new().str


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id = Column(String, primary_key=True, default=get_ulid)
    exp = Column(Integer, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )
    scopes = Column("scopes", String)


async def get_session(dbname: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "databases", dbname)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}?timeout=30", echo=False
    )
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session
