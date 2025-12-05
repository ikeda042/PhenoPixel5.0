from sqlalchemy import Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
import os
from typing import Dict


Base = declarative_base()
_ENGINE_CACHE: Dict[str, AsyncEngine] = {}
_SESSION_FACTORY_CACHE: Dict[str, sessionmaker] = {}


class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)
    user_id = Column(String, nullable=True)


async def get_session(dbname: str):
    """
    Yield an AsyncSession using a cached engine per DB to avoid creating
    unbounded engines (which was leading to timeouts after repeated access).
    """
    session_factory = _get_session_factory(dbname)
    async with session_factory() as session:
        yield session


def _get_session_factory(dbname: str) -> sessionmaker:
    if dbname in _SESSION_FACTORY_CACHE:
        return _SESSION_FACTORY_CACHE[dbname]

    engine = _get_engine(dbname)
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    _SESSION_FACTORY_CACHE[dbname] = async_session
    return async_session


def _get_engine(dbname: str) -> AsyncEngine:
    if dbname in _ENGINE_CACHE:
        return _ENGINE_CACHE[dbname]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "databases", dbname)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}?timeout=30",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    _ENGINE_CACHE[dbname] = engine
    return engine


async def dispose_cached_engines() -> None:
    """
    Dispose all cached engines (invoked on application shutdown).
    """
    engines = list(_ENGINE_CACHE.values())
    _ENGINE_CACHE.clear()
    _SESSION_FACTORY_CACHE.clear()
    for engine in engines:
        await engine.dispose()
