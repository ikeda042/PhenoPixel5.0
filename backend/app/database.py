import os
from typing import Dict

from sqlalchemy import BLOB, FLOAT, Column, Integer, String
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

from db_engine import create_sqlite_engine, db_semaphore

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
    db_path = _db_path(dbname)
    session_factory = _get_session_factory(dbname)
    async with db_semaphore(db_path):
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

    db_path = _db_path(dbname)
    engine = create_sqlite_engine(db_path, echo=False)
    _ENGINE_CACHE[dbname] = engine
    return engine


def _db_path(dbname: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(base_dir, "databases")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, dbname)


async def dispose_cached_engines() -> None:
    """
    Dispose all cached engines (invoked on application shutdown).
    """
    engines = list(_ENGINE_CACHE.values())
    _ENGINE_CACHE.clear()
    _SESSION_FACTORY_CACHE.clear()
    for engine in engines:
        await engine.dispose()
