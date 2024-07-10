from sqlalchemy import Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os


Base = declarative_base()
Base2 = declarative_base()


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


class Cell2(Base2):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True) | None
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)
    max_brightness = Column(FLOAT)
    min_brightness = Column(FLOAT)
    mean_brightness_raw = Column(FLOAT)
    mean_brightness_normalized = Column(FLOAT)
    median_brightness_raw = Column(FLOAT)
    median_brightness_normalized = Column(FLOAT)
    ph_max_brightness = Column(FLOAT)
    ph_min_brightness = Column(FLOAT)
    ph_mean_brightness_raw = Column(FLOAT)
    ph_mean_brightness_normalized = Column(FLOAT)
    ph_median_brightness_raw = Column(FLOAT)
    ph_median_brightness_normalized = Column(FLOAT)


# async def get_session(dbname: str):
#     engine = create_async_engine(f"sqlite+aiosqlite:///{dbname}", echo=False)
#     async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
#     async with async_session() as session:
#         yield session


async def get_session(dbname: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "databases", dbname)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session
