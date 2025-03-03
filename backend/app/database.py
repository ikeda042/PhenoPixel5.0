from sqlalchemy import Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os


Base = declarative_base()


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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "databases", dbname)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}?timeout=30", echo=False,connect_args={"check_same_thread": False}
    )
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session
