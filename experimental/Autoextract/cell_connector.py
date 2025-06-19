from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel
from sqlalchemy import (
    Float,
    Integer,
    LargeBinary,
    String,
    create_engine,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

class Base(DeclarativeBase): 
    pass


class AsyncBase(AsyncAttrs, DeclarativeBase): 
    pass

class Cell(Base, AsyncBase):  

    __tablename__: str = "cells"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(String)
    label_experiment: Mapped[str] = mapped_column(String)
    manual_label: Mapped[int] = mapped_column(Integer)
    perimeter: Mapped[float] = mapped_column(Float)
    area: Mapped[float] = mapped_column(Float)
    img_ph: Mapped[bytes] = mapped_column(LargeBinary)
    img_fluo1: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    img_fluo2: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    contour: Mapped[bytes] = mapped_column(LargeBinary)
    center_x: Mapped[float] = mapped_column(Float)
    center_y: Mapped[float] = mapped_column(Float)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class CellBaseModel(BaseModel):
    cell_id: str
    label_experiment: str
    manual_label: int
    perimeter: float
    area: float
    img_ph: bytes
    img_fluo1: Optional[bytes] = None
    img_fluo2: Optional[bytes] = None
    contour: bytes
    center_x: float
    center_y: float
    user_id: Optional[str] = None

    class Config:
        from_attributes: bool = True 


class CellConnector:
    @classmethod
    def get_cells(
        cls,
        *,
        dbname: str = "test_database.db",
        label: str = "1",
    ) -> List[Cell]:
        db_url: str = f"sqlite:///{dbname}"
        engine = create_engine(db_url, future=True)

        Base.metadata.create_all(engine)

        SessionLocal: sessionmaker[Session] = sessionmaker(bind=engine, future=True)
        with SessionLocal() as session:
            cells: List[Cell] = (
                session.query(Cell)
                .filter(Cell.manual_label == label)
                .all()
            )
        return cells
    
    @classmethod
    async def get_cells_async(
        cls,
        *,
        dbname: str = "test_database.db",
        label: str = "1",
    ) -> List[Cell]:
        db_url: str = f"sqlite+aiosqlite:///{dbname}"
        engine = create_async_engine(db_url, future=True)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        AsyncSessionMaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=engine,
            expire_on_commit=False,
            future=True,
        )

        async with AsyncSessionMaker() as session:
            result = await session.scalars(
                select(Cell).where(Cell.manual_label == label)
            )
            cells: List[Cell] = result.all()

        await engine.dispose()
        return cells
