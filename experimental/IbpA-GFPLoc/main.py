import asyncio
from sqlalchemy import Column, Integer, String, BLOB, FLOAT, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle

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


class IbpaGfpLoc:
    def __init__(self) -> None:
        self._engine = create_async_engine(
            "sqlite+aiosqlite:///experimental/IbpA-GFPLoc/sk326gen120min.db?timeout=30",
            echo=False,
        )
        self._async_session = sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    @classmethod
    async def _async_imdecode(cls, data: bytes) -> np.ndarray:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            img = await loop.run_in_executor(
                executor, cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
            )
        return img

    @classmethod
    async def _draw_contour(
        cls, image: np.ndarray, contour: bytes, thickness: int = 1
    ) -> np.ndarray:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            contour = pickle.loads(contour)
            image = await loop.run_in_executor(
                executor,
                lambda: cv2.drawContours(image, contour, -1, (0, 255, 0), thickness),
            )
        return image

    async def _get_cells(self) -> list[Cell]:
        async with self._async_session() as session:
            result = await session.execute(select(Cell))
            cells = result.scalars().all()
            return cells

    @classmethod
    async def _parse_image(
        cls,
        data: bytes,
        contour: bytes | None = None,
        brightness_factor: float = 1.0,
    ):
        img = await cls._async_imdecode(data)
        if contour:
            img = await cls._draw_contour(img, contour)
        if brightness_factor != 1.0:
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

        ret, buffer = cv2.imencode(".png", img)
        if ret:
            cv2.imwrite("experimental/IbpA-GFPLoc/images/output_image.png", img)

        return {"status": "success", "message": "Image saved to output_image.png"}

    async def main(self):
        cells: list[Cell] = await self._get_cells()
        print(cells)
        cell1: Cell = cells[0]
        await self._parse_image(cell1.img_ph, cell1.contour)


if __name__ == "__main__":
    ibpa_gfp_loc = IbpaGfpLoc()
    asyncio.run(ibpa_gfp_loc.main())
