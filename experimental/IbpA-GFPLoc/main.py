import asyncio
from sqlalchemy import Column, Integer, String, BLOB, FLOAT, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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


async def main():
    engine = create_async_engine(
        "sqlite+aiosqlite:///experimental/IbpA-GFPLoc/sk326gen120min.db?timeout=30",
        echo=False,
    )
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        result = await session.execute(select(Cell))
        cells = result.scalars().all()
        print(cells)


async def async_imdecode(data: bytes) -> np.ndarray:
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=10) as executor:
        img = await loop.run_in_executor(
            executor, cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
        )
    return img


async def parse_image(
    data: bytes,
    contour: bytes | None = None,
    scale_bar: bool = False,
    brightness_factor: float = 1.0,
):
    # Decode the main image using async_imdecode
    img = await async_imdecode(data)

    # Draw contour if provided (implement your own logic here)
    if contour:
        # Example: decode contour data and draw
        # contour_pts = np.frombuffer(contour, dtype=np.int32).reshape(-1, 1, 2)
        # cv2.drawContours(img, [contour_pts], -1, (0, 255, 0), 2)
        pass

    # Adjust brightness if needed
    if brightness_factor != 1.0:
        img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

    # Draw scale bar if needed (implement your own logic here)
    if scale_bar:
        # Example placeholder function call
        # img = draw_scale_bar_with_centered_text(img)
        pass

    # Encode the image to PNG in a blocking call
    ret, buffer = cv2.imencode(".png", img)
    if ret:
        # Save the image locally
        with open("output_image.png", "wb") as f:
            f.write(buffer)

    return {"status": "success", "message": "Image saved to output_image.png"}


if __name__ == "__main__":
    asyncio.run(main())
