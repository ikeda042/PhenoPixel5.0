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
            loaded_contour = pickle.loads(contour)
            if not isinstance(loaded_contour, list):
                loaded_contour = [loaded_contour]
            image = await loop.run_in_executor(
                executor,
                lambda: cv2.drawContours(
                    image, loaded_contour, -1, (0, 255, 0), thickness
                ),
            )
        return image

    async def _get_cells(self) -> list[Cell]:
        async with self._async_session() as session:
            result = await session.execute(select(Cell))
            cells = result.scalars().all()
            return cells

    @classmethod
    def _subtract_background(
        cls, gray_img: np.ndarray, kernel_size: int = 21
    ) -> np.ndarray:
        # エリプス形状のカーネルを生成
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        # モルフォロジーオープニングで背景を推定
        background = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
        # 背景を引き算（cv2.subtractは負の値を0にしてくれる）
        subtracted = cv2.subtract(gray_img, background)
        return subtracted

    @classmethod
    async def _parse_image(
        cls,
        data: bytes,
        contour: bytes | None = None,
        brightness_factor: float = 1.0,
        save_name: str = "output_image.png",
        fill: bool = False,
    ):
        # 画像をデコード
        img = await cls._async_imdecode(data)
        # 絶対輝度解析のため、グレースケールに変換
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 背景を推定して引き算
        gray_img = cls._subtract_background(gray_img)

        # 輝度補正が必要な場合
        if brightness_factor != 1.0:
            gray_img = cv2.convertScaleAbs(gray_img, alpha=brightness_factor, beta=0)

        if contour:
            if fill:
                loaded_contour = pickle.loads(contour)
                if not isinstance(loaded_contour, list):
                    loaded_contour = [loaded_contour]
                # 画像と同じサイズのマスク（初期値0）を作成
                mask = np.zeros(gray_img.shape, dtype=np.uint8)
                # 輪郭内部を255で塗りつぶし
                cv2.fillPoly(mask, loaded_contour, 255)
                # 輪郭線のみを黒（0）で描画（輪郭内部は保持）
                cv2.polylines(mask, loaded_contour, isClosed=True, color=0, thickness=1)
                # マスク適用：内部のみ輝度を保持
                gray_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)
            else:
                # オプション：輪郭を画像上に描画（視覚確認用）
                loaded_contour = pickle.loads(contour)
                if not isinstance(loaded_contour, list):
                    loaded_contour = [loaded_contour]
                gray_img = cv2.drawContours(
                    gray_img, loaded_contour, -1, (0,), thickness=1
                )

        ret, buffer = cv2.imencode(".png", gray_img)
        if ret:
            cv2.imwrite(f"experimental/IbpA-GFPLoc/images/{save_name}", gray_img)

        return {"status": "success", "message": f"Image saved to {save_name}"}

    async def main(self):
        cells: list[Cell] = await self._get_cells()
        print(cells)
        cell: Cell = cells[0]
        # 例として、塗りつぶしオプション有効の場合
        await self._parse_image(
            data=cell.img_fluo1,
            contour=cell.contour,
            save_name=f"{cell.cell_id}.png",
            fill=True,
        )


if __name__ == "__main__":
    ibpa_gfp_loc = IbpaGfpLoc()
    asyncio.run(ibpa_gfp_loc.main())
