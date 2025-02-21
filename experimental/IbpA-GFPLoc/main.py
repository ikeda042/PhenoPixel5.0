import asyncio
import cv2
import numpy as np
import math
from sqlalchemy import Column, Integer, String, BLOB, FLOAT, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from concurrent.futures import ThreadPoolExecutor
import pickle

Base = declarative_base()


class Cell(Base):
    """
    ORM model representing a cell.

    Attributes:
        id (int): Primary key.
        cell_id (str): Identifier of the cell.
        label_experiment (str): Experiment label.
        manual_label (int): Manual label assigned to the cell.
        perimeter (float): Perimeter of the cell.
        area (float): Area of the cell.
        img_ph (bytes): Phase image data.
        img_fluo1 (bytes): Fluorescence image data (channel 1).
        img_fluo2 (bytes): Fluorescence image data (channel 2).
        contour (bytes): Pickled contour data.
        center_x (float): X coordinate of the cell center.
        center_y (float): Y coordinate of the cell center.
    """

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
    """
    Class for handling image processing and database operations for the IbpA-GFPLoc experiment.
    """

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
        """
        Asynchronously decode image data from bytes.

        Args:
            data (bytes): Encoded image data.

        Returns:
            np.ndarray: Decoded image in BGR format.
        """
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
        """
        Asynchronously draw contours on an image.

        Args:
            image (np.ndarray): The input image.
            contour (bytes): Pickled contour data.
            thickness (int, optional): Thickness of the drawn contour. Defaults to 1.

        Returns:
            np.ndarray: Image with the contours drawn.
        """
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
        """
        Asynchronously retrieve cell data from the database.

        Returns:
            list[Cell]: A list of Cell objects.
        """
        async with self._async_session() as session:
            result = await session.execute(select(Cell))
            cells = result.scalars().all()
            return cells

    @classmethod
    def _subtract_background(
        cls, gray_img: np.ndarray, kernel_size: int = 21
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Subtract the background from a grayscale image using morphological opening.

        The background is estimated by applying a morphological opening operation with an elliptical kernel.
        This operation removes small bright regions (such as cells) while preserving the larger background structure.
        The estimated background is then subtracted from the original grayscale image using cv2.subtract,
        which clips negative values to zero.

        Mathematical formulation:
            B(x,y) = \mathrm{morph\_open}(I(x,y))
            I\_sub(x,y) = I(x,y) - B(x,y)

        LaTeX生コード:
        ```
        B(x,y) = \mathrm{morph\_open}(I(x,y))
        I\_sub(x,y) = I(x,y) - B(x,y)
        ```

        Args:
            gray_img (np.ndarray): Grayscale image.
            kernel_size (int, optional): Size of the elliptical kernel used for the morphological operation. Defaults to 21.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Background-subtracted grayscale image.
                - Estimated background image.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        background = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
        subtracted = cv2.subtract(gray_img, background)
        return subtracted, background

    @classmethod
    async def _parse_image(
        cls,
        data: bytes,
        contour: bytes | None = None,
        brightness_factor: float = 1.0,
        save_name: str = "output_image.png",
        fill: bool = False,
        save_background: bool = False,
    ):
        """
        Process an image by decoding, converting to grayscale, subtracting background, adjusting brightness,
        and optionally processing contours.

        Args:
            data (bytes): Encoded image data.
            contour (bytes | None, optional): Pickled contour data. Defaults to None.
            brightness_factor (float, optional): Factor for brightness adjustment. Defaults to 1.0.
            save_name (str, optional): File name for saving the processed image. Defaults to "output_image.png".
            fill (bool, optional): If True, applies a mask to keep only the interior of the contour. Defaults to False.
            save_background (bool, optional): If True, saves the estimated background image for visualization. Defaults to False.

        Returns:
            dict: Dictionary containing the status, message, and the processed image array.
        """
        img = await cls._async_imdecode(data)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 背景推定と差し引き
        subtracted, background = cls._subtract_background(gray_img)
        if save_background:
            cv2.imwrite(
                f"experimental/IbpA-GFPLoc/images/{save_name}_background.png",
                background,
            )
        gray_img = subtracted
        if brightness_factor != 1.0:
            gray_img = cv2.convertScaleAbs(gray_img, alpha=brightness_factor, beta=0)
        if contour:
            if fill:
                loaded_contour = pickle.loads(contour)
                if not isinstance(loaded_contour, list):
                    loaded_contour = [loaded_contour]
                mask = np.zeros(gray_img.shape, dtype=np.uint8)
                cv2.fillPoly(mask, loaded_contour, 255)
                cv2.polylines(mask, loaded_contour, isClosed=True, color=0, thickness=1)
                gray_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)
            else:
                loaded_contour = pickle.loads(contour)
                if not isinstance(loaded_contour, list):
                    loaded_contour = [loaded_contour]
                gray_img = cv2.drawContours(
                    gray_img, loaded_contour, -1, (0,), thickness=1
                )
        ret, buffer = cv2.imencode(".png", gray_img)
        if ret:
            cv2.imwrite(f"experimental/IbpA-GFPLoc/images/{save_name}", gray_img)
        return {
            "status": "success",
            "message": f"Image saved to {save_name}",
            "image": gray_img,
        }

    @staticmethod
    def combine_images(
        images: list[np.ndarray], output_filename: str = "combined.png"
    ) -> None:
        """
        Combine a list of images into a single grid image and save it.

        各画像サイズが異なる場合、最大の高さ・幅に合わせて黒いパディングを追加します。
        グリッドは、画像数の平方根を元に自動計算されます。

        Args:
            images (list[np.ndarray]): 処理済み画像のリスト。
            output_filename (str, optional): 保存するファイル名。 Defaults to "combined.png".
        """
        # グレースケールの場合はBGRに変換
        converted_images = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            converted_images.append(img)

        # 各画像の最大の高さと幅を取得
        max_h = max(img.shape[0] for img in converted_images)
        max_w = max(img.shape[1] for img in converted_images)
        padded_images = []
        for img in converted_images:
            h, w = img.shape[:2]
            padded = np.zeros((max_h, max_w, 3), dtype=img.dtype)
            padded[:h, :w] = img
            padded_images.append(padded)
        num_images = len(padded_images)
        grid_cols = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_cols)
        rows = []
        for i in range(grid_rows):
            row_imgs = []
            for j in range(grid_cols):
                idx = i * grid_cols + j
                if idx < num_images:
                    row_imgs.append(padded_images[idx])
                else:
                    row_imgs.append(
                        np.zeros((max_h, max_w, 3), dtype=converted_images[0].dtype)
                    )
            rows.append(np.hstack(row_imgs))
        combined_image = np.vstack(rows)
        cv2.imwrite(output_filename, combined_image)
        print(f"Combined image saved to {output_filename}")

    async def main(self):
        """
        Retrieve cell data from the database, process each cell's image,
        and combine all processed images into one image file ("combined.png").
        """
        cells: list[Cell] = await self._get_cells()
        processed_images = []
        for cell in cells:
            result = await self._parse_image(
                data=cell.img_fluo1,
                contour=cell.contour,
                save_name=f"{cell.cell_id}.png",
                fill=True,
                save_background=True,
            )
            processed_img = result.get("image")
            if processed_img is not None:
                processed_images.append(processed_img)
        if processed_images:
            self.combine_images(
                processed_images,
                output_filename="experimental/IbpA-GFPLoc/images/combined.png",
            )


if __name__ == "__main__":
    ibpa_gfp_loc = IbpaGfpLoc()
    asyncio.run(ibpa_gfp_loc.main())
