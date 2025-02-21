import asyncio
import cv2
import numpy as np
import math
import pickle
import os
from typing import Optional, List, Tuple, Dict, Any, Union

from sqlalchemy import Column, Integer, String, BLOB, FLOAT, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker, declarative_base
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy.optimize import curve_fit

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
        img_fluo1 (Optional[bytes]): Fluorescence image data (channel 1).
        img_fluo2 (Optional[bytes]): Fluorescence image data (channel 2).
        contour (bytes): Pickled contour data.
        center_x (float): X coordinate of the cell center.
        center_y (float): Y coordinate of the cell center.
    """

    __tablename__ = "cells"
    id: int = Column(Integer, primary_key=True)
    cell_id: str = Column(String)
    label_experiment: str = Column(String)
    manual_label: int = Column(Integer)
    perimeter: float = Column(FLOAT)
    area: float = Column(FLOAT)
    img_ph: bytes = Column(BLOB)
    img_fluo1: Optional[bytes] = Column(BLOB, nullable=True)
    img_fluo2: Optional[bytes] = Column(BLOB, nullable=True)
    contour: bytes = Column(BLOB)
    center_x: float = Column(FLOAT)
    center_y: float = Column(FLOAT)


class IbpaGfpLoc:
    """
    Class for handling image processing and database operations for the IbpA-GFPLoc experiment.
    """

    def __init__(self) -> None:
        self._engine: AsyncEngine = create_async_engine(
            "sqlite+aiosqlite:///experimental/IbpA-GFPLoc/sk326gen120min.db?timeout=30",
            echo=False,
        )
        self._async_session: sessionmaker = sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    @classmethod
    async def _async_imdecode(cls, data: bytes) -> np.ndarray:
        """
        Asynchronously decode image data from bytes.
        """
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            img: np.ndarray = await loop.run_in_executor(
                executor, cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
            )
        return img

    @classmethod
    async def _draw_contour(
        cls, image: np.ndarray, contour: bytes, thickness: int = 1
    ) -> np.ndarray:
        """
        Asynchronously draw contours on an image.
        """
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            loaded_contour: Any = pickle.loads(contour)
            if not isinstance(loaded_contour, list):
                loaded_contour = [loaded_contour]
            image = await loop.run_in_executor(
                executor,
                lambda: cv2.drawContours(
                    image, loaded_contour, -1, (0, 255, 0), thickness
                ),
            )
        return image

    async def _get_cells(self, label: str = "1") -> List[Cell]:
        """
        Asynchronously retrieve cell data from the database.
        """
        async with self._async_session() as session:
            result: Any = await session.execute(
                select(Cell).where(Cell.manual_label == label)
            )
            cells: List[Cell] = result.scalars().all()
            return cells

    @classmethod
    def _subtract_background(
        cls, gray_img: np.ndarray, kernel_size: int = 21
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Subtract the background from a grayscale image using morphological opening.

        Mathematical formulation:
            B(x,y) = \mathrm{morph\_open}(I(x,y))
            I\_sub(x,y) = I(x,y) - B(x,y)

        LaTeX生コード:
        ```
        B(x,y) = \mathrm{morph\_open}(I(x,y))
        I\_sub(x,y) = I(x,y) - B(x,y)
        ```
        """
        kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        background: np.ndarray = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
        subtracted: np.ndarray = cv2.subtract(gray_img, background)
        return subtracted, background

    @classmethod
    async def _parse_image(
        cls,
        data: bytes,
        contour: Optional[bytes] = None,
        brightness_factor: float = 1.0,
        save_name: str = "output_image.png",
        fill: bool = False,
        save_background: bool = False,
    ) -> Dict[str, Union[str, np.ndarray]]:
        """
        Process an image by decoding, converting to grayscale, subtracting background,
        adjusting brightness, and optionally processing contours.
        """
        img: np.ndarray = await cls._async_imdecode(data)
        gray_img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subtracted: np.ndarray
        background: np.ndarray
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
            loaded_contour: Any = pickle.loads(contour)
            if fill:
                if not isinstance(loaded_contour, list):
                    loaded_contour = [loaded_contour]
                mask: np.ndarray = np.zeros(gray_img.shape, dtype=np.uint8)
                cv2.fillPoly(mask, loaded_contour, 255)
                cv2.polylines(mask, loaded_contour, isClosed=True, color=0, thickness=1)
                gray_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)
            else:
                if not isinstance(loaded_contour, list):
                    loaded_contour = [loaded_contour]
                gray_img = cv2.drawContours(
                    gray_img, loaded_contour, -1, (0,), thickness=1
                )
        ret: bool
        buffer: np.ndarray
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
        images: List[np.ndarray], output_filename: str = "combined.png"
    ) -> None:
        """
        Combine a list of images into a single grid image and save it.
        """
        converted_images: List[np.ndarray] = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            converted_images.append(img)

        max_h: int = max(img.shape[0] for img in converted_images)
        max_w: int = max(img.shape[1] for img in converted_images)
        padded_images: List[np.ndarray] = []
        for img in converted_images:
            h: int = img.shape[0]
            w: int = img.shape[1]
            padded: np.ndarray = np.zeros((max_h, max_w, 3), dtype=img.dtype)
            padded[:h, :w] = img
            padded_images.append(padded)
        num_images: int = len(padded_images)
        grid_cols: int = math.ceil(math.sqrt(num_images))
        grid_rows: int = math.ceil(num_images / grid_cols)
        rows: List[np.ndarray] = []
        for i in tqdm(range(grid_rows)):
            row_imgs: List[np.ndarray] = []
            for j in range(grid_cols):
                idx: int = i * grid_cols + j
                if idx < num_images:
                    row_imgs.append(padded_images[idx])
                else:
                    row_imgs.append(
                        np.zeros((max_h, max_w, 3), dtype=converted_images[0].dtype)
                    )
            rows.append(np.hstack(row_imgs))
        combined_image: np.ndarray = np.vstack(rows)
        cv2.imwrite(output_filename, combined_image)
        print(f"Combined image saved to {output_filename}")

    @staticmethod
    def _generate_jet_image_array(
        processed_img: np.ndarray,
        cell: Cell,
        global_extent: float,
        global_max_brightness: float,
    ) -> np.ndarray:
        """
        Generate a jet colormap image array of the processed image,
        ensuring that the cell centroid is at (0,0) and the axis scale is unified.
        """
        h, w = processed_img.shape
        extent: List[float] = [
            -cell.center_x,
            w - cell.center_x,
            -cell.center_y,
            h - cell.center_y,
        ]
        fig, ax = plt.subplots()
        ax.imshow(
            processed_img,
            cmap="jet_r",
            extent=extent,
            origin="lower",
            vmin=0,
            vmax=global_max_brightness,
        )
        ax.set_xlim([-global_extent, global_extent])
        ax.set_ylim([-global_extent, global_extent])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Cell {cell.cell_id}")
        fig.canvas.draw()
        jet_img: np.ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        jet_img = jet_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return jet_img

    @staticmethod
    def twoD_Gaussian(
        coordinates: Tuple[np.ndarray, np.ndarray],
        A: float,
        x0: float,
        y0: float,
        sigma_x: float,
        sigma_y: float,
        offset: float,
    ) -> np.ndarray:
        """
        2次元ガウス関数モデル

        数式:
            f(x,y) = A * exp(-(((x-x0)^2/(2\sigma_x^2)) + ((y-y0)^2/(2\sigma_y^2)))) + offset

        LaTeX生コード:
        ```
        f(x,y) = A \exp\left(-\left(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}\right)\right) + \text{offset}
        ```
        """
        x, y = coordinates
        exp_component = np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2))
        )
        return A * exp_component + offset

    @staticmethod
    def fit_gaussian_to_roi(roi_img: np.ndarray) -> Optional[Dict[str, float]]:
        """
        ROI内の画像に対して2次元ガウスフィッティングを実施する関数。

        Returns:
            dict: フィッティングパラメータと、ガウスの積分強度
                  integrated_intensity = A * 2πσ_xσ_y
        """
        h, w = roi_img.shape
        # ROI画像を浮動小数点に変換してスケール調整
        roi_float = roi_img.astype(np.float64)
        scale_factor = 1.0
        low_threshold = 50.0  # スケーリング対象の閾値（必要に応じて調整）
        if np.max(roi_float) < low_threshold and np.max(roi_float) != 0:
            scale_factor = low_threshold / np.max(roi_float)
            roi_scaled = roi_float * scale_factor
        else:
            roi_scaled = roi_float

        x = np.arange(0, w, 1)
        y = np.arange(0, h, 1)
        x, y = np.meshgrid(x, y)
        xdata = np.vstack((x.ravel(), y.ravel()))
        ydata = roi_scaled.ravel()

        A_init = np.max(roi_scaled) - np.min(roi_scaled)
        x0_init = w / 2
        y0_init = h / 2
        sigma_x_init = w / 4
        sigma_y_init = h / 4
        offset_init = np.min(roi_scaled)
        initial_guess = (
            A_init,
            x0_init,
            y0_init,
            sigma_x_init,
            sigma_y_init,
            offset_init,
        )
        try:
            popt, _ = curve_fit(
                IbpaGfpLoc.twoD_Gaussian, xdata, ydata, p0=initial_guess
            )
        except Exception as e:
            print(f"フィッティングに失敗しました: {e}")
            return None
        A, x0, y0, sigma_x, sigma_y, offset = popt
        # スケーリング前の値に戻す
        A_original = A / scale_factor
        offset_original = offset / scale_factor
        integrated_intensity = A_original * 2 * np.pi * sigma_x * sigma_y
        return {
            "A": A_original,
            "x0": x0,
            "y0": y0,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "offset": offset_original,
            "integrated_intensity": integrated_intensity,
        }

    @staticmethod
    def detect_dots(image: np.ndarray) -> List[cv2.KeyPoint]:
        """
        ドット（蛋白質凝集箇所）を検出するためにSimpleBlobDetectorを使用する関数。
        低輝度のドットも検出できるよう、minThreshold を 1 に設定。
        """
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 1
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 500
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)
        return keypoints

    def quantify_cell(self, processed_img: np.ndarray) -> float:
        """
        画像内の各ドットに対して2次元ガウスフィッティングを行い、
        各ドットの積分値を合計することで、蛋白質凝集強度を定量化する関数。
        """
        keypoints = IbpaGfpLoc.detect_dots(processed_img)
        total_intensity = 0.0
        for kp in keypoints:
            x_center, y_center = int(round(kp.pt[0])), int(round(kp.pt[1]))
            half_size = int(round(kp.size / 2)) if kp.size > 0 else 10
            x1 = max(x_center - half_size, 0)
            y1 = max(y_center - half_size, 0)
            x2 = min(x_center + half_size, processed_img.shape[1])
            y2 = min(y_center + half_size, processed_img.shape[0])
            roi = processed_img[y1:y2, x1:x2]
            result = IbpaGfpLoc.fit_gaussian_to_roi(roi)
            if result is not None:
                total_intensity += result["integrated_intensity"]
        return total_intensity

    def quantify_cell_sum(self, processed_img: np.ndarray) -> float:
        """
        画素値総和による蛋白質凝集強度の定量化。
        """
        keypoints = IbpaGfpLoc.detect_dots(processed_img)
        total_intensity = 0.0
        for kp in keypoints:
            x_center, y_center = int(round(kp.pt[0])), int(round(kp.pt[1]))
            half_size = int(round(kp.size / 2)) if kp.size > 0 else 10
            x1 = max(x_center - half_size, 0)
            y1 = max(y_center - half_size, 0)
            x2 = min(x_center + half_size, processed_img.shape[1])
            y2 = min(y_center + half_size, processed_img.shape[0])
            roi = processed_img[y1:y2, x1:x2]
            total_intensity += float(np.sum(roi))
        return total_intensity

    def quantify_cell_area_weighted(self, processed_img: np.ndarray) -> float:
        """
        面積加重平均による蛋白質凝集強度の定量化。
        """
        keypoints = IbpaGfpLoc.detect_dots(processed_img)
        total_intensity = 0.0
        for kp in keypoints:
            x_center, y_center = int(round(kp.pt[0])), int(round(kp.pt[1]))
            half_size = int(round(kp.size / 2)) if kp.size > 0 else 10
            x1 = max(x_center - half_size, 0)
            y1 = max(y_center - half_size, 0)
            x2 = min(x_center + half_size, processed_img.shape[1])
            y2 = min(y_center + half_size, processed_img.shape[0])
            roi = processed_img[y1:y2, x1:x2]
            area = roi.size
            mean_intensity = float(np.mean(roi))
            total_intensity += mean_intensity * area
        return total_intensity

    def quantify_cell_connected_components(
        self, processed_img: np.ndarray, threshold: int = 50
    ) -> float:
        """
        しきい値処理と連結成分解析による蛋白質凝集強度の定量化。
        """
        _, binary_img = cv2.threshold(processed_img, threshold, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(binary_img.astype(np.uint8))
        total_intensity = 0.0
        for label in range(1, num_labels):
            mask = labels == label
            total_intensity += float(np.sum(processed_img[mask]))
        return total_intensity

    async def main(self) -> None:
        """
        Retrieve cell data from the database, process each cell's image,
        generate jet-colormap plots with unified scale and cell centroid at (0,0),
        combine all processed images into one image file, and quantify protein aggregation.
        """
        jet_dir: str = "experimental/IbpA-GFPLoc/jet"
        os.makedirs(jet_dir, exist_ok=True)

        cells: List[Cell] = await self._get_cells()
        global_extent: float = 0.0
        for cell in cells:
            # Assume cell.img_fluo1 is not None
            img: np.ndarray = await self._async_imdecode(cell.img_fluo1)  # type: ignore
            gray_img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray_img.shape
            left: float = cell.center_x
            right: float = w - cell.center_x
            bottom: float = cell.center_y
            top: float = h - cell.center_y
            cell_extent: float = max(left, right, bottom, top)
            if cell_extent > global_extent:
                global_extent = cell_extent

        cell_images: List[Tuple[Cell, np.ndarray]] = []
        processed_images: List[np.ndarray] = []
        jet_images: List[np.ndarray] = []
        for cell in tqdm(cells):
            result: Dict[str, Union[str, np.ndarray]] = await self._parse_image(
                data=cell.img_fluo1,  # type: ignore
                contour=cell.contour,
                brightness_factor=1,
                save_name=f"{cell.cell_id}.png",
                fill=True,
                save_background=False,
            )
            processed_img: Optional[np.ndarray] = result.get("image")  # type: ignore
            if processed_img is not None:
                cell_images.append((cell, processed_img))
                processed_images.append(processed_img)
                # 蛋白質凝集強度の定量化（各手法による評価）
                intensity_gaussian = self.quantify_cell(processed_img)
                intensity_sum = self.quantify_cell_sum(processed_img)
                intensity_area = self.quantify_cell_area_weighted(processed_img)
                intensity_cc = self.quantify_cell_connected_components(processed_img)
                print(f"Cell {cell.cell_id}: Gaussian Intensity = {intensity_gaussian}")
                print(f"Cell {cell.cell_id}: Sum Intensity = {intensity_sum}")
                print(
                    f"Cell {cell.cell_id}: Area Weighted Intensity = {intensity_area}"
                )
                print(
                    f"Cell {cell.cell_id}: Connected Components Intensity = {intensity_cc}"
                )
        if processed_images:
            global_max_brightness: float = max(np.max(img) for _, img in cell_images)
            for cell, processed_img in cell_images:
                jet_img: np.ndarray = IbpaGfpLoc._generate_jet_image_array(
                    processed_img, cell, global_extent, global_max_brightness
                )
                jet_images.append(jet_img)
            IbpaGfpLoc.combine_images(
                processed_images,
                output_filename="experimental/IbpA-GFPLoc/combined_raw.png",
            )
            IbpaGfpLoc.combine_images(
                jet_images,
                output_filename="experimental/IbpA-GFPLoc/combined_jet.png",
            )


if __name__ == "__main__":
    images_dir: str = "experimental/IbpA-GFPLoc/images/"
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            file_path: str = os.path.join(images_dir, file)
            os.remove(file_path)
    ibpa_gfp_loc: IbpaGfpLoc = IbpaGfpLoc()
    asyncio.run(ibpa_gfp_loc.main())
