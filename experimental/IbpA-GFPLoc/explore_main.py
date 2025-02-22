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
        area (float): Area of the cell (以前はデータベース上の値を使用していたが、今回の定量化では輪郭から計算するため使用しません)。
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
    area: float = Column(FLOAT)
    perimeter: float = Column(FLOAT)
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
        annotation: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate a jet colormap image array of the processed image,
        ensuring that the cell centroid is at (0,0) and the axis scale is unified.
        また、annotation が指定された場合は、fig上にテキスト描画する。
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
        if annotation is not None:
            ax.text(
                0.05,
                0.95,
                annotation,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                color="white",
                bbox=dict(facecolor="black", alpha=0.5),
            )
        fig.canvas.draw()
        # buffer_rgbaを使ってRGBAのバッファを取得
        jet_img: np.ndarray = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        # RGBAの形状にreshapeする（4チャネルなので）
        jet_img = jet_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # αチャネルを削除してRGB画像に変換する
        jet_img = jet_img[..., :3]
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
        roi_float = roi_img.astype(np.float64)
        scale_factor = 1.0
        low_threshold = 50.0
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

    def quantify_cell_simple(self, processed_img: np.ndarray, cell: Cell) -> float:
        """
        細胞内の合計輝度を輪郭内の面積で割ることで蛋白質凝集強度を定量化する関数。

        数式:
            score = (細胞内の総輝度) / (輪郭内の面積)
        ```
        """
        total_intensity = float(np.sum(processed_img))
        loaded_contour = pickle.loads(cell.contour)
        if not isinstance(loaded_contour, list):
            loaded_contour = [loaded_contour]
        contour_area = sum(cv2.contourArea(cnt) for cnt in loaded_contour)
        return total_intensity / contour_area if contour_area != 0 else 0.0

    @staticmethod
    def quantify_cell_peak(processed_img: np.ndarray, cell: Cell) -> float:
        """
        細胞内の輝度が高い上位10%のピクセルの平均値を返す関数。

        数式:
            \text{peak} = \frac{1}{N_{top}} \sum_{i=1}^{N_{top}} I_{(i)}
        （ただし、N_{top} は細胞内のピクセル数の10%を表し、I_{(i)} は細胞内の輝度を高い順に並べた値）

        LaTeX生コード:
        ```
        \text{peak} = \frac{1}{N_{top}} \sum_{i=1}^{N_{top}} I_{(i)}
        ```
        """
        cell_pixels = processed_img[processed_img > 0]
        if cell_pixels.size == 0:
            return 0.0
        sorted_pixels = np.sort(cell_pixels)[::-1]
        num_top = int(np.ceil(0.1 * len(sorted_pixels)))
        top_pixels = sorted_pixels[:num_top]
        return float(np.mean(top_pixels))

    def quantify_cell_composite(
        self,
        simple: float,
        peak: float,
        global_max_simple: float,
        global_max_peak: float,
    ) -> float:
        """
        総合スコアはピークスコアのみを用いる。

        LaTeX生コード:
        ```
        \text{composite} = \text{peak}
        ```
        """
        return peak

    async def main(self) -> None:
        """
        Retrieve cell data from the database, process each cell's image,
        generate jet-colormap plots with unified scale and cell centroid at (0,0),
        combine all processed images into one image file, and quantify protein aggregation.
        各セルのjet画像には、定量化したシンプルスコア、ピークスコア、総合スコア（＝ピークスコア）を注釈として描画する。
        最終的に、総合スコア順にraw画像とjet画像の両方を結合して出力する。
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

        cell_images: List[Tuple[Cell, np.ndarray, Dict[str, float]]] = []
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
                intensity_simple = self.quantify_cell_simple(processed_img, cell)
                intensity_peak = IbpaGfpLoc.quantify_cell_peak(processed_img, cell)
                print(
                    f"Cell {cell.cell_id}: Simple = {intensity_simple:.2f}, Peak = {intensity_peak:.2f}"
                )
                cell_images.append(
                    (
                        cell,
                        processed_img,
                        {"simple": intensity_simple, "peak": intensity_peak},
                    )
                )

        if cell_images:
            global_max_brightness: float = max(np.max(img) for _, img, _ in cell_images)
            global_max_simple: float = max(info["simple"] for _, _, info in cell_images)
            global_max_peak: float = max(info["peak"] for _, _, info in cell_images)

            cell_images_with_jet: List[
                Tuple[Cell, np.ndarray, Dict[str, float], np.ndarray]
            ] = []
            for cell, processed_img, intensities in cell_images:
                composite = self.quantify_cell_composite(
                    intensities["simple"],
                    intensities["peak"],
                    global_max_simple,
                    global_max_peak,
                )
                intensities["composite"] = composite
                # annotationでは総合スコアはピークスコアのみなので、シンプルスコアは省略可
                annotation = f"P: {intensities['peak']:.2f}, C: {composite:.2f}"
                jet_img: np.ndarray = IbpaGfpLoc._generate_jet_image_array(
                    processed_img,
                    cell,
                    global_extent,
                    global_max_brightness,
                    annotation,
                )
                cell_images_with_jet.append((cell, processed_img, intensities, jet_img))

            # 総合スコア(composite)の降順にソート（＝ピークスコア順にソート）
            cell_images_sorted = sorted(
                cell_images_with_jet, key=lambda x: x[2]["composite"], reverse=True
            )
            processed_images_sorted: List[np.ndarray] = [
                entry[1] for entry in cell_images_sorted
            ]
            jet_images_sorted: List[np.ndarray] = [
                entry[3] for entry in cell_images_sorted
            ]

            IbpaGfpLoc.combine_images(
                processed_images_sorted,
                output_filename="experimental/IbpA-GFPLoc/combined_raw.png",
            )
            IbpaGfpLoc.combine_images(
                jet_images_sorted,
                output_filename="experimental/IbpA-GFPLoc/combined_jet.png",
            )


images_dir: str = "experimental/IbpA-GFPLoc/images/"
if os.path.exists(images_dir):
    for file in os.listdir(images_dir):
        file_path: str = os.path.join(images_dir, file)
        os.remove(file_path)
ibpa_gfp_loc: IbpaGfpLoc = IbpaGfpLoc()
asyncio.run(ibpa_gfp_loc.main())
