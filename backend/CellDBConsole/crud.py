from __future__ import annotations
from CellDBConsole.schemas import CellId, CellMorhology
from database import get_session, Cell
from sqlalchemy.future import select
from exceptions import CellNotFoundError
import cv2
import numpy as np
from numpy.linalg import eig, inv
from fastapi.responses import StreamingResponse
import io
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
import scipy
from scipy.optimize import minimize


class SyncChores:
    @staticmethod
    def poly_fit(U: list[list[float]], degree: int = 1) -> list[float]:
        u1_values = np.array([i[1] for i in U])
        f_values = np.array([i[0] for i in U])
        W = np.vander(u1_values, degree + 1)

        return inv(W.T @ W) @ W.T @ f_values

    @staticmethod
    def calc_arc_length(theta: list[float], u_1_1: float, u_1_2: float) -> float:
        fx = lambda x: np.sqrt(
            1
            + sum(i * j * x ** (i - 1) for i, j in enumerate(theta[::-1][1:], start=1))
            ** 2
        )
        arc_length, _ = scipy.integrate.quad(fx, u_1_1, u_1_2, epsabs=1e-01)
        return arc_length

    @staticmethod
    def find_minimum_distance_and_point(coefficients, x_Q, y_Q):
        # 関数の定義
        def f_x(x):
            return sum(
                coefficient * x**i for i, coefficient in enumerate(coefficients[::-1])
            )

        # 点Qから関数上の点までの距離 D の定義
        def distance(x):
            return np.sqrt((x - x_Q) ** 2 + (f_x(x) - y_Q) ** 2)

        # scipyのminimize関数を使用して最短距離を見つける
        # 初期値は0とし、精度は低く設定して計算速度を向上させる
        result = minimize(
            distance, 0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-2}
        )

        # 最短距離とその時の関数上の点
        x_min = result.x[0]
        min_distance = distance(x_min)
        min_point = (x_min, f_x(x_min))

        return min_distance, min_point

    @staticmethod
    def basis_conversion(
        contour: list[list[int]],
        X: np.ndarray,
        center_x: float,
        center_y: float,
        coordinates_incide_cell: list[list[int]],
    ) -> list[list[float]]:
        Sigma = np.cov(X)
        eigenvalues, eigenvectors = eig(Sigma)
        if eigenvalues[1] < eigenvalues[0]:
            Q = np.array([eigenvectors[1], eigenvectors[0]])
            U = [Q.transpose() @ np.array([i, j]) for i, j in coordinates_incide_cell]
            U = [[j, i] for i, j in U]
            contour_U = [Q.transpose() @ np.array([j, i]) for i, j in contour]
            contour_U = [[j, i] for i, j in contour_U]
            center = [center_x, center_y]
            u1_c, u2_c = center @ Q
        else:
            Q = np.array([eigenvectors[0], eigenvectors[1]])
            U = [
                Q.transpose() @ np.array([j, i]).transpose()
                for i, j in coordinates_incide_cell
            ]
            contour_U = [Q.transpose() @ np.array([i, j]) for i, j in contour]
            center = [center_x, center_y]
            u2_c, u1_c = center @ Q

        u1 = [i[1] for i in U]
        u2 = [i[0] for i in U]
        u1_contour = [i[1] for i in contour_U]
        u2_contour = [i[0] for i in contour_U]
        min_u1, max_u1 = min(u1), max(u1)
        return u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U

    @staticmethod
    def calculate_volume_and_widths(
        u1_adj: list[float], u2_adj: list[float], split_num: int, deltaL: float
    ) -> tuple[float, list[float]]:
        volume = 0
        widths = []
        y_mean = 0
        for i in range(split_num):
            x_0 = min(u1_adj) + i * deltaL
            x_1 = min(u1_adj) + (i + 1) * deltaL
            points = [p for p in zip(u1_adj, u2_adj) if x_0 <= p[0] <= x_1]
            if points:
                y_mean = sum([i[1] for i in points]) / len(points)
            volume += y_mean**2 * np.pi * deltaL
            widths.append(y_mean)
        return volume, widths


class AsyncChores:
    @staticmethod
    async def async_imdecode(data: bytes) -> np.ndarray:
        """
        Decode an image from bytes.

        Parameters:
        - data: Image data in bytes.

        Returns:
        - Image in numpy array format.
        """
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            img = await loop.run_in_executor(
                executor, cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
            )
        return img

    @staticmethod
    async def async_cv2_imencode(img) -> tuple[bool, np.ndarray]:
        """
        Encode an image to PNG format.

        Parameters:
        - img: Image to encode.

        Returns:
        - Tuple containing success status and image buffer.
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            success, buffer = await loop.run_in_executor(
                executor, lambda: cv2.imencode(".png", img)
            )
        return success, buffer

    @staticmethod
    async def draw_contour(image: np.ndarray, contour: bytes) -> np.ndarray:
        """
        Draw a contour on an image.

        Parameters:
        - image: Image on which to draw the contour.
        - contour: Contour data in bytes.

        Returns:
        - Image with the contour drawn on it.
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            contour = pickle.loads(contour)
            image = await loop.run_in_executor(
                executor,
                lambda: cv2.drawContours(image, contour, -1, (0, 255, 0), 1),
            )
        return image

    @staticmethod
    async def draw_scale_bar_with_centered_text(image_ph) -> np.ndarray:
        """
        Draws a 5 um white scale bar on the lower right corner of the image with "5 um" text centered under it.
        Assumes 1 pixel = 0.0625 um.

        Parameters:
        - image_ph: Input image on which the scale bar and text will be drawn.

        Returns:
        - Modified image with the scale bar and text.
        """
        pixels_per_um = 1 / 0.0625
        scale_bar_um = 5

        scale_bar_length_px = int(scale_bar_um * pixels_per_um)

        scale_bar_thickness = 2
        scale_bar_color = (255, 255, 255)

        margin = 20
        x1 = image_ph.shape[1] - margin - scale_bar_length_px
        y1 = image_ph.shape[0] - margin
        x2 = x1 + scale_bar_length_px
        y2 = y1 + scale_bar_thickness

        cv2.rectangle(
            image_ph, (x1, y1), (x2, y2), scale_bar_color, thickness=cv2.FILLED
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{scale_bar_um} um"
        text_scale = 0.4
        text_thickness = 0
        text_color = (255, 255, 255)

        text_size = cv2.getTextSize(text, font, text_scale, text_thickness)[0]
        text_x = x1 + (scale_bar_length_px - text_size[0]) // 2
        text_y = y2 + text_size[1] + 5

        cv2.putText(
            image_ph,
            text,
            (text_x, text_y),
            font,
            text_scale,
            text_color,
            text_thickness,
        )

        return image_ph

    @staticmethod
    async def async_eig(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            eigenvalues, eigenvectors = await loop.run_in_executor(executor, eig, Sigma)
        return eigenvalues, eigenvectors

    @staticmethod
    async def async_pickle_loads(data: bytes) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            result = await loop.run_in_executor(executor, pickle.loads, data)
        return result

    @staticmethod
    async def calculate_volume_and_widths(
        u1_adj: list[float], u2_adj: list[float], split_num: int, deltaL: float
    ) -> tuple[float, list[float]]:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            result = await loop.run_in_executor(
                executor,
                SyncChores.calculate_volume_and_widths,
                u1_adj,
                u2_adj,
                split_num,
                deltaL,
            )
        return result

    @staticmethod
    async def poly_fit(U: list[list[float]], degree: int = 1) -> list[float]:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            result = await loop.run_in_executor(
                executor, SyncChores.poly_fit, U, degree
            )
        return result

    @staticmethod
    async def morpho_analysis(
        contour: bytes, polyfit_degree: int | None = None
    ) -> np.ndarray:
        img_size = 100
        contour_unpickled = await AsyncChores.async_pickle_loads(contour)
        contour = np.array([[j, i] for i, j in [i[0] for i in contour_unpickled]])

        X = np.array(
            [
                [i[1] for i in contour],
                [i[0] for i in contour],
            ]
        )

        # 基底変換関数を呼び出して必要な変数を取得
        u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U = (
            SyncChores.basis_conversion(contour, X, img_size / 2, img_size / 2, contour)
        )

        # 中心座標(u1_c, u2_c)が(0,0)になるように補正
        u1_adj = u1 - u1_c
        u2_adj = u2 - u2_c
        cell_length = max(u1_adj) - min(u1_adj)
        deltaL = cell_length / 20

        if polyfit_degree is None or polyfit_degree == 1:
            area = cv2.contourArea(np.array(contour))
            volume, widths = await AsyncChores.calculate_volume_and_widths(
                u1_adj, [abs(i) for i in u2_adj], 20, deltaL
            )
            width = sum(sorted(widths, reverse=True)[:3]) * 2 / 3
        else:
            theta = await AsyncChores.poly_fit(
                np.array([u2, u1]).T, degree=polyfit_degree
            )
            y = np.polyval(theta, np.linspace(min(u1_adj), max(u1_adj), 1000))
            cell_length = SyncChores.calc_arc_length(theta, min(u1), max(u1))
            area = cv2.contourArea(np.array(contour))

            raw_points = []
            for i, j in zip(u1, u2):
                min_distance, min_point = SyncChores.find_minimum_distance_and_point(
                    theta, i, j
                )
                arc_length = SyncChores.calc_arc_length(theta, min(u1), i)
                raw_points.append([arc_length, min_distance])
            raw_points.sort(key=lambda x: x[0])

            volume = 0
            widths = []
            y_mean = 0
            for i in range(20):
                x_0 = i * deltaL
                x_1 = (i + 1) * deltaL
                points = [p for p in raw_points if x_0 <= p[0] <= x_1]
                if points:
                    y_mean = sum([i[1] for i in points]) / len(points)
                volume += y_mean**2 * np.pi * deltaL
                widths.append(y_mean)
            width = sum(sorted(widths, reverse=True)[:3]) * 2 / 3
        return CellMorhology(
            cell_id="",
            area=area,
            volume=volume,
            width=width,
            length=cell_length,
            contour_raw=contour,
            converted_contour=contour_U,
            center_raw=[img_size / 2, img_size / 2],
            center_converted=[u1_c, u2_c],
            coefficents=theta,
        )


class CellCrudBase:
    def __init__(self, db_name: str) -> None:
        self.db_name: str = db_name

    @staticmethod
    async def parse_image(
        data: bytes,
        contour: bytes | None = None,
        scale_bar: bool = False,
        brightness_factor: float = 1.0,
    ) -> StreamingResponse:
        """
        Parse the image data and return a StreamingResponse object.

        Parameters:
        - data: Image data in bytes.
        - contour: Contour data in bytes.
        - scale_bar: Whether to draw a scale bar on the image.

        Returns:
        - StreamingResponse object with the image data.
        """
        img = await AsyncChores.async_imdecode(data)
        if contour:
            img = await AsyncChores.draw_contour(img, contour)
        if brightness_factor != 1.0:
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        if scale_bar:
            img = await AsyncChores.draw_scale_bar_with_centered_text(img)
        _, buffer = await AsyncChores.async_cv2_imencode(img)
        buffer_io = io.BytesIO(buffer)
        return StreamingResponse(buffer_io, media_type="image/png")

    async def read_cell_ids(self, label: str | None = None) -> list[CellId]:
        """
        Read all cell IDs from the database.

        Parameters:
        - label: Optional label to filter cells by.

        Returns:
        - List of CellId objects.
        """
        stmt = select(Cell)
        if label:
            stmt = stmt.where(Cell.manual_label == label)
        async for session in get_session(dbname=self.db_name):
            result = await session.execute(stmt)
            cells: list[Cell] = result.scalars().all()
        await session.close()
        return [CellId(cell_id=cell.cell_id) for cell in cells]

    async def read_cell_ids_count(self, label: str | None = None) -> int:
        """
        Read the number of cell IDs from the database.

        Parameters:
        - label: Optional label to filter cells by.

        Returns:
        - Number of cell IDs with respect to the label.
        """
        return len(await self.read_cell_ids(self.db_name, label))

    async def read_cell(self, cell_id: str) -> Cell:
        """
        Read a cell by its ID.

        Parameters:
        - cell_id: ID of the cell to fetch.

        Returns:
        - Cell object with the given ID.
        """
        stmt = select(Cell).where(Cell.cell_id == cell_id)
        async for session in get_session(dbname=self.db_name):
            result = await session.execute(stmt)
            cell: Cell = result.scalars().first()
        await session.close()
        if cell is None:
            raise CellNotFoundError(cell_id, "Cell with given ID does not exist")
        return cell

    async def get_cell_ph(
        self, cell_id: str, draw_contour: bool = False, draw_scale_bar: bool = False
    ) -> StreamingResponse:
        """
        Get the phase contrast images for a cell by its ID.

        Parameters:
        - cell_id: ID of the cell to fetch images for.
        - draw_contour: Whether to draw the contour on the image.
        - draw_scale_bar: Whether to draw the scale bar on the image.

        Returns:
        - StreamingResponse object with the phase contrast image data.
        """
        cell = await self.read_cell(cell_id)
        if draw_contour:
            return await self.parse_image(
                data=cell.img_ph, contour=cell.contour, scale_bar=draw_scale_bar
            )
        return await self.parse_image(data=cell.img_ph, scale_bar=draw_scale_bar)

    async def get_cell_fluo(
        self,
        cell_id: str,
        draw_contour: bool = False,
        draw_scale_bar: bool = False,
        brightness_factor: float = 1.0,
    ) -> StreamingResponse:
        """
        Get the fluorescence images for a cell by its ID.

        Parameters:
        - cell_id: ID of the cell to fetch images for.
        - draw_contour: Whether to draw the contour on the image.
        - draw_scale_bar: Whether to draw the scale bar on the image.
        - brightness_factor: Brightness factor to apply to the image.

        Returns:
        - StreamingResponse object with the fluorescence image data.
        """
        cell = await self.read_cell(cell_id)
        if draw_contour:
            return await self.parse_image(
                data=cell.img_fluo1,
                contour=cell.contour,
                scale_bar=draw_scale_bar,
                brightness_factor=brightness_factor,
            )
        return await self.parse_image(
            data=cell.img_fluo1,
            scale_bar=draw_scale_bar,
            brightness_factor=brightness_factor,
        )

    async def get_cell_contour(self, cell_id: str) -> list[list[float]]:
        cell = await self.read_cell(cell_id)
        return await AsyncChores.async_pickle_loads(cell.contour)

    async def morpho_analysis(
        self, cell_id: str, polyfit_degree: int | None = None
    ) -> tuple[float, float, float, float, dict]:
        """
        Perform morphological analysis on a cell by its ID.

        Parameters:
        - cell_id: ID of the cell to perform analysis on.
        - polyfit_degree: Degree of the polynomial to fit the cell boundary.

        Returns:
        - Tuple containing the area, volume, width, and cell length of the cell.
        """
        cell = await self.read_cell(cell_id)
        return await AsyncChores.morpho_analysis(cell.contour, polyfit_degree)
