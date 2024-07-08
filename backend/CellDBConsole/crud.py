from CellDBConsole.schemas import CellId
from database import get_session, Cell
from sqlalchemy.future import select
from exceptions import CellNotFoundError
import cv2
import numpy as np
from numpy.linalg import eig
from fastapi.responses import StreamingResponse
import io
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from __future__ import annotations


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
    async def async_cv2_imencode(img):
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
    async def draw_scale_bar_with_centered_text(image_ph):
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
    async def async_eig(Sigma: np.ndarray):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            eigenvalues, eigenvectors = await loop.run_in_executor(executor, eig, Sigma)
        return eigenvalues, eigenvectors

    async def basis_conversion(
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
        img = await CellCrudBase.async_imdecode(data)
        if contour:
            cv2.drawContours(img, pickle.loads(contour), -1, (0, 255, 0), 1)
        if brightness_factor != 1.0:
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        if scale_bar:
            img = await CellCrudBase.draw_scale_bar_with_centered_text(img)
        _, buffer = await CellCrudBase.async_cv2_imencode(img)
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
            return await self.parse_image(cell.img_ph, cell.contour, draw_scale_bar)
        return await self.parse_image(cell.img_ph, scale_bar=draw_scale_bar)

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
                cell.img_fluo1, cell.contour, draw_scale_bar, brightness_factor
            )
        return await self.parse_image(
            cell.img_fluo1,
            scale_bar=draw_scale_bar,
            brightness_factor=brightness_factor,
        )

    async def get_cell_contour(self, cell_id: str) -> bytes:
        cell = await self.read_cell(cell_id)
        return cell.contour


class CellAnalysisCrud:
    def __init__(self) -> None:
        pass


async def replot_blocking_operations(
    gray: np.ndarray,
    image_fluo: np.ndarray,
    contour_raw: bytes,
):
    class Point:
        def __init__(self, u1: float, G: float):
            self.u1: float = u1
            self.G: float = G

        def __gt__(self, other: Point):
            return self.u1 > other.u1

    contour = [[j, i] for i, j in [i[0] for i in pickle.loads(contour_raw)]]
    coords_inside_cell_1, points_inside_cell_1, projected_points = [], [], []
    for i in range(image_fluo.shape[1]):
        for j in range(image_fluo.shape[0]):
            if cv2.pointPolygonTest(pickle.loads(contour_raw), (i, j), False) >= 0:
                coords_inside_cell_1.append([i, j])
                points_inside_cell_1.append(gray[j][i])
    X = np.array(
        [
            [i[1] for i in coords_inside_cell_1],
            [i[0] for i in coords_inside_cell_1],
        ]
    )
    (
        u1,
        u2,
        u1_contour,
        u2_contour,
        min_u1,
        max_u1,
        u1_c,
        u2_c,
        U,
        contour_U,
    ) = basis_conversion(
        contour,
        X,
        image_fluo.shape[0] // 2,
        image_fluo.shape[1] // 2,
        coords_inside_cell_1,
    )
    min_u1, max_u1 = min(u1), max(u1)
    fig = plt.figure(figsize=[6, 6])
    cmap = plt.get_cmap("inferno")
    x = np.linspace(0, 100, 1000)
    max_points = max(points_inside_cell_1)
    plt.scatter(
        u1,
        u2,
        c=[i / max_points for i in points_inside_cell_1],
        s=10,
        cmap=cmap,
    )
    # plt.scatter(u1_contour, u2_contour, s=10, color="lime")
    W = np.array([[i**4, i**3, i**2, i, 1] for i in [i[1] for i in U]])
    f = np.array([i[0] for i in U])
    theta = inv(W.transpose() @ W) @ W.transpose() @ f
    x = np.linspace(min_u1, max_u1, 1000)
    y = [
        theta[0] * i**4 + theta[1] * i**3 + theta[2] * i**2 + theta[3] * i + theta[4]
        for i in x
    ]
    plt.plot(x, y, color="blue", linewidth=1)
    plt.scatter(
        min_u1,
        theta[0] * min_u1**4
        + theta[1] * min_u1**3
        + theta[2] * min_u1**2
        + theta[3] * min_u1
        + theta[4],
        s=100,
        color="red",
        zorder=100,
        marker="x",
    )
    plt.scatter(
        max_u1,
        theta[0] * max_u1**4
        + theta[1] * max_u1**3
        + theta[2] * max_u1**2
        + theta[3] * max_u1
        + theta[4],
        s=100,
        color="red",
        zorder=100,
        marker="x",
    )

    plt.xlabel("u1")
    plt.ylabel("u2")
    plt.axis("equal")
    plt.xlim(min_u1 - 80, max_u1 + 80)
    plt.ylim(u2_c - 80, u2_c + 80)

    # Y軸の範囲を取得
    ymin, ymax = plt.ylim()
    y_pos = ymin + 0.2 * (ymax - ymin)
    y_pos_text = ymax - 0.15 * (ymax - ymin)
    plt.text(
        u1_c,
        y_pos_text,
        s=f"",
        color="red",
        ha="center",
        va="top",
    )
    for u, g in zip(u1, points_inside_cell_1):
        point = Point(u, g)
        projected_points.append(point)
    sorted_projected_points = sorted(projected_points)
    # add second axis
    ax2 = plt.twinx()
    ax2.grid(False)
    ax2.set_xlabel("u1")
    ax2.set_ylabel("Brightness")
    ax2.set_ylim(0, 900)
    ax2.set_xlim(min_u1 - 40, max_u1 + 40)
    ax2.scatter(
        [i.u1 for i in sorted_projected_points],
        [i.G for i in sorted_projected_points],
        color="lime",
        s=1,
    )
    await save_fig_async(fig, "temp_replot.png")
    plt.close()
