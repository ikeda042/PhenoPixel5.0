from CellDBConsole.schemas import CellId
from database import get_session, Cell
from sqlalchemy.future import select
from exceptions import CellNotFoundError
import cv2
import numpy as np
from fastapi.responses import StreamingResponse
import io
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor


class CellCrudBase:
    def __init__(self, db_name: str) -> None:
        self.db_name: str = db_name

    @staticmethod
    async def async_imdecode(data: bytes) -> np.ndarray:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            img = await loop.run_in_executor(
                executor, cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
            )
        return img

    @staticmethod
    async def async_cv2_imencode(img):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            success, buffer = await loop.run_in_executor(
                executor, lambda: cv2.imencode(".png", img)
            )
        return success, buffer

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
        # Conversion factor and scale bar desired length
        pixels_per_um = 1 / 0.0625  # pixels per micrometer
        scale_bar_um = 5  # scale bar length in micrometers

        # Calculate the scale bar length in pixels
        scale_bar_length_px = int(scale_bar_um * pixels_per_um)

        # Define the scale bar thickness and color
        scale_bar_thickness = 2  # in pixels
        scale_bar_color = (255, 255, 255)  # white for the scale bar

        # Determine the position for the scale bar (lower right corner)
        margin = 20  # margin from the edges in pixels, increased for text space
        x1 = image_ph.shape[1] - margin - scale_bar_length_px
        y1 = image_ph.shape[0] - margin
        x2 = x1 + scale_bar_length_px
        y2 = y1 + scale_bar_thickness

        # Draw the scale bar
        cv2.rectangle(
            image_ph, (x1, y1), (x2, y2), scale_bar_color, thickness=cv2.FILLED
        )

        # Add text "5 um" under the scale bar
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "5 um"
        text_scale = 0.4  # font scale
        text_thickness = 1  # font thickness
        text_color = (255, 255, 255)  # white for the text

        # Calculate text size to position it
        text_size = cv2.getTextSize(text, font, text_scale, text_thickness)[0]
        # Calculate the starting x-coordinate of the text to center it under the scale bar
        text_x = x1 + (scale_bar_length_px - text_size[0]) // 2
        text_y = y2 + text_size[1] + 5  # a little space below the scale bar

        # Draw the text
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

    async def read_cell_ids(self, label: str | None = None) -> list[CellId]:
        stmt = select(Cell)
        if label:
            stmt = stmt.where(Cell.manual_label == label)
        async for session in get_session(dbname=self.db_name):
            result = await session.execute(stmt)
            cells: list[Cell] = result.scalars().all()
        await session.close()
        return [CellId(cell_id=cell.cell_id) for cell in cells]

    async def read_cell_ids_count(self, label: str | None = None) -> int:
        return len(await self.read_cell_ids(self.db_name, label))

    async def read_cell(self, cell_id: str) -> Cell:
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
