from schemas import CellDB, CellDBAll, BasicCellInfo, CellStatsv2, CellId
from database import get_session, Cell, Cell2
from sqlalchemy.future import select
from Exceptions import CellNotFoundError
import cv2
import numpy as np
from fastapi.responses import StreamingResponse
import aiofiles


class CellCrudBase:
    def __init__(self, db_name: str) -> None:
        self.db_name: str = db_name

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

    async def get_cell_ph(self, cell_id: str) -> bytes:
        cell: bytes = await self.get_cell_ph(cell_id)
        image_ph = cv2.imdecode(np.frombuffer(cell, dtype=np.uint8), cv2.IMREAD_COLOR)
        _, buffer = cv2.imencode(".png", image_ph)
        async with aiofiles.open("temp.png", "wb") as afp:
            await afp.write(buffer)
        return StreamingResponse(open("temp.png", "rb"), media_type="image/png")

    async def get_cell_fluo(self, cell_id: str) -> bytes:
        async for session in get_session(self.db_name):
            result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
            cell: Cell = result.scalars().first()
            if cell is None:
                raise CellNotFoundError(
                    cell_id, "Cell with given ID does not exist for fluorescence image"
                )
        await session.close()
        return cell.img_fluo1

    async def get_cell_contour(self, cell_id: str) -> bytes:
        async for session in get_session(self.db_name):
            result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
            cell: Cell = result.scalars().first()
            if cell is None:
                raise CellNotFoundError(
                    cell_id, "Cell with given ID does not exist for contour"
                )
        await session.close()
        return cell.contour
