from schemas import CellDB, CellDBAll, BasicCellInfo, CellStatsv2, CellId
from database import get_session, Cell, Cell2
from sqlalchemy.future import select
from Exceptions import CellNotFoundError


class CellCrudBase:
    def __init__(self) -> None:
        self.db_name = "test_database.db"

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

    async def read_cell(self, cell_id: str) -> CellDBAll:
        stmt = select(Cell)
        stmt = stmt.where(Cell.cell_id == cell_id)
        async for session in get_session(dbname=self.db_name):
            result = await session.execute(stmt)
            cell: Cell = result.scalars().first()
        await session.close()
        return CellDBAll(
            cell_id=cell.cell_id,
            label_experiment=cell.label_experiment,
            manual_label=cell.manual_label,
            perimeter=round(cell.perimeter, 2),
            area=cell.area,
            img_ph=bytes(cell.img_ph),
            img_fluo1=bytes(cell.img_fluo1) if cell.img_fluo1 else None,
            img_fluo2=bytes(cell.img_fluo2) if cell.img_fluo2 else None,
            contour=bytes(cell.contour),
            center_x=cell.center_x,
            center_y=cell.center_y,
        )

    async def get_cell_ph(self, cell_id: str) -> bytes:
        async for session in get_session(self.db_name):
            result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
            cell: Cell = result.scalars().first()
            if cell is None:
                raise CellNotFoundError(
                    cell_id, "Cell with given ID does not exist for fluorescence image"
                )
        await session.close()
        return cell.img_ph

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
