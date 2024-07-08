from schemas import CellDB, CellDBAll, BasicCellInfo, CellStatsv2, CellId
from database import get_session, Cell, Cell2
from sqlalchemy.future import select
from Exceptions import CellNotFoundError


class CellCrudBase:
    async def read_cell_ids(
        self, dbname: str, label: str | None = None
    ) -> list[CellId]:
        stmt = select(Cell)
        if label:
            stmt = stmt.where(Cell.manual_label == label)
        async for session in get_session(dbname=dbname):
            result = await session.execute(stmt)
            cells: list[Cell] = result.scalars().all()
        await session.close()
        return [CellId(cell_id=cell.cell_id) for cell in cells]

    async def read_cell(self, dbname: str, cell_id: str) -> CellDBAll:
        stmt = select(Cell)
        stmt = stmt.where(Cell.cell_id == cell_id)
        async for session in get_session(dbname=dbname):
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

    async def count_cells_with_label(self, db_name: str, label: str = "1") -> int:
        async for session in get_session(db_name):
            result = await session.execute(
                select(Cell).where(Cell.manual_label == label)
            )
            cells = result.scalars().all()
        await session.close()
        return len(cells)

    async def get_cell_ph(self, db_name: str, cell_id: str) -> bytes:
        async for session in get_session(db_name):
            result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
            cell: Cell = result.scalars().first()
            if cell is None:
                raise CellNotFoundError(
                    cell_id, "Cell with given ID does not exist for fluorescence image"
                )
        await session.close()
        return cell.img_ph

    async def get_cell_fluo(self, db_name: str, cell_id: str) -> bytes:
        async for session in get_session(db_name):
            result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
            cell: Cell = result.scalars().first()
            if cell is None:
                raise CellNotFoundError(
                    cell_id, "Cell with given ID does not exist for fluorescence image"
                )
        await session.close()
        return cell.img_fluo1

    async def get_cell_contour(self, db_name: str, cell_id: str) -> bytes:
        async for session in get_session(db_name):
            result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
            cell: Cell = result.scalars().first()
            if cell is None:
                raise CellNotFoundError(
                    cell_id, "Cell with given ID does not exist for contour"
                )
        await session.close()
        return cell.contour


async def get_cell_ids(dbname: str) -> list[CellId]:
    async for session in get_session(dbname=dbname):
        result = await session.execute(select(Cell))
        cells: list[Cell] = result.scalars().all()
    await session.close()
    return [CellId(cell_id=cell.cell_id) for cell in cells]


async def get_cell(dbname: str, cell_id: str) -> CellDBAll:
    async for session in get_session(dbname=dbname):
        result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
        cell: Cell = result.scalars().first()
    await session.close()
    return CellDBAll(
        cell_id=cell.cell_id,
        label_experiment=cell.label_experiment,
        manual_label=cell.manual_label,
        perimeter=round(cell.perimeter, 2),
        area=cell.area,
        img_ph=bytes(cell.img_ph),
        img_fluo1=bytes(cell.img_fluo1),
        img_fluo2=None,
        contour=bytes(cell.contour),
        center_x=cell.center_x,
        center_y=cell.center_y,
    )


async def get_cells_with_label(dbname: str, label: str = "1") -> list[CellDB]:
    async for session in get_session(dbname=dbname):
        result = await session.execute(select(Cell).where(Cell.manual_label == label))
        cells: list[Cell] = result.scalars().all()
    await session.close()
    return [
        CellDB(
            cell_id=cell.cell_id,
            label_experiment=cell.label_experiment,
            manual_label=cell.manual_label,
            perimeter=round(cell.perimeter, 2),
            area=cell.area,
        )
        for cell in cells
    ]


async def count_cells_with_label(db_name: str, label: str = "1") -> int:
    async for session in get_session(db_name):
        result = await session.execute(select(Cell).where(Cell.manual_label == label))
        cells = result.scalars().all()
    await session.close()
    return len(cells)


async def get_cell_ph(db_name: str, cell_id: str) -> bytes:
    async for session in get_session(db_name):
        result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
        cell: Cell = result.scalars().first()
    await session.close()
    return cell.img_ph


async def get_cell_fluo(db_name: str, cell_id: str) -> bytes:
    async for session in get_session(db_name):
        result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
        cell: Cell = result.scalars().first()
        if cell is None:
            raise CellNotFoundError(
                cell_id, "Cell with given ID does not exist for fluorescence image"
            )
    await session.close()
    return cell.img_fluo1


async def get_cell_contour(db_name: str, cell_id: str) -> bytes:
    async for session in get_session(db_name):
        result = await session.execute(select(Cell).where(Cell.cell_id == cell_id))
        cell: Cell = result.scalars().first()
        if cell is None:
            raise CellNotFoundError(
                cell_id, "Cell with given ID does not exist for contour"
            )
    await session.close()
    return cell.contour


async def get_cell_stats_v2(db_name: str) -> list[CellStatsv2]:
    async for session in get_session(db_name):
        result = await session.execute(select(Cell2))
        cells = result.scalars().all()
    await session.close()
    return [
        CellStatsv2(
            cell_id=cell.cell_id,
            center_x=cell.center_x,
            center_y=cell.center_y,
            basic_cell_info=BasicCellInfo(
                cell_id=cell.cell_id,
                label_experiment=cell.label_experiment,
                manual_label=cell.manual_label,
                perimeter=cell.perimeter,
                area=cell.area,
            ),
            ph_max_brightness=cell.ph_max_brightness,
            ph_min_brightness=cell.ph_min_brightness,
            ph_mean_brightness_raw=cell.ph_mean_brightness_raw,
            ph_mean_brightness_normalized=cell.ph_mean_brightness_normalized,
            ph_median_brightness_raw=cell.ph_median_brightness_raw,
            ph_median_brightness_normalized=cell.ph_median_brightness_normalized,
            max_brightness=cell.max_brightness,
            min_brightness=cell.min_brightness,
            mean_brightness_raw=cell.mean_brightness_raw,
            mean_brightness_normalized=cell.mean_brightness_normalized,
            median_brightness_raw=cell.median_brightness_raw,
            median_brightness_normalized=cell.median_brightness_normalized,
        )
        for cell in cells
    ]
