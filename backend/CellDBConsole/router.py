from fastapi import APIRouter
from CellDBConsole.crud import CellCrudBase

router_cell = APIRouter(prefix="/cells", tags=["cells"])
# define a global var called db_name
db_name = "test_database.db"


@router_cell.get("/cells")
async def read_cell_ids(db_bame: str = "test_database.db", label: str | None = None):
    return await CellCrudBase(db_name=db_bame).read_cell_ids(label=label)


@router_cell.get("/cells/{cell_id}/ph_image")
async def get_cell_ph(
    cell_id: str, db_bame: str = "test_database.db", draw_contour: bool = False
):
    return await CellCrudBase(db_name=db_bame).get_cell_ph(
        cell_id=cell_id, draw_contour=draw_contour
    )
