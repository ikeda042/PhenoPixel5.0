from fastapi import APIRouter
from CellDBConsole.crud import CellCrudBase
from CellDBConsole.schemas import CellMorhology
from fastapi.responses import JSONResponse
from typing import Literal
from fastapi.responses import StreamingResponse

router_cell = APIRouter(prefix="/cells", tags=["cells"])


# define a global var called db_name
db_name = "test_database.db"


@router_cell.get("/")
async def read_cell_ids(db_bame: str = "test_database.db", label: str | None = None):
    return await CellCrudBase(db_name=db_bame).read_cell_ids(label=label)


@router_cell.get("/{cell_id}/ph_image")
async def get_cell_ph(
    cell_id: str,
    db_bame: str = "test_database.db",
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
):
    return await CellCrudBase(db_name=db_bame).get_cell_ph(
        cell_id=cell_id, draw_contour=draw_contour, draw_scale_bar=draw_scale_bar
    )


@router_cell.get("/{cell_id}/fluo_image")
async def get_cell_fluo(
    cell_id: str,
    db_bame: str = "test_database.db",
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
    brightness_factor: float = 1.0,
):
    return await CellCrudBase(db_name=db_bame).get_cell_fluo(
        cell_id=cell_id,
        draw_contour=draw_contour,
        draw_scale_bar=draw_scale_bar,
        brightness_factor=brightness_factor,
    )


@router_cell.get("/{cell_id}/contour/{contour_type}")
async def get_cell_contour(
    cell_id: str,
    contour_type: Literal["raw", "converted"] = "copn",
    polyfit_degree: int = 3,
    db_name: str = "test_database.db",
):
    contours = await CellCrudBase(db_name=db_name).get_cell_contour_plot_data(
        cell_id=cell_id
    )
    if contour_type == "raw":
        contour = contours["raw"].tolist()
        return JSONResponse(content={"contour": contour})
    else:
        contour = contours["converted"].tolist()
        return JSONResponse(content={"contour": contour})


@router_cell.get("/{cell_id}/morphology", response_model=CellMorhology)
async def get_cell_morphology(
    cell_id: str, polyfit_degree: int = 3, db_name: str = "test_database.db"
):
    cell_morphology: CellMorhology = await CellCrudBase(
        db_name=db_name
    ).morpho_analysis(cell_id=cell_id, polyfit_degree=polyfit_degree)
    return cell_morphology


@router_cell.get("/{cell_id}/replot", response_class=StreamingResponse)
async def replot_cell(cell_id: str, degree: int = 3, db_name: str = "test_database.db"):
    return await CellCrudBase(db_name=db_name).replot(cell_id=cell_id, degree=degree)
