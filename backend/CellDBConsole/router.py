from fastapi import APIRouter
from CellDBConsole.crud import CellCrudBase
from CellDBConsole.schemas import CellMorhology
from fastapi.responses import JSONResponse
from typing import Literal

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
    contour_type: Literal["raw", "converted"] = "raw",
    db_name: str = "test_database.db",
):
    # if contour_type == "raw":
    #     contour = [
    #         [int(j), int(i)]
    #         for i, j in [
    #             i[0]
    #             for i in await CellCrudBase(db_name=db_name).get_cell_contour(
    #                 cell_id=cell_id
    #             )
    #         ]
    #     ]
    # else:
    #     contour = []

    cell_morphology: CellMorhology = await CellCrudBase(
        db_name=db_name
    ).morpho_analysis(cell_id=cell_id, polyfit_degree=3)

    if contour_type == "raw":
        contour = cell_morphology.contour_raw
        return JSONResponse(content={"contour": contour})
    else:
        contour = cell_morphology.converted_contour
        return JSONResponse(content={"contour": contour})


@router_cell.get("/{cell_id}/morphology", response_model=CellMorhology)
async def get_cell_morphology(cell_id: str, db_bame: str = "test_database.db"):
    return await CellCrudBase(db_name=db_bame).morpho_analysis(
        cell_id=cell_id, polyfit_degree=3
    )
