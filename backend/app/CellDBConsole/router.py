from fastapi import APIRouter
from CellDBConsole.crud import CellCrudBase, AsyncChores
from CellDBConsole.schemas import CellMorhology
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Literal
import os
from fastapi import UploadFile

router_cell = APIRouter(prefix="/cells", tags=["cells"])
router_database = APIRouter(prefix="/databases", tags=["databases"])
# define a global var called db_name
db_name = "test_database.db"


@router_cell.get("/database/healtcheck")
async def db_healthcheck():
    return await CellCrudBase(db_name="test_database.db").get_cell_ph(
        cell_id="F0C1", draw_contour=False, draw_scale_bar=False
    )


@router_cell.get("/{db_name}/{label}")
async def read_cell_ids(db_name: str, label: str):
    if label == "1000":
        label = "N/A"
    return await CellCrudBase(db_name=db_name).read_cell_ids(label=label)


@router_cell.get("/{cell_id}/{db_name}/{draw_contour}/{draw_scale_bar}/ph_image")
async def get_cell_ph(
    cell_id: str,
    db_name: str,
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
):
    return await CellCrudBase(db_name=db_name).get_cell_ph(
        cell_id=cell_id, draw_contour=draw_contour, draw_scale_bar=draw_scale_bar
    )


@router_cell.get("/{cell_id}/{db_name}/{draw_contour}/{draw_scale_bar}/fluo_image")
async def get_cell_fluo(
    cell_id: str,
    db_name: str,
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
    brightness_factor: float = 1.0,
):
    return await CellCrudBase(db_name=db_name).get_cell_fluo(
        cell_id=cell_id,
        draw_contour=draw_contour,
        draw_scale_bar=draw_scale_bar,
        brightness_factor=brightness_factor,
    )


@router_cell.get("/{cell_id}/contour/{contour_type}")
async def get_cell_contour(
    cell_id: str,
    db_name: str,
    contour_type: Literal["raw", "converted"] = "raw",
    polyfit_degree: int = 3,
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
async def get_cell_morphology(cell_id: str, db_name: str, polyfit_degree: int = 3):
    cell_morphology: CellMorhology = await CellCrudBase(
        db_name=db_name
    ).morpho_analysis(cell_id=cell_id, polyfit_degree=polyfit_degree)
    return cell_morphology


@router_cell.get("/{cell_id}/{db_name}/replot", response_class=StreamingResponse)
async def replot_cell(cell_id: str, db_name: str, degree: int = 3):
    return await CellCrudBase(db_name=db_name).replot(cell_id=cell_id, degree=degree)


@router_cell.get("/{cell_id}/{db_name}/path", response_class=StreamingResponse)
async def get_cell_path(cell_id: str, db_name: str, degree: int = 3):
    return await CellCrudBase(db_name=db_name).find_path(cell_id=cell_id, degree=degree)


@router_database.post("/upload")
async def upload_database(file: UploadFile = UploadFile(...)):
    await AsyncChores().upload_file_chunked(file)
    return file.filename


@router_database.get("/")
async def get_databases():
    return await AsyncChores().get_database_names()
