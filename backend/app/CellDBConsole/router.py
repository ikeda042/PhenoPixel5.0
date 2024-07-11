from fastapi import APIRouter
from CellDBConsole.crud import CellCrudBase, AsyncChores
from CellDBConsole.schemas import CellMorhology
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Literal
import os
from fastapi import UploadFile

router_cell = APIRouter(prefix="/cells", tags=["cells"])
router_database = APIRouter(prefix="/databases", tags=["databases"])


@router_cell.get("/database/healtcheck")
async def db_healthcheck():
    return await CellCrudBase(db_name="test_database.db").get_cell_ph(
        cell_id="F0C1", draw_contour=False, draw_scale_bar=False
    )


@router_cell.get("/{db_name}/{label}")
async def read_cell_ids(db_name: str, label: str):
    await AsyncChores().validate_database_name(db_name)
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
    await AsyncChores().validate_database_name(db_name)
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
    await AsyncChores().validate_database_name(db_name)
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
    await AsyncChores().validate_database_name(db_name)
    contours = await CellCrudBase(db_name=db_name).get_cell_contour_plot_data(
        cell_id=cell_id
    )
    if contour_type == "raw":
        contour = contours["raw"].tolist()
        return JSONResponse(content={"contour": contour})
    else:
        contour = contours["converted"].tolist()
        return JSONResponse(content={"contour": contour})


@router_cell.get("/{cell_id}/{db_name}/morphology", response_model=CellMorhology)
async def get_cell_morphology(cell_id: str, db_name: str, polyfit_degree: int = 3):
    await AsyncChores().validate_database_name(db_name)
    cell_morphology: CellMorhology = await CellCrudBase(
        db_name=db_name
    ).morpho_analysis(cell_id=cell_id, polyfit_degree=polyfit_degree)
    return cell_morphology


@router_cell.get("/{cell_id}/{db_name}/replot", response_class=StreamingResponse)
async def replot_cell(cell_id: str, db_name: str, degree: int = 3):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).replot(cell_id=cell_id, degree=degree)


@router_cell.get("/{cell_id}/{db_name}/path", response_class=StreamingResponse)
async def get_cell_path(cell_id: str, db_name: str, degree: int = 3):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).find_path(cell_id=cell_id, degree=degree)


@router_cell.get("/{db_name}/{label}/{cell_id}/mean_fluo_intensities")
async def get_mean_fluo_intensities(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    ret: list[float] = await CellCrudBase(
        db_name=db_name
    ).get_all_mean_normalized_fluo_intensities(label=label, cell_id=cell_id)
    return ret


@router_cell.get("/{db_name}/{label}/{cell_id}/mean_fluo_intensities")
async def get_mean_fluo_intensities(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    ret: list[float] = await CellCrudBase(
        db_name=db_name
    ).get_all_mean_normalized_fluo_intensities(label=label, cell_id=cell_id)
    return ret


@router_database.post("/upload")
async def upload_database(file: UploadFile = UploadFile(...)):
    db_name = file.filename
    if not db_name.endswith(".db"):
        return JSONResponse(content={"message": "Please upload a .db file."})
    exisisting_dbs = await AsyncChores().get_database_names()
    if db_name in exisisting_dbs:
        return JSONResponse(content={"message": f"Database {db_name} already exists."})
    await AsyncChores().upload_file_chunked(file)
    return file.filename


@router_database.get("/")
async def get_databases():
    return await AsyncChores().get_database_names()
