import os
from typing import Literal

from fastapi import APIRouter, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from CellDBConsole.crud import AsyncChores, CellCrudBase
from CellDBConsole.schemas import CellMorhology, MetadataUpdateRequest

router_cell = APIRouter(prefix="/cells", tags=["cells"])
router_database = APIRouter(prefix="/databases", tags=["databases"])


@router_cell.get("/database/healthcheck")
async def db_healthcheck():
    return await CellCrudBase(db_name="test_database.db").get_cell_ph(
        cell_id="F0C5", draw_contour=False, draw_scale_bar=False
    )


@router_cell.get("/database/healthcheck/fluo")
async def db_healthcheck_fluo():
    return await CellCrudBase(db_name="test_database.db").get_cell_fluo(
        cell_id="F0C5", draw_contour=False, draw_scale_bar=False
    )


@router_cell.get("/database/healthcheck/replot")
async def db_healthcheck_replot():
    return StreamingResponse(
        await CellCrudBase(db_name="test_database.db").replot(cell_id="F0C5", degree=3),
        media_type="image/png",
    )


@router_cell.get("/database/healthcheck/3d")
async def db_healthcheck_3d():
    return StreamingResponse(
        await CellCrudBase(db_name="test_database.db").get_cloud_points(
            cell_id="F0C5", mode="fluo", angle=0
        ),
        media_type="image/png",
    )


@router_cell.get("/{db_name}/{cell_id}/label")
async def read_cell_label(db_name: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).read_cell_label(cell_id=cell_id)


@router_cell.patch("/{db_name}/{cell_id}/{label}")
async def update_cell_label(db_name: str, cell_id: str, label: str):
    if "-uploaded" not in db_name:
        raise HTTPException(
            status_code=400,
            detail="Please provide the name of the uploaded database.",
        )

    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).update_label(
        cell_id=cell_id, label=label
    )


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
    if "-single_layer" in db_name:
        raise HTTPException(
            status_code=404,
            detail="Fluo does not exist in single layer databases. Please use the ph endpoint.",
        )
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
    ).get_all_mean_normalized_fluo_intensities(
        label=label, cell_id=cell_id, y_label="Mean Normalized Fluorescence Intensity"
    )
    return ret


@router_cell.get("/{db_name}/{label}/{cell_id}/median_fluo_intensities")
async def get_median_fluo_intensities(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    ret: list[float] = await CellCrudBase(
        db_name=db_name
    ).get_all_median_normalized_fluo_intensities(
        label=label, cell_id=cell_id, y_label="Median Normalized Fluorescence Intensity"
    )
    return ret


@router_cell.get(
    "/{db_name}/{label}/median_fluo_intensities/csv", response_class=StreamingResponse
)
async def get_median_fluo_intensities_csv(db_name: str, label: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(
        db_name=db_name
    ).get_all_median_normalized_fluo_intensities_csv(label=label)


@router_cell.get(
    "/{db_name}/{label}/mean_fluo_intensities/csv", response_class=StreamingResponse
)
async def get_mean_fluo_intensities_csv(db_name: str, label: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(
        db_name=db_name
    ).get_all_mean_normalized_fluo_intensities_csv(label=label)


@router_cell.get(
    "/{db_name}/{label}/{cell_id}/heatmap", response_class=StreamingResponse
)
async def get_heatmap(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).heatmap_path(cell_id=cell_id, degree=4)


@router_cell.get(
    "/{db_name}/{label}/{cell_id}/heatmap_all_abs", response_class=StreamingResponse
)
async def get_heatmap_all_abs(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).heatmap_all_abs(label=label)


@router_cell.get(
    "/{db_name}/{label}/{cell_id}/heatmap/csv", response_class=StreamingResponse
)
async def get_heatmap_csv(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_peak_path_csv(cell_id=cell_id)


@router_cell.get(
    "/{db_name}/{label}/{cell_id}/heatmap/bulk/csv", response_class=StreamingResponse
)
async def get_heatmap_bulk_csv(db_name: str, label: str = "1"):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_peak_paths_csv(degree=4, label=label)


@router_cell.get(
    "/{db_name}/{label}/{cell_id}/paths_plot", response_class=StreamingResponse
)
async def get_paths_plot(db_name: str, label: str = 1):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).plot_peak_paths(label=label)


@router_cell.get("/{db_name}/{cell_id}/3d")
async def get_3d_plot(db_name: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    image_buf = await CellCrudBase(db_name=db_name).get_cloud_points(cell_id=cell_id)
    return StreamingResponse(image_buf, media_type="image/png")


@router_cell.get("/{db_name}/{cell_id}/3d-ph")
async def get_3d_plot(db_name: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    image_buf = await CellCrudBase(db_name=db_name).get_cloud_points(
        cell_id=cell_id, mode="ph"
    )
    return StreamingResponse(image_buf, media_type="image/png")


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


@router_database.patch("/{db_name}")
async def update_database_to_label_completed(db_name: str):
    file_path = os.path.join("databases", db_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    await CellCrudBase(db_name).rename_database_to_completed()
    return JSONResponse(content={"message": "Database updated"})


@router_database.get("/{db_name}")
async def check_if_database_updated_once(db_name: str):
    return await CellCrudBase(db_name).check_if_database_updated()


@router_database.get("/download-completed/{db_name}")
async def download_db(db_name: str):
    file_path = os.path.join("databases", db_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    if "-completed" not in db_name:
        raise HTTPException(
            status_code=400,
            detail="Please provide the name of the completed database.",
        )
    return FileResponse(
        f"databases/{db_name.split('/')[-1]}",
    )


@router_database.get("/{db_name}/combined_images")
async def get_cell_images_combined(
    db_name: str,
    label: Literal["N/A", "1", "2", "3"] = "1",
    mode: Literal["fluo", "ph", "ph_contour", "fluo_contour"] = "fluo",
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_cell_images_combined(
        label=label, mode=mode
    )


@router_database.get("/{db_name}/metadata")
async def get_metadata(db_name: str):
    return await CellCrudBase(db_name=db_name).get_metadata()


@router_database.patch("/{db_name}/update-metadata")
async def update_label_experiment(db_name: str, request: MetadataUpdateRequest):
    if db_name == "test_database.db":
        raise HTTPException(
            status_code=400, detail="Cannot update metadata for the test database."
        )
    return await CellCrudBase(db_name=db_name).update_all_cells_metadata(
        request.metadata
    )
