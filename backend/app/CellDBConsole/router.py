import os
from typing import Literal

from fastapi import APIRouter, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import asyncio
from databases.migration import migrate

from CellDBConsole.crud import AsyncChores, CellCrudBase
from CellDBConsole.schemas import CellMorhology, MetadataUpdateRequest
from CellAI.crud import CellAiCrudBase
from OAuth2.login_manager import get_account_optional
from OAuth2.crud import UserCrud

router_cell = APIRouter(prefix="/cells", tags=["cells"])
router_database = APIRouter(prefix="/databases", tags=["databases"])


@router_cell.patch("/redetect_contour_t1/{db_name}/{cell_id}")
async def patch_cell_contour_t1(
    db_name: str,
    cell_id: str,
    model: Literal["T1"] = "T1",
):
    """
    PATCH: あるDBのあるcell_idの輪郭データを、predict_contour_draw() で算出したものに置き換える
    """
    predicted_contour = await CellAiCrudBase(
        db_name, model_path=model
    ).predict_contour_draw(cell_id)

    await CellCrudBase(db_name).update_contour(cell_id, predicted_contour)

    return {
        "status": "success",
        "updated_cell_id": cell_id,
        "predicted_contour": predicted_contour,
    }


@router_cell.patch("/redetect_contour_canny/{db_name}/{cell_id}")
async def patch_cell_contour_canny(
    db_name: str,
    cell_id: str,
    canny_thresh2: int = 100,
):
    """
    PATCH: あるDBのあるcell_idの輪郭データを、predict_contour_canny() で算出したものに置き換える
    """
    canny_contour = await CellCrudBase(db_name).get_contour_canny_draw(
        cell_id, canny_thresh2
    )
    await CellCrudBase(db_name).update_contour(cell_id, canny_contour)

    return {
        "status": "success",
        "updated_cell_id": cell_id,
        "predicted_contour": canny_contour,
    }


@router_cell.patch("/elastic_contour/{db_name}/{cell_id}")
async def patch_cell_contour_elastic(
    db_name: str,
    cell_id: str,
    delta: int = 0,
):
    """Expand or shrink contour by delta pixels."""
    new_contour = await CellCrudBase(db_name).elastic_contour(cell_id, delta)
    await CellCrudBase(db_name).update_contour(cell_id, new_contour)
    return {
        "status": "success",
        "updated_cell_id": cell_id,
        "predicted_contour": new_contour,
    }


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


@router_cell.get("/{db_name}/{label}")
async def read_cell_ids(db_name: str, label: str):
    await AsyncChores().validate_database_name(db_name)
    if label == "1000":
        label = "N/A"
    if label == "74":
        label = None
    return await CellCrudBase(db_name=db_name).read_cell_ids(label=label)


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

    # "1000" is used on the frontend to represent "N/A" since slashes are not
    # allowed in the URL path parameters. Convert it back before updating the DB.
    if label == "1000":
        label = "N/A"

    return await CellCrudBase(db_name=db_name).update_label(
        cell_id=cell_id, label=label
    )


@router_cell.get("/{cell_id}/{db_name}/{draw_contour}/{draw_scale_bar}/ph_image")
async def get_cell_ph(
    cell_id: str,
    db_name: str,
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
    resize_factor: float = 1.0,
    contour_thickness: int = 1,
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_cell_ph(
        cell_id=cell_id,
        draw_contour=draw_contour,
        draw_scale_bar=draw_scale_bar,
        resize_factor=resize_factor,
        contour_thickness=contour_thickness,
    )


@router_cell.get("/{cell_id}/{db_name}/{draw_contour}/{draw_scale_bar}/fluo_image")
async def get_cell_fluo(
    cell_id: str,
    db_name: str,
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
    brightness_factor: float = 1.0,
    resize_factor: float = 1.0,
    contour_thickness: int = 1,
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
        resize_factor=resize_factor,
        contour_thickness=contour_thickness,
    )


@router_cell.get("/{cell_id}/{db_name}/{draw_contour}/{draw_scale_bar}/fluo2_image")
async def get_cell_fluo2(
    cell_id: str,
    db_name: str,
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
    brightness_factor: float = 1.0,
    resize_factor: float = 1.0,
    contour_thickness: int = 1,
):
    if "-single_layer" in db_name:
        raise HTTPException(
            status_code=404,
            detail="Fluo does not exist in single layer databases. Please use the ph endpoint.",
        )
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_cell_fluo2(
        cell_id=cell_id,
        draw_contour=draw_contour,
        draw_scale_bar=draw_scale_bar,
        brightness_factor=brightness_factor,
        resize_factor=resize_factor,
        contour_thickness=contour_thickness,
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
async def replot_cell(
    cell_id: str, db_name: str, degree: int = 3, channel: int = 1, dark_mode: bool = False
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).replot(
        cell_id=cell_id, degree=degree, channel=channel, dark_mode=dark_mode
    )


@router_cell.get("/{cell_id}/{db_name}/path", response_class=StreamingResponse)
async def get_cell_path(
    cell_id: str, db_name: str, degree: int = 3, channel: int = 1
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).find_path(
        cell_id=cell_id, degree=degree, channel=channel
    )


@router_cell.get("/{cell_id}/{db_name}/laplacian", response_class=StreamingResponse)
async def get_cell_laplacian(
    cell_id: str, db_name: str, channel: int = 1, brightness_factor: float = 1.0
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_laplacian(
        cell_id, channel, brightness_factor
    )


@router_cell.get("/{cell_id}/{db_name}/sobel", response_class=StreamingResponse)
async def get_cell_sobel(
    cell_id: str, db_name: str, channel: int = 1, brightness_factor: float = 1.0
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_sobel(
        cell_id, channel, brightness_factor
    )


@router_cell.get("/{cell_id}/{db_name}/hu_mask", response_class=StreamingResponse)
async def get_cell_hu_mask(cell_id: str, db_name: str, channel: int = 1):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_hu_mask(cell_id, channel)


@router_cell.get("/{cell_id}/{db_name}/map256", response_class=StreamingResponse)
async def get_cell_map256(
    cell_id: str,
    db_name: str,
    degree: int = 4,
    channel: int = 1,
    img_type: Literal["fluo", "ph"] = "fluo",
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_map256(
        cell_id, degree=degree, channel=channel, img_type=img_type
    )


@router_cell.get("/{cell_id}/{db_name}/map256_jet", response_class=StreamingResponse)
async def get_cell_map256_jet(
    cell_id: str,
    db_name: str,
    degree: int = 4,
    channel: int = 1,
    img_type: Literal["fluo", "ph"] = "fluo",
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_map256_jet(
        cell_id, degree=degree, channel=channel, img_type=img_type
    )


@router_cell.get("/{cell_id}/{db_name}/map256_clip", response_class=StreamingResponse)
async def get_cell_map256_clip(
    cell_id: str,
    db_name: str,
    degree: int = 4,
    channel: int = 1,
    img_type: Literal["fluo", "ph"] = "fluo",
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_map256_clip(
        cell_id, degree=degree, channel=channel, img_type=img_type
    )


@router_cell.get("/{db_name}/{label}/{cell_id}/mean_fluo_intensities")
async def get_mean_fluo_intensities(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    ret: StreamingResponse = await CellCrudBase(
        db_name=db_name
    ).get_all_mean_normalized_fluo_intensities(
        label=label, cell_id=cell_id, y_label="Mean Normalized Fluorescence Intensity"
    )
    return ret


@router_cell.get("/{db_name}/{label}/{cell_id}/median_fluo_intensities")
async def get_median_fluo_intensities(
    db_name: str,
    label: str,
    cell_id: str,
    img_type: Literal["fluo1", "fluo2"] = "fluo1",
):
    await AsyncChores().validate_database_name(db_name)
    ret: StreamingResponse = await CellCrudBase(
        db_name=db_name
    ).get_all_median_normalized_fluo_intensities(
        label=label,
        cell_id=cell_id,
        y_label="Median Normalized Fluorescence Intensity",
        img_type=img_type,
    )
    return ret


@router_cell.get("/{db_name}/{label}/{cell_id}/ibpa_ratio")
async def get_ibpa_ratio(
    db_name: str,
    label: str,
    cell_id: str,
    img_type: Literal["fluo1", "fluo2"] = "fluo1",
):
    await AsyncChores().validate_database_name(db_name)
    ret: StreamingResponse = await CellCrudBase(
        db_name=db_name
    ).get_ibpa_ratio(
        label=label,
        cell_id=cell_id,
        y_label="IbpA High-intensity Ratio",
        img_type=img_type,
    )
    return ret


@router_cell.get(
    "/{db_name}/{label}/ibpa_ratio/csv", response_class=StreamingResponse
)
async def get_ibpa_ratio_csv(
    db_name: str,
    label: str,
    img_type: Literal["fluo1", "fluo2"] = "fluo1",
):
    await AsyncChores().validate_database_name(db_name)
    ret: StreamingResponse = await CellCrudBase(
        db_name=db_name
    ).get_ibpa_ratio_csv(label=label, img_type=img_type)
    return ret


@router_cell.get("/{db_name}/{label}/{cell_id}/var_fluo_intensities")
async def get_var_fluo_intensities(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    ret: StreamingResponse = await CellCrudBase(
        db_name=db_name
    ).get_all_variance_normalized_fluo_intensities(
        label=label,
        cell_id=cell_id,
        y_label="Variance Normalized Fluorescence Intensity",
    )
    return ret


@router_cell.get("/{db_name}/{label}/{cell_id}/sd_fluo_intensities")
async def get_sd_fluo_intensities(
    db_name: str,
    label: str,
    cell_id: str,
    img_type: Literal["ph", "fluo1", "fluo2"] = "fluo1",
):
    await AsyncChores().validate_database_name(db_name)
    ret: StreamingResponse = await CellCrudBase(
        db_name=db_name
    ).get_all_sd_normalized_fluo_intensities(
        label=label,
        cell_id=cell_id,
        y_label="SD Normalized Fluorescence Intensity",
        img_type=img_type,
    )
    return ret


@router_cell.get("/{db_name}/{label}/{cell_id}/cv_fluo_intensities")
async def get_cv_fluo_intensities(
    db_name: str,
    label: str,
    cell_id: str,
    img_type: Literal["ph", "fluo1", "fluo2"] = "fluo1",
):
    await AsyncChores().validate_database_name(db_name)
    ret: StreamingResponse = await CellCrudBase(
        db_name=db_name
    ).get_all_cv_normalized_fluo_intensities(
        label=label,
        cell_id=cell_id,
        y_label="CV Normalized Fluorescence Intensity",
        img_type=img_type,
    )
    return ret


@router_cell.get("/{db_name}/{label}/{cell_id}/area_fraction")
async def get_area_fraction(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    ret: StreamingResponse = await CellCrudBase(
        db_name=db_name
    ).get_all_area_fractions(
        label=label,
        cell_id=cell_id,
        y_label="Nucleoid Area Fraction",
    )
    return ret


@router_cell.get(
    "/{db_name}/{label}/median_fluo_intensities/csv", response_class=StreamingResponse
)
async def get_median_fluo_intensities_csv(
    db_name: str, label: str, img_type: Literal["fluo1", "fluo2"] = "fluo1"
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(
        db_name=db_name
    ).get_all_median_normalized_fluo_intensities_csv(label=label, img_type=img_type)


@router_cell.get(
    "/{db_name}/{label}/mean_fluo_intensities/csv", response_class=StreamingResponse
)
async def get_mean_fluo_intensities_csv(db_name: str, label: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(
        db_name=db_name
    ).get_all_mean_normalized_fluo_intensities_csv(label=label)


@router_cell.get(
    "/{db_name}/{label}/var_fluo_intensities/csv", response_class=StreamingResponse
)
async def get_var_fluo_intensities_csv(db_name: str, label: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(
        db_name=db_name
    ).get_all_variance_normalized_fluo_intensities_csv(label=label)


@router_cell.get(
    "/{db_name}/{label}/sd_fluo_intensities/csv", response_class=StreamingResponse
)
async def get_sd_fluo_intensities_csv(
    db_name: str,
    label: str,
    img_type: Literal["ph", "fluo1", "fluo2"] = "fluo1",
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(
        db_name=db_name
    ).get_all_sd_normalized_fluo_intensities_csv(label=label, img_type=img_type)


@router_cell.get(
    "/{db_name}/{label}/cv_fluo_intensities/csv", response_class=StreamingResponse
)
async def get_cv_fluo_intensities_csv(
    db_name: str,
    label: str,
    img_type: Literal["ph", "fluo1", "fluo2"] = "fluo1",
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(
        db_name=db_name
    ).get_all_cv_normalized_fluo_intensities_csv(label=label, img_type=img_type)


@router_cell.get(
    "/{db_name}/{label}/pixel_engine/csv", response_class=StreamingResponse
)
async def export_pixel_engine_csv(
    db_name: str,
    label: str,
    img_type: Literal["ph", "fluo1", "fluo2"] = "fluo1",
):
    await AsyncChores().validate_database_name(db_name)
    normalized_label: str | None
    if label == "74":
        normalized_label = None
    elif label == "1000":
        normalized_label = "N/A"
    else:
        normalized_label = label

    return await CellCrudBase(db_name=db_name).get_pixel_intensities_csv(
        label=normalized_label, img_type=img_type
    )


@router_cell.get(
    "/{db_name}/{label}/area_fraction/csv", response_class=StreamingResponse
)
async def get_area_fraction_csv(db_name: str, label: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(
        db_name=db_name
    ).get_all_area_fractions_csv(label=label)


@router_cell.get(
    "/{db_name}/{label}/{cell_id}/heatmap", response_class=StreamingResponse
)
async def get_heatmap(db_name: str, label: str, cell_id: str):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).heatmap_path(cell_id=cell_id, degree=4)


@router_cell.get("/{db_name}/{cell_id}/distribution", response_class=StreamingResponse)
async def get_fluo_distribution(
    db_name: str, cell_id: str, channel: int = 1
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).extract_intensity_and_create_histogram(
        label="", cell_id=cell_id, channel=channel
    )


@router_cell.get(
    "/{db_name}/{label}/{cell_id}/distribution_normalized",
    response_class=StreamingResponse,
)
async def get_fluo_distribution_normalized(
    db_name: str, cell_id: str, channel: int = 1
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(
        db_name=db_name
    ).extract_normalized_intensity_and_create_histogram(
        label="", cell_id=cell_id, channel=channel
    )


@router_cell.get(
    "/{db_name}/{label}/{cell_id}/distribution_normalized/raw_points",
)
async def get_fluo_distribution_normalized_raw_points(
    db_name: str, cell_id: str, channel: int = 1
):
    await AsyncChores().validate_database_name(db_name)
    return JSONResponse(
        content={
            "raw_points": await CellCrudBase(
                db_name=db_name
            ).extract_normalized_intensities_raw(cell_id=cell_id, channel=channel)
        }
    )


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
    "/{db_name}/{label}/{cell_id}/heatmap/bulk/csv",
    response_class=StreamingResponse,
)
async def get_heatmap_bulk_csv(db_name: str, label: str = "1", channel: int = 1):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_peak_paths_csv(
        degree=4, label=label, channel=channel
    )


@router_cell.get(
    "/{db_name}/{label}/{cell_id}/paths_plot", response_class=StreamingResponse
)
async def get_paths_plot(db_name: str, label: str = 1):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).plot_peak_paths(label=label)


@router_cell.get("/{db_name}/{cell_id}/3d")
async def get_3d_plot(db_name: str, cell_id: str, channel: int = 1):
    await AsyncChores().validate_database_name(db_name)
    image_buf = await CellCrudBase(db_name=db_name).get_cloud_points(
        cell_id=cell_id, channel=channel
    )
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
    saved_name = f"{file.filename.split('/')[-1].split('.')[0]}-uploaded.db"
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, migrate, saved_name)
    return file.filename


@router_database.get("/")
async def get_databases(account=Depends(get_account_optional)):
    if account is None:
        return await AsyncChores().get_database_names()

    user_id = account.id
    handle_id = account.handle_id

    account_obj = await UserCrud.get_by_id(user_id)
    if account_obj.is_admin:
        return await AsyncChores().get_database_names()

    return await AsyncChores().get_database_names(handle_id=handle_id)


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
    return FileResponse(
        f"databases/{db_name.split('/')[-1]}",
    )


@router_database.get("/{db_name}/combined_images")
async def get_cell_images_combined(
    db_name: str,
    label: Literal[
        "N/A",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ] = "1",
    mode: Literal[
        "fluo",
        "ph",
        "ph_contour",
        "fluo_contour",
        "fluo2",
        "fluo2_contour",
        "replot_fluo1",
        "replot_fluo2",
    ] = "fluo",
):
    await AsyncChores().validate_database_name(db_name)
    return await CellCrudBase(db_name=db_name).get_cell_images_combined(
        label=label, mode=mode
    )


@router_database.get("/{db_name}/metadata")
async def get_metadata(db_name: str):
    return await CellCrudBase(db_name=db_name).get_metadata()


@router_database.get("/{db_name}/has-fluo2")
async def check_has_fluo2(db_name: str):
    await AsyncChores().validate_database_name(db_name)
    exists = await CellCrudBase(db_name=db_name).has_fluo2()
    return JSONResponse(content={"has_fluo2": exists})


@router_database.patch("/{db_name}/update-metadata")
async def update_label_experiment(db_name: str, request: MetadataUpdateRequest):
    if db_name == "test_database.db":
        raise HTTPException(
            status_code=400, detail="Cannot update metadata for the test database."
        )
    return await CellCrudBase(db_name=db_name).update_all_cells_metadata(
        request.metadata
    )
