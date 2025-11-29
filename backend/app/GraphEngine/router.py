from fastapi import APIRouter, HTTPException, Query
from fastapi import UploadFile
import pandas as pd
from fastapi.responses import StreamingResponse
import io
from GraphEngine.crud import GraphEngineCrudBase
from GraphEngine.mcpr_crud import _draw_graph_from_memory, combine_images_in_memory
import asyncio
from CellDBConsole.crud import CellCrudBase, AsyncChores as CellAsyncChores

router_graphengine = APIRouter(prefix="/graph_engine", tags=["gragh_engine"])


@router_graphengine.post("/heatmap_abs")
async def create_heatmap_abs(file: UploadFile):
    content = await file.read()
    data = pd.read_csv(
        io.StringIO(content.decode("utf-8")), header=None
    ).values.tolist()
    return StreamingResponse(
        await GraphEngineCrudBase.process_heatmap_abs(data, dpi=100),
        media_type="image/png",
    )


@router_graphengine.post("/heatmap_rel")
async def create_heatmap_rel(file: UploadFile):
    content = await file.read()
    data = pd.read_csv(
        io.StringIO(content.decode("utf-8")), header=None
    ).values.tolist()
    return StreamingResponse(
        await GraphEngineCrudBase.process_heatmap_rel(data, dpi=100),
        media_type="image/png",
    )


@router_graphengine.post("/distribution")
async def create_distribution(file: UploadFile):
    content = await file.read()
    data = pd.read_csv(io.StringIO(content.decode("utf-8")), header=None).values.tolist()
    return StreamingResponse(
        await GraphEngineCrudBase.process_distribution(data, dpi=100),
        media_type="image/png",
    )


@router_graphengine.post("/distribution_box")
async def create_distribution_box(file: UploadFile):
    content = await file.read()
    data = pd.read_csv(io.StringIO(content.decode("utf-8")), header=None).values.tolist()
    return StreamingResponse(
        await GraphEngineCrudBase.process_distribution_box(data, dpi=100),
        media_type="image/png",
    )


@router_graphengine.get("/cell_lengths")
async def create_cell_length_boxplot(
    db_name: str = Query(..., description="Target database name"),
    label: str = Query(..., description="Manual label to filter"),
):
    try:
        await CellAsyncChores().validate_database_name(db_name)
    except Exception:
        raise HTTPException(status_code=404, detail="Database not found")

    try:
        lengths = await CellCrudBase(db_name).get_cell_lengths_by_label(label)
        if not lengths:
            raise HTTPException(
                status_code=404, detail="No cells found for the specified label."
            )

        buf = await GraphEngineCrudBase.boxplot_from_values(
            lengths,
            title=f"{db_name} | label {label}",
            xlabel="Cell length (μm)",
            dpi=180,
        )
        return StreamingResponse(buf, media_type="image/png")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create plot: {exc}") from exc


@router_graphengine.post("/mcpr")
async def mcpr(
    file: UploadFile,
    blank_index: str = "2",
    timespan_sec: int = 180,
    lower_OD: float = 0.1,
    upper_OD: float = 0.3,
):
    image_list = await asyncio.to_thread(
        _draw_graph_from_memory, file, blank_index, timespan_sec, lower_OD, upper_OD
    )
    per_row = 2 if len(image_list) < 3 else len(image_list) // 2
    combined_image = await combine_images_in_memory(image_list, per_row)

    buf = io.BytesIO(combined_image)
    buf.seek(0)

    headers = {"Content-Disposition": f'inline; filename="combined.png"'}
    return StreamingResponse(buf, media_type="image/png", headers=headers)
