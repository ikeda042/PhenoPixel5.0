from fastapi import APIRouter
from fastapi import UploadFile
import pandas as pd
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
import io
from GraphEngine.crud import GraphEngineCrudBase, MCPROperation

router_graphengine = APIRouter(prefix="/graph_engine", tags=["gragh_engine"])


@router_graphengine.post("/heatmap_abs")
async def create_heatmap_abs(file: UploadFile):
    content = await file.read()
    data = pd.read_csv(
        io.StringIO(content.decode("utf-8")), header=None
    ).values.tolist()
    return StreamingResponse(
        await GraphEngineCrudBase.process_heatmap_abs(data), media_type="image/png"
    )


@router_graphengine.post("/heatmap_rel")
async def create_heatmap_rel(file: UploadFile):
    content = await file.read()
    data = pd.read_csv(
        io.StringIO(content.decode("utf-8")), header=None
    ).values.tolist()
    return StreamingResponse(
        await GraphEngineCrudBase.process_heatmap_rel(data), media_type="image/png"
    )


@router_graphengine.post("/mcpr_operations/")
async def mcpr_operations(
    file: UploadFile, blank_index: str = "", timespan_sec: int = 180
):
    file_content = await file.read()
    image_buffers = await MCPROperation._draw_graph(
        file_content, blank_index, timespan_sec
    )
    combined_image_buffer = await MCPROperation.combine_images(image_buffers, per_row=3)
    return StreamingResponse(combined_image_buffer, media_type="image/png")
