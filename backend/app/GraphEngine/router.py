from fastapi import APIRouter
from fastapi import UploadFile
import pandas as pd
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
import io
from GraphEngine.crud import GraphEngineCrudBase
from GraphEngine.mcpr_crud import _draw_graph_from_memory, combine_images_in_memory
from fastapi.responses import FileResponse
import asyncio

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


@router_graphengine.post("/mcpr")
async def mcpr(file: UploadFile, blank_index: str = "2", timespan_sec: int = 180):
    image_list = await asyncio.to_thread(
        _draw_graph_from_memory, file, blank_index, timespan_sec
    )
    per_row = 2 if len(image_list) < 3 else len(image_list) // 2
    combined_image = await combine_images_in_memory(image_list, per_row)

    buf = io.BytesIO(combined_image)
    buf.seek(0)

    headers = {"Content-Disposition": f'inline; filename="combined.png"'}
    return StreamingResponse(buf, media_type="image/png", headers=headers)
