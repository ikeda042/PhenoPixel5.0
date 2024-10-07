from fastapi import APIRouter
from fastapi import UploadFile
import pandas as pd
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
import io
from GraphEngine.crud import GraphEngineCrudBase

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
