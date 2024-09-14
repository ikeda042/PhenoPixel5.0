from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
from fastapi import UploadFile
import aiofiles
from fastapi import HTTPException
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import io
from GraphEngine.crud import GraphEngineCrudBase

router_graphengine = APIRouter(prefix="/graph_engine", tags=["gragh_engine"])


@router_graphengine.post("/heatmap_abs")
async def create_heatmap(file: UploadFile):
    content = await file.read()
    data = pd.read_csv(
        io.StringIO(content.decode("utf-8")), header=None
    ).values.tolist()
    return StreamingResponse(
        await GraphEngineCrudBase.process_heatmap(data), media_type="image/png"
    )
