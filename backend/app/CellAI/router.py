from fastapi import APIRouter
from CellAI.crud import CellAiCrudBase
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Literal
import os
from fastapi import UploadFile
from fastapi import HTTPException

router_cell_ai = APIRouter(prefix="/cell_ai", tags=["cell_ai"])


@router_cell_ai.get("/{db_name}/{cell_id}")
async def get_predicted_contour(db_name: str, cell_id: str, model: Literal["T1", "T2"]):
    return await CellAiCrudBase(
        db_name,
    ).predict_contour(cell_id)
