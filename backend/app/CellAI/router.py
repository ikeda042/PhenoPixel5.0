from fastapi import APIRouter
from CellDBConsole.crud import CellCrudBase, AsyncChores
from CellDBConsole.schemas import CellMorhology
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Literal
import os
from fastapi import UploadFile
from fastapi import HTTPException

router_cell_ai = APIRouter(prefix="/cell_ai", tags=["cell_ai"])
