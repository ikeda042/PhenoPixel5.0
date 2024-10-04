from fastapi import APIRouter
from CellExtraction.crud import ExtractionCrudBase
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Literal
import os
from fastapi import UploadFile
import aiofiles
from fastapi import HTTPException
import shutil

router_tl_engine = APIRouter(prefix="/tl-engine_x100", tags=["tl_engine_x100"])
