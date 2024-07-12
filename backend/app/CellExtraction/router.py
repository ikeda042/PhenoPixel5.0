from fastapi import APIRouter
from CellExtraction.crud import ExtractionCrudBase
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Literal
import os
from fastapi import UploadFile
import aiofiles
from fastapi import HTTPException

router_cell_extraction = APIRouter(prefix="/cell_extraction", tags=["cell_extraction"])


@router_cell_extraction.post("/upload_nd2")
async def upload_nd2_file(file: UploadFile):
    file_path = file.filename
    file_path = os.path.join("uploaded_files", file_path)
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await file.read(1024 * 1024 * 100):
                await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"filename": file.filename})
