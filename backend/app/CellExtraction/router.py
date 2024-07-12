from fastapi import APIRouter
from CellExtraction.crud import ExtractionCrudBase
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
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


@router_cell_extraction.get("/{db_name}/{mode}")
async def extract_cells(
    db_name: str, mode: Literal["single_layer", "dual_layer", "triple_layer"] = "dual"
):
    file_path = os.path.join("databases", db_name).replace(".db", "-uploaded.db")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        extractor = ExtractionCrudBase(nd2_path=file_path, mode=mode)
        out_db = await extractor.main()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
