from fastapi import APIRouter
from TimeLapseEngine.crud import TimelapseEngineCrudBase
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Literal
import os
from fastapi import UploadFile
import aiofiles
from fastapi import HTTPException
import shutil

router_tl_engine = APIRouter(prefix="/tl-engine_x100", tags=["tl_engine_x100"])


@router_tl_engine.post("/nd2_files")
async def upload_nd2_file(file: UploadFile):
    filename = file.filename.split(".")[0]
    ext = file.filename.split(".")[1]
    file_path = filename + "_timelapse." + ext
    file_path = os.path.join("uploaded_files", file_path)
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await file.read(1024 * 1024 * 100):
                await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"filename": file.filename})


@router_tl_engine.get("/nd2_files")
async def get_nd2_files():
    return JSONResponse(
        content={"files": await TimelapseEngineCrudBase("").get_nd2_filenames()}
    )
