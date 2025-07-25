import os
from datetime import datetime

import aiofiles
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse

UPLOAD_DIR = "filemanager_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router_file_manager = APIRouter(tags=["file_manager"])


@router_file_manager.get("/files")
async def list_files():
    try:
        files = []
        for f in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, f)
            if os.path.isfile(file_path) and not f.startswith("."):
                files.append(
                    {
                        "name": f,
                        "size": os.path.getsize(file_path),
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat(),
                    }
                )
        return files
    except FileNotFoundError:
        return []


@router_file_manager.post("/upload")
async def upload_file(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(
        content={
            "filename": file.filename,
            "size": os.path.getsize(file_path),
            "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
        }
    )


@router_file_manager.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=file_name)


@router_file_manager.delete("/delete/{file_name}")
async def delete_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(file_path)
    return JSONResponse(content={"detail": "File deleted"})
