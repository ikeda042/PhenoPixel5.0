from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import aiofiles
from Dev.crud import HINETLogin

router_dev = APIRouter(prefix="/dev", tags=["dev"])


@router_dev.post("/xlsx_files")
async def upload_xlsx_file(file: UploadFile):
    file_path = file.filename
    file_path = os.path.join("uploaded_files", file_path)
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await file.read(1024 * 1024 * 100):
                await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"filename": file.filename})


@router_dev.get("/hinet-login")
async def login_to_hinet():
    hinet_login = HINETLogin()
    await hinet_login.login()
    return JSONResponse(content={"message": "Login successful."})


@router_dev.get("/git-pull")
async def pull_from_git():
    await HINETLogin.git_pull()
    return JSONResponse(content={"message": "Git pull successful."})


