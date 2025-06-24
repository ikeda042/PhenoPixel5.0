from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from CDT.crud import CdtCrudBase
from typing import List

router_cdt = APIRouter(prefix="/cdt", tags=["cdt"])


@router_cdt.post("/nagg")
async def calc_nagg(ctrl_file: UploadFile, files: List[UploadFile]):
    ctrl_content = await ctrl_file.read()
    await CdtCrudBase.set_ctrl(ctrl_content)
    file_contents = []
    for f in files:
        content = await f.read()
        file_contents.append((f.filename, content))
    results = await CdtCrudBase.analyze_files(file_contents)
    return JSONResponse(content=[r.__dict__ for r in results])
