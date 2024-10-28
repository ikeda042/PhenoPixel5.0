import os
import shutil
from typing import Literal
import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from CellExtraction.crud import ExtractionCrudBase
from CellExtraction.schemas import CellExtractionResponse

router_cell_extraction = APIRouter(prefix="/cell_extraction", tags=["cell_extraction"])


@router_cell_extraction.get("/ph_contours/{session_ulid}/count")
async def get_ph_contours_count(session_ulid: str):
    return JSONResponse(
        content={
            "count": await ExtractionCrudBase("").get_ph_contours_num(ulid=session_ulid)
        }
    )


@router_cell_extraction.get(
    "/ph_contours/{session_ulid}/{frame_num}", response_class=StreamingResponse
)
async def get_ph_contours(frame_num: int, session_ulid: str):
    return await ExtractionCrudBase("").get_ph_contours(frame_num, session_ulid)


@router_cell_extraction.post("/nd2_files")
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


@router_cell_extraction.get("/nd2_files")
async def get_nd2_files():
    return JSONResponse(
        content={"files": await ExtractionCrudBase("").get_nd2_filenames()}
    )


@router_cell_extraction.delete("/nd2_files/{file_name}")
async def delete_nd2_file(file_name: str):
    file_path = os.path.join("uploaded_files", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    await ExtractionCrudBase("").delete_nd2_file(file_path)
    return JSONResponse(content={"message": "File deleted"})


@router_cell_extraction.get("/{db_name}/{mode}")
async def extract_cells(
    db_name: str,
    mode: Literal["single_layer", "dual_layer", "triple_layer"] = "dual",
    param1: int = 100,
    image_size: int = 200,
    reverse_layers: bool = False,
):
    # ph_contours_dir = "ph_contours"
    # try:
    #     shutil.rmtree(ph_contours_dir)
    # except:
    #     pass

    file_path = os.path.join("uploaded_files", db_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        extractor = ExtractionCrudBase(
            nd2_path=file_path,
            mode=mode,
            param1=param1,
            image_size=image_size,
            reverse_layers=reverse_layers,
        )
        # return value : num_tiff:int, ulid:str
        ret = await extractor.main()
        return CellExtractionResponse(num_tiff=int(ret[0]), ulid=str(ret[1]))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router_cell_extraction.delete("ph_contours/{ulid}")
async def delete_extracted_files(ulid: str):
    ph_contours_dir = f"ph_contours{ulid}"
    try:
        shutil.rmtree(ph_contours_dir)
    except:
        raise HTTPException(status_code=404, detail="Files not found")
    return JSONResponse(content={"message": "Files deleted"})
