import json
import os
import shutil
from typing import Literal
import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from CellExtraction.crud import ExtractionCrudBase, notify_slack_database_created
from CellExtraction.schemas import CellExtractionResponse, FrameSplitConfig
from OAuth2.login_manager import get_account_optional

router_cell_extraction = APIRouter(prefix="/cell_extraction", tags=["cell_extraction"])


def _ensure_non_overlapping_ranges(configs: list[FrameSplitConfig]) -> None:
    if not configs:
        return
    ordered = sorted(configs, key=lambda cfg: (cfg.frame_start, cfg.frame_end))
    for prev, curr in zip(ordered, ordered[1:]):
        if curr.frame_start <= prev.frame_end:
            raise HTTPException(
                status_code=400,
                detail="Split frame ranges must not overlap.",
            )


def _parse_split_frames(split_frames: str | None) -> list[FrameSplitConfig]:
    if not split_frames:
        return []
    try:
        data = json.loads(split_frames)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid split_frames: {exc.msg}")
    if not isinstance(data, list):
        raise HTTPException(
            status_code=400, detail="split_frames must be a JSON array"
        )
    try:
        configs = [FrameSplitConfig(**item) for item in data]
    except (TypeError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid split config: {exc}")
    _ensure_non_overlapping_ranges(configs)
    return configs


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
    filename = os.path.basename(file.filename)
    base, ext = os.path.splitext(filename)
    sanitized = base.replace(".", "p") + ext
    file_path = os.path.join("uploaded_files", sanitized)
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await file.read(1024 * 1024 * 100):
                await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"filename": sanitized})


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
    mode: Literal[
        "single_layer",
        "dual_layer",
        "triple_layer",
        "quad_layer",
        "dual_layer_reversed",
    ] = "dual",
    param1: int = 100,
    image_size: int = 200,
    reverse_layers: bool = False,
    split_frames: str | None = None,
    account=Depends(get_account_optional),
):

    file_path = os.path.join("uploaded_files", db_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    frame_splits = _parse_split_frames(split_frames)
    try:
        extractor = ExtractionCrudBase(
            nd2_path=file_path,
            mode=mode,
            param1=param1,
            image_size=image_size,
            reverse_layers=reverse_layers,
            user_id=account.handle_id if account else None,
            frame_splits=frame_splits,
        )
        ret = await extractor.main()
        raw_databases = ret[2] if len(ret) >= 3 else []
        if isinstance(raw_databases, list):
            created_databases = raw_databases
        else:
            created_databases = [
                {"frame_start": 0, "frame_end": 0, "db_name": str(raw_databases)}
            ]
        primary_db_name = (
            created_databases[0]["db_name"] if created_databases else ret[2]
        )
        return CellExtractionResponse(
            num_tiff=int(ret[0]),
            ulid=str(ret[1]),
            db_name=str(primary_db_name),
            created_databases=created_databases,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router_cell_extraction.delete("/ph_contours_delete/{ulid}")
async def delete_extracted_files(ulid: str):
    ph_contours_dir = f"ph_contours{ulid}"
    try:
        shutil.rmtree(ph_contours_dir)
    except:
        raise HTTPException(status_code=404, detail="Files not found")
    return JSONResponse(content={"message": "Files deleted"})


@router_cell_extraction.post("/databases/{db_name}/notify_created")
async def notify_database_created(db_name: str):
    db_path = os.path.join("databases", db_name)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="File not found")
    await notify_slack_database_created(db_path)
    return JSONResponse(content={"message": "Slack notified"})
