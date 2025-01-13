from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
import aiofiles

# 上で定義した TimelapseEngineCrudBase をインポート
from .crud import TimelapseEngineCrudBase

router_tl_engine = APIRouter(prefix="/tlengine", tags=["tlengine"])


@router_tl_engine.post("/nd2_files")
async def upload_nd2_file(file: UploadFile):
    """
    ND2ファイルをアップロードして保存するエンドポイント
    """
    if not file.filename.endswith(".nd2"):
        raise HTTPException(status_code=400, detail="Only .nd2 files are accepted")

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
    """
    アップロード済みND2ファイルの一覧を取得するエンドポイント。
    """
    return JSONResponse(
        content={"files": await TimelapseEngineCrudBase("").get_nd2_filenames()}
    )


@router_tl_engine.get("/nd2_files/{file_name}")
async def parse_timelapse_nd2(file_name: str):
    """
    タイムラプスND2ファイルを解析し、TIFF形式に分割保存するエンドポイント。
    解析が終わったら {"message": "Timelapse extracted"} を返す。
    """
    return await TimelapseEngineCrudBase(file_name).main()


@router_tl_engine.get("/nd2_files/{file_name}/fields")
async def get_fields_of_nd2_file(file_name: str):
    """
    ND2ファイルからFieldの一覧を取得するエンドポイント。
    例：{"fields": ["Field_1", "Field_2", ...]}
    """
    fields = await TimelapseEngineCrudBase(file_name).get_fields_of_nd2()
    return JSONResponse(content={"fields": fields})


@router_tl_engine.get("/nd2_files/{file_name}/gif/{Field}")
async def download_combined_gif(file_name: str, Field: str):
    """
    解析済みの画像から GIF を生成し、ストリーミングで返すエンドポイント。
    """
    gif_buffer = await TimelapseEngineCrudBase(file_name).create_combined_gif(Field)
    return StreamingResponse(
        gif_buffer,
        media_type="image/gif",
        headers={"Content-Disposition": "attachment; filename=combined.gif"},
    )


@router_tl_engine.delete("/nd2_files")
async def delete_nd2_file(file_path: str):
    """
    指定したND2ファイルを削除するエンドポイント。
    例）/tlengine/nd2_files?file_path=uploaded_files/xxx_timelapse.nd2
    """
    await TimelapseEngineCrudBase("").delete_nd2_file(file_path)
    return JSONResponse(content={"detail": f"{file_path} deleted successfully."})


@router_tl_engine.get("/nd2_files/{file_name}/cells/{Field}")
async def extract_cells(file_name: str, Field: str):
    """
    セルを抽出し、データベースに保存するエンドポイント。
    デバッグ用途で使う。
    """
    db_name = file_name.split(".")[0] + f"_{Field}_cells.db"
    await TimelapseEngineCrudBase(file_name).extract_cells(field=Field, dbname=db_name)
    return JSONResponse(content={"message": "Cells extracted and saved to database."})


@router_tl_engine.get("/nd2_files/{file_name}/cells")
async def extract_all_cells(file_name: str):
    """
    全てのフィールドからセルを抽出し、データベースに保存するエンドポイント。
    """
    db_name = file_name.split(".")[0] + "_cells.db"
    async for Field in TimelapseEngineCrudBase(file_name).get_fields_of_nd2():
        await TimelapseEngineCrudBase(file_name).extract_cells(
            field=Field, dbname=db_name
        )
    return JSONResponse(content={"message": "Cells extracted and saved to database."})
