from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import os
import aiofiles
from TimeLapseEngine.crud import TimelapseDatabaseCrud
import io

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


@router_tl_engine.get("/nd2_files/{file_name}/cells")
async def extract_all_cells(file_name: str, param_1: int, crop_size: int = 200):
    """
    全てのフィールドからセルを抽出し、データベースに保存するエンドポイント。
    """
    db_name = file_name.split(".")[0] + "_cells.db"
    fields = await TimelapseEngineCrudBase(file_name).get_fields_of_nd2()
    db_path = f"timelapse_databases/{db_name}"
    if os.path.exists(db_path):
        os.remove(db_path)
    for Field in fields:
        await TimelapseEngineCrudBase(file_name).extract_cells(
            field=Field, dbname=db_name, param1=param_1, crop_size=crop_size
        )
    return JSONResponse(content={"message": "Cells extracted and saved to database."})


@router_tl_engine.get("/nd2_files/{file_name}/cells/{field_name}/{cell_number}/gif")
async def create_gif_for_cell_endpoint(
    file_name: str, field_name: str, cell_number: int
):
    """
    セル番号に対応する GIF を生成し、ストリーミングで返すエンドポイント。
    """
    db_name = file_name.split(".")[0] + f"_cells.db"
    crud = TimelapseEngineCrudBase(file_name)
    gif_buffer = await crud.create_gif_for_cell(
        field=field_name, cell_number=cell_number, dbname=db_name
    )

    return StreamingResponse(
        gif_buffer,
        media_type="image/gif",
        headers={"Content-Disposition": "attachment; filename=cell.gif"},
    )


@router_tl_engine.get("/nd2_files/{file_name}/cells/{field_name}/gif")
async def create_gif_for_cells_endpoint(file_name: str, field_name: str):
    """
    指定したフィールド内の全セルについて、GIF を生成し、ストリーミングで返すエンドポイント。
    """
    db_name = file_name.split(".")[0] + f"_cells.db"
    gif_buffer = await TimelapseEngineCrudBase(file_name).create_gif_for_cells(
        field=field_name, dbname=db_name
    )

    return StreamingResponse(
        gif_buffer,
        media_type="image/gif",
        headers={"Content-Disposition": "attachment; filename=cells.gif"},
    )


@router_tl_engine.get("/databases/{db_name}/cells/by_field/{field}")
async def read_cells_by_field(db_name: str, field: str):
    """
    指定したデータベース(db_name)から、指定したFieldに属するセル一覧を取得するエンドポイント
    """
    crud = TimelapseDatabaseCrud(dbname=db_name)
    cells = await crud.get_cells_by_field(field)
    # 必要に応じてレスポンスを整形
    # 例: JSON で返す
    cell_list = []
    for c in cells:
        cell_list.append(
            {
                "id": c.id,
                "cell_id": c.cell_id,
                "field": c.field,
                "time": c.time,
                "cell": c.cell,
                "area": c.area,
                "perimeter": c.perimeter,
            }
        )
    return JSONResponse(content={"cells": cell_list})


@router_tl_engine.get("/databases/{db_name}/cells/by_id/{cell_id}")
async def read_cell_by_cell_id(db_name: str, cell_id: str):
    """
    指定したデータベース(db_name)から、cell_id で 1件だけ取得するエンドポイント
    """
    crud = TimelapseDatabaseCrud(dbname=db_name)
    try:
        cell = await crud.get_cell_by_id(cell_id)
        return JSONResponse(
            content={
                "id": cell.id,
                "cell_id": cell.cell_id,
                "field": cell.field,
                "time": cell.time,
                "cell": cell.cell,
                "area": cell.area,
                "perimeter": cell.perimeter,
                "manual_label": cell.manual_label,
                "is_dead": cell.is_dead,
            }
        )
    except:
        # get_cell_by_id で該当が無い場合は scalar_one() が例外を出す
        raise HTTPException(status_code=404, detail=f"Cell not found: {cell_id}")


@router_tl_engine.get(
    "/databases/{db_name}/cells/by_field/{field}/cell_number/{cell_number}"
)
async def read_cells_by_cell_number(db_name: str, field: str, cell_number: int):
    """
    指定したデータベース(db_name)から、field + cell_number で該当するセルを全部取得するエンドポイント
    """
    crud = TimelapseDatabaseCrud(dbname=db_name)
    cells = await crud.get_cells_by_cell_number(field, cell_number)
    if not cells:
        raise HTTPException(
            status_code=404,
            detail=f"No cells found in db={db_name}, field={field}, cell_number={cell_number}",
        )

    cell_list = []
    for c in cells:
        cell_list.append(
            {
                "id": c.id,
                "cell_id": c.cell_id,
                "field": c.field,
                "time": c.time,
                "cell": c.cell,
                "area": c.area,
                "perimeter": c.perimeter,
                "manual_label": c.manual_label,
            }
        )
    return JSONResponse(content={"cells": cell_list})


@router_tl_engine.get("/databases/{db_name}/cells/gif/{field}/{cell_number}")
async def get_cell_gif(
    db_name: str,
    field: str,
    cell_number: int,
    channel: str = "ph",
):
    """
    指定したフィールド & セル番号 の時系列 GIF を返すエンドポイント
    チャネルはクエリパラメータ ?channel=ph|fluo1|fluo2 （デフォルト ph）
    """
    crud = TimelapseDatabaseCrud(dbname=db_name)
    gif_buffer: io.BytesIO = await crud.get_cells_gif_by_cell_number(
        field, cell_number, channel
    )
    return StreamingResponse(
        gif_buffer,
        media_type="image/gif",
        headers={
            "Content-Disposition": f"attachment; filename={field}_{cell_number}_{channel}.gif"
        },
    )


@router_tl_engine.get("/databases")
async def get_databases():
    """
    データベースの一覧を取得するエンドポイント
    """
    return JSONResponse(
        content={"databases": await TimelapseDatabaseCrud("").get_database_names()}
    )


@router_tl_engine.get("/databases/{db_name}/fields")
async def get_fields_of_db(db_name: str):
    """
    指定したデータベース(db_name)のフィールド一覧を取得するエンドポイント
    """
    crud = TimelapseDatabaseCrud(dbname=db_name)
    return JSONResponse(content={"fields": await crud.get_fields_of_db()})


@router_tl_engine.get("/databases/{db_name}/fields/{field}/cell_numbers")
async def get_cell_numbers_of_field(db_name: str, field: str):
    """
    指定したデータベース(db_name)のフィールド(field)のセル番号一覧を取得するエンドポイント
    """
    crud = TimelapseDatabaseCrud(dbname=db_name)
    return JSONResponse(
        content={"cell_numbers": await crud.get_cell_numbers_of_field(field)}
    )


@router_tl_engine.patch("/databases/{db_name}/cells/{base_cell_id}/label")
async def update_manual_label(db_name: str, base_cell_id: str, label: str):
    """
    指定したデータベース(db_name)のセル(cell_id)の manual_label を更新するエンドポイント
    """
    crud = TimelapseDatabaseCrud(dbname=db_name)
    result = await crud.update_manual_label(base_cell_id, label)
    return JSONResponse(content={"updated": result})


@router_tl_engine.patch("/databases/{db_name}/cells/{base_cell_id}/dead/{is_dead}")
async def update_dead_status(db_name: str, base_cell_id: str, is_dead: int):
    """
    指定したデータベース(db_name)のセル(cell_id)の is_dead を更新するエンドポイント
    """
    crud = TimelapseDatabaseCrud(dbname=db_name)
    result = await crud.update_dead_status(base_cell_id, is_dead)
    return JSONResponse(content={"updated": result})


@router_tl_engine.get("/databases/{db_name}/cells/{field}/{cell_number}/contour_areas")
async def get_contour_areas_by_cell_number(db_name: str, field: str, cell_number: int):
    """
    指定したデータベース(db_name)のセル(field, cell_number)の輪郭面積を取得するエンドポイント
    """
    crud = TimelapseDatabaseCrud(dbname=db_name)
    areas = await crud.get_contour_areas_by_cell_number(field, cell_number)
    return JSONResponse(content={"areas": areas})
