from fastapi import FastAPI
import uvicorn
from tmp import init, extract_nd2, image_process, AsyncCellCRUD, Cell
import asyncio
from fastapi.responses import FileResponse
import os
import re
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

# setting cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"Response": "Root"}


@app.get("/init/{filename}")
async def init_api(filename: str):
    loop = asyncio.get_event_loop()

    await loop.run_in_executor(None, extract_nd2, filename)
    num_images = await loop.run_in_executor(
        None, init, filename.split(".")[0] + ".tif", 100, "dual"
    )
    await loop.run_in_executor(
        None, image_process, filename.split(".")[0] + ".tif", 85, 255, 100
    )


def custom_sort_key(filename):
    match = re.match(r"F(\d+)C(\d+)", filename)
    if match:
        f_num = int(match.group(1))
        c_num = int(match.group(2))
        return (f_num, c_num)
    return (float("inf"), float("inf"))


@app.get("/image/filenames")
async def get_image_filenames():
    filenames = sorted(os.listdir("TempData/app_data"), key=custom_sort_key)
    return {"filenames": filenames}


@app.get("/image/filenames/count")
async def get_image_filenames_count():
    return {"count": len(os.listdir("TempData/app_data"))}


@app.get("/image/{filename}")
async def get_image(filename: str):
    return FileResponse(f"TempData/app_data/{filename}")


@app.get("/database/cell/{dbname}")
async def get_cell(dbname: str):
    CELLDB: AsyncCellCRUD = AsyncCellCRUD(db_name=dbname)
    cell_ids: list[str] = await CELLDB.read_all_cell_ids()
    return {"cells": cell_ids}


@app.get("/database/cell/{dbname}/{cell_id}/manual_label")
async def get_cell_manual_label(dbname: str, cell_id: str):
    cell_id = cell_id.replace(" ", "").replace("\n", "")
    if "." in cell_id:
        cell_id = cell_id.split(".")[0]
    CELLDB: AsyncCellCRUD = AsyncCellCRUD(db_name=dbname)
    cell: Cell = await CELLDB.read_cell(cell_id)
    if cell is None:
        return {"cell": None}
    return {"cell": cell.manual_label}


@app.patch("/database/cell/{dbname}/{cell_id}/{label}")
async def update_cell(dbname: str, cell_id: str, label: str):
    cell_id = cell_id.replace(" ", "").replace("\n", "")
    if "." in cell_id:
        cell_id = cell_id.split(".")[0]
    CELLDB: AsyncCellCRUD = AsyncCellCRUD(db_name=dbname)
    await CELLDB.update_cell_manual_label(cell_id, label)
    return {"response": "success"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
