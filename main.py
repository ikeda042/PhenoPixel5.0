from fastapi import FastAPI
import uvicorn
from tmp import init, extract_nd2, image_process
import asyncio
from fastapi.responses import FileResponse
import os
import re

## CORs
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
