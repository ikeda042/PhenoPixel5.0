from fastapi import FastAPI
import uvicorn
from tmp import init, extract_nd2, image_process
import asyncio
from fastapi.responses import FileResponse
import os


app = FastAPI()


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


@app.get("/image/filenames")
async def get_image_filenames():

    return {"filenames": os.listdir("TempData/app_data")}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
