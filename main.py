from fastapi import FastAPI
import uvicorn
from tmp import init, extract_nd2
import asyncio


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Response": "Root"}


@app.get("/init/{filename}")
async def init_api(filename: str):
    loop = asyncio.get_event_loop()

    await loop.run_in_executor(None, extract_nd2, filename)
    return await loop.run_in_executor(
        None, init, filename.split(".")[0] + ".tif", 100, "dual"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
