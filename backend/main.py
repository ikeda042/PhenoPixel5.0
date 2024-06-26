from fastapi import FastAPI
import uvicorn
from tmp import init
import asyncio


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Response": "Root"}


@app.get("/init/{filename}")
async def init_api(filename: str):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, init, filename, "dual")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
