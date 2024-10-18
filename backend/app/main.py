import os

import aiohttp
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from CellAI.router import router_cell_ai
from CellDBConsole.router import router_cell, router_database
from CellExtraction.router import router_cell_extraction
from Dev.router import router_dev
from Dropbox.router import router_dropbox
from GraphEngine.router import router_graphengine
from TimeLapseEngine.router import router_tl_engine
from results.router import router_results
from Dev.crud import HINETLogin

load_dotenv()

api_title = os.getenv("API_TITLE", "FastAPI")
api_prefix = os.getenv("API_PREFIX", "/api")
test_env = os.getenv("TEST_ENV", "")
app = FastAPI(
    title=api_title,
    docs_url=f"{api_prefix}/docs",
    openapi_url=f"{api_prefix}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
async def root():
    return {
        "title": api_title,
        "api_prefix": api_prefix,
        "docs_url": f"{api_prefix}/docs",
        "openapi_url": f"{api_prefix}/openapi.json",
        "status": "ok",
        "code": 200,
    }


@app.get(f"{api_prefix}/healthcheck")
async def healthcheck():
    return {"status": "ok"}


@app.get(f"{api_prefix}/env")
async def get_env():
    return {
        "API_TITLE": api_title,
        "API_PREFIX": api_prefix,
        "TEST_ENV": test_env,
    }


async def check_internet_connection():
    url = "https://www.google.com"
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url):
                return True
        except aiohttp.ClientError:
            return False


@app.get(f"{api_prefix}/internet-connection")
async def internet_connection():
    return {"status": await check_internet_connection()}


@app.post(f"{api_prefix}/replace_env")
async def replace_env(file: UploadFile):
    contents = await file.read()
    with open(".env", "wb") as f:
        f.write(contents)
    return {"status": "ok"}


async def periodic_task(interval: int):
    while True:
        connection_status = await check_internet_connection()
        print(f"Internet connection status: {connection_status}")
        await asyncio.sleep(interval)


async def periodic_task(interval: int):
    while True:
        if await check_internet_connection():
            await HINETLogin().login()
        await asyncio.sleep(interval)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_task(1))


app.include_router(router_cell, prefix=api_prefix)
app.include_router(router_database, prefix=api_prefix)
app.include_router(router_cell_extraction, prefix=api_prefix)
app.include_router(router_dev, prefix=api_prefix)
app.include_router(router_graphengine, prefix=api_prefix)
app.include_router(router_tl_engine, prefix=api_prefix)
app.include_router(router_cell_ai, prefix=api_prefix)
app.include_router(router_dropbox, prefix=api_prefix)
app.include_router(router_results, prefix=api_prefix)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
