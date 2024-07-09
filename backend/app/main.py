from fastapi import FastAPI
from CellDBConsole.router import router_cell
from fastapi.middleware.cors import CORSMiddleware
import os


api_title = os.getenv("API_TITLE", "FastAPI")
api_prefix = os.getenv("API_PREFIX", "/api")
app = FastAPI(
    title=api_title,
    docs_url=f"{api_prefix}/docs",
    openapi_url=f"{api_prefix}/openapi.json",
)

origins = ["https://phenopixel5.site", "*", "http://localhost:3000/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(f"{api_prefix}/healthcheck")
async def healthcheck():
    return {"status": "ok"}


app.include_router(router_cell, prefix=api_prefix)
