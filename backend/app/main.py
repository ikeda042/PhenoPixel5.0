from fastapi import FastAPI
from CellDBConsole.router import router_cell, router_database
from CellExtraction.router import router_cell_extraction
from GraphEngine.router import router_graphengine
from Dev.router import router_dev
from TimeLapseEngine.router import router_tl_engine
from fastapi.middleware.cors import CORSMiddleware
import os

api_title = os.getenv("API_TITLE", "FastAPI")
api_prefix = os.getenv("API_PREFIX", "/api")
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


app.include_router(router_cell, prefix=api_prefix)
app.include_router(router_database, prefix=api_prefix)
app.include_router(router_cell_extraction, prefix=api_prefix)
app.include_router(router_dev, prefix=api_prefix)
app.include_router(router_graphengine, prefix=api_prefix)
app.include_router(router_tl_engine, prefix=api_prefix)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
