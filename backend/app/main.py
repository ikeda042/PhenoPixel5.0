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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(f"{api_prefix}/healthcheck")
async def healthcheck():
    return {"status": "ok"}


app.include_router(router_cell, prefix=api_prefix)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
