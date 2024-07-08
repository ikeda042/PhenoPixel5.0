from fastapi import FastAPI
import uvicorn
from fastapi.responses import StreamingResponse
from CellDBConsole.router import router_cell

app = FastAPI(
    title="CellAPI", docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json"
)

cors_origins = [
    "localhost",
    "localhost:8080",
    "localhost:3000",
]


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


app.include_router(router_cell)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
