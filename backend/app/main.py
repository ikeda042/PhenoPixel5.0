from fastapi import FastAPI
import uvicorn
from CellDBConsole.router import router_cell
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CellAPI", docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json"
)

cors_origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


app.include_router(router_cell)
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_certfile="./cert.pem",
        ssl_keyfile="./key.pem",
    )
