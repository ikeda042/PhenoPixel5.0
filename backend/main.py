from fastapi import FastAPI
import uvicorn
from crud import CellCrudBase

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


# define a global var called db_name
db_name = "test_database.db"


@app.get("/cells")
async def read_cell_ids(label: str | None = None):
    return await CellCrudBase().read_cell_ids(dbname=db_name, label=label)


@app.get("/cells/{cell_id}")
async def get_cell(cell_id: str):
    return await CellCrudBase().get_cell_ph(dbname=db_name, cell_id=cell_id)


# @app.get("/cells/label/{label}")
# async def get_cells_by_label(label: str):
#     return await get_cells_with_label(dbname=db_name, label=label)


# @app.get("/cells/label/{label}/count")
# async def count_cells_by_label(label: str):
#     return await count_cells_with_label(db_name, label=label)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
