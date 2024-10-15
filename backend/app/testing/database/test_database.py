import pytest
from httpx import AsyncClient
import os


test_db_path: str = "../../databases/test_database.db"


@pytest.mark.anyio
async def test_database():
    assert True


@pytest.mark.anyio
async def test_database_healthcheck(client: AsyncClient):
    response = await client.get("/api/cells/database/healthcheck")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_database_read_cell_ids(client: AsyncClient):
    # http://localhost:8000/api/cells/test_database.db/1
    response = await client.get("/api/cells/.../databases/test_database.db/1")
    print(os.listdir("../../databases"))
    assert response.status_code == 200
