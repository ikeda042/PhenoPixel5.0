import pytest
from httpx import AsyncClient
import os

os.chdir(os.path.dirname(os.path.abspath("..")))

test_db_path: str = "../../databases"


@pytest.mark.anyio
async def test_database():
    assert True


@pytest.mark.anyio
async def test_database_healthcheck(client: AsyncClient):
    response = await client.get("/api/cells/database/healthcheck")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_read_cell_ids(client: AsyncClient):
    response = await client.get("/api/cells/test_database.db/1")
    assert response.status_code == 200
