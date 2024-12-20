import pytest
from httpx import AsyncClient
import os
import asyncio

os.chdir(os.path.dirname(os.path.abspath("..")))


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
    response_template = [
        {"cell_id": "F0C1"},
        {"cell_id": "F0C2"},
        {"cell_id": "F0C3"},
        {"cell_id": "F0C7"},
        {"cell_id": "F0C9"},
        {"cell_id": "F0C11"},
        {"cell_id": "F0C16"},
        {"cell_id": "F0C18"},
        {"cell_id": "F0C20"},
        {"cell_id": "F0C21"},
        {"cell_id": "F0C24"},
        {"cell_id": "F0C27"},
        {"cell_id": "F1C0"},
        {"cell_id": "F1C2"},
        {"cell_id": "F1C3"},
        {"cell_id": "F1C6"},
        {"cell_id": "F1C9"},
        {"cell_id": "F1C10"},
        {"cell_id": "F1C11"},
        {"cell_id": "F1C12"},
        {"cell_id": "F1C13"},
        {"cell_id": "F1C20"},
        {"cell_id": "F2C1"},
        {"cell_id": "F2C2"},
        {"cell_id": "F2C5"},
        {"cell_id": "F2C6"},
        {"cell_id": "F2C7"},
        {"cell_id": "F2C11"},
        {"cell_id": "F2C14"},
        {"cell_id": "F2C18"},
        {"cell_id": "F2C19"},
        {"cell_id": "F3C1"},
        {"cell_id": "F3C2"},
        {"cell_id": "F3C8"},
        {"cell_id": "F3C12"},
        {"cell_id": "F3C15"},
        {"cell_id": "F3C19"},
        {"cell_id": "F3C25"},
        {"cell_id": "F3C27"},
        {"cell_id": "F3C28"},
        {"cell_id": "F4C1"},
        {"cell_id": "F4C5"},
        {"cell_id": "F4C8"},
        {"cell_id": "F4C18"},
        {"cell_id": "F5C1"},
        {"cell_id": "F5C3"},
        {"cell_id": "F5C5"},
        {"cell_id": "F5C6"},
        {"cell_id": "F5C7"},
        {"cell_id": "F5C8"},
        {"cell_id": "F5C10"},
        {"cell_id": "F5C11"},
        {"cell_id": "F5C19"},
        {"cell_id": "F5C20"},
        {"cell_id": "F5C21"},
        {"cell_id": "F5C23"},
        {"cell_id": "F5C25"},
        {"cell_id": "F6C0"},
        {"cell_id": "F6C2"},
        {"cell_id": "F6C4"},
        {"cell_id": "F6C6"},
        {"cell_id": "F6C8"},
        {"cell_id": "F6C10"},
        {"cell_id": "F6C13"},
        {"cell_id": "F6C15"},
        {"cell_id": "F6C16"},
        {"cell_id": "F6C21"},
        {"cell_id": "F6C22"},
        {"cell_id": "F6C23"},
        {"cell_id": "F6C24"},
        {"cell_id": "F6C25"},
        {"cell_id": "F6C26"},
        {"cell_id": "F6C27"},
        {"cell_id": "F6C30"},
        {"cell_id": "F6C31"},
        {"cell_id": "F7C2"},
        {"cell_id": "F7C5"},
        {"cell_id": "F7C10"},
        {"cell_id": "F7C11"},
        {"cell_id": "F7C17"},
        {"cell_id": "F7C24"},
        {"cell_id": "F7C26"},
        {"cell_id": "F7C28"},
        {"cell_id": "F7C31"},
        {"cell_id": "F7C34"},
        {"cell_id": "F7C35"},
        {"cell_id": "F7C37"},
        {"cell_id": "F7C38"},
        {"cell_id": "F7C41"},
        {"cell_id": "F7C42"},
        {"cell_id": "F7C43"},
        {"cell_id": "F7C44"},
    ]
    assert response.json() == response_template


@pytest.mark.anyio
async def test_read_cells_with_label_1(client: AsyncClient):
    cell_ids = [
        "F0C1",
        "F0C2",
        "F0C3",
        "F0C7",
        "F0C9",
        "F0C11",
        "F0C16",
        "F0C18",
        "F0C20",
        "F0C21",
        "F0C24",
        "F0C27",
        "F1C0",
        "F1C2",
        "F1C3",
        "F1C6",
        "F1C9",
        "F1C10",
        "F1C11",
        "F1C12",
        "F1C13",
        "F1C20",
    ]

    async def fetch_cell_label(cell_id):
        response = await client.get(f"/api/cells/test_database.db/{cell_id}/label")
        assert response.status_code == 200
        label = response.json()
        assert label == 1
        return cell_id, label

    results = await asyncio.gather(*(fetch_cell_label(cell_id) for cell_id in cell_ids))

    for cell_id, label in results:
        assert label == 1


@pytest.mark.anyio
async def test_read_cell_label(client: AsyncClient):
    response = await client.get("/api/cells/test_database.db/F0C1/label")
    assert response.status_code == 200
    response_template = 1
    assert response.json() == response_template


@pytest.mark.anyio
async def test_read_cell_ph(client: AsyncClient):
    response = await client.get("/api/cells/F0C1/test_database.db/False/False/ph_image")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_read_cell_fluo(client: AsyncClient):
    response = await client.get(
        "/api/cells/F0C1/test_database.db/False/False/fluo_image"
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_read_cell_contour(client: AsyncClient):
    response = await client.get(
        "/api/cells/F0C1/contour/raw", params={"db_name": "test_database.db"}
    )
    assert response.status_code == 200
    assert "contour" in response.json()

    response = await client.get(
        "/api/cells/F0C1/contour/converted", params={"db_name": "test_database.db"}
    )
    assert response.status_code == 200
    assert "contour" in response.json()


@pytest.mark.anyio
async def test_read_cell_morphology(client: AsyncClient):
    response = await client.get(
        "/api/cells/F0C1/test_database.db/morphology", params={"polyfit_degree": 3}
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_multiple_connections(client: AsyncClient):
    async def fetch_cell_label(cell_id):
        response = await client.get(f"/api/cells/test_database.db/{cell_id}/label")
        assert response.status_code == 200
        label = response.json()
        assert label == 1
        return cell_id, label

    results = await asyncio.gather(
        *(fetch_cell_label("F0C1") for _ in range(10)), return_exceptions=True
    )

    for cell_id, label in results:
        assert label == 1


@pytest.mark.anyio
async def test_get_cell_metadata(client: AsyncClient):
    response = await client.get("/api/cells/test_database.db/metadata")
    assert response.status_code == 200
