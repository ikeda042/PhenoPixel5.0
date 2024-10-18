import pytest
from httpx import AsyncClient
import os
import asyncio

os.chdir(os.path.dirname(os.path.abspath("..")))


@pytest.mark.anyio
async def test_dropbox():
    assert True


@pytest.mark.anyio
async def test_dropbox_healthcheck(client: AsyncClient):
    response = await client.get("/api/dropbox/connection_check")
    assert response.status_code == 200
    assert response.json()["status"] == True


@pytest.mark.anyio
async def test_dropbox_access_token(client: AsyncClient):
    response = await client.get("/api/dropbox/access_token")
    assert response.status_code == 200
