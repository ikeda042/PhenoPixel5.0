import pytest
from httpx import AsyncClient
import os
from unittest.mock import patch
import base64

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
    username = "admin"
    password = "test"
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    headers = {"Authorization": f"Basic {encoded_credentials}"}

    with patch("Auth.crud.Auth.get_account", return_value="admin"), patch(
        "Auth.crud.Auth.verify_password", return_value=True
    ):
        response = await client.get("/api/dropbox/access_token", headers=headers)
        assert response.status_code == 200
