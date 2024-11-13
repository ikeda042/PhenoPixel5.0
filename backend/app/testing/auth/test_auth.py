import pytest
from httpx import AsyncClient
import os
from unittest.mock import patch


os.chdir(os.path.dirname(os.path.abspath("..")))


@pytest.mark.anyio
async def test_protected_route(client: AsyncClient):
    with patch("Auth.crud.Auth.get_account", return_value="ikeda042"):
        response = await client.get("/api/auth/login")
        assert response.status_code == 200
