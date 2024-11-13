import pytest
from httpx import AsyncClient
import os
from unittest.mock import patch


os.chdir(os.path.dirname(os.path.abspath("..")))


@pytest.mark.anyio
async def test_protected_route(client: AsyncClient):
    with patch("Auth.crud.Auth.verify_password", return_value=True):
        response = await client.post(
            "/api/auth/login",
            params={
                "plain_password": "test",
                "hashed_password": "hashed_test",
            },
        )
        assert response.status_code == 200
        assert response.json() == {"is_verified": True}
