import pytest
from httpx import AsyncClient, ASGITransport
from main import app


@pytest.fixture(scope="session")
def anyio_backend():
    """
    See https://anyio.readthedocs.io/en/stable/testing.html#using-async-fixtures-with-higher-scopes
    """
    return "asyncio"


@pytest.fixture(scope="session")
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client
