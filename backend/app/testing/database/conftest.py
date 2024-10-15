import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pytest
from httpx import AsyncClient
from app.main import app  # FastAPI などのアプリケーションをインポート


@pytest.fixture(scope="session")
def anyio_backend():
    """
    See https://anyio.readthedocs.io/en/stable/testing.html#using-async-fixtures-with-higher-scopes
    """
    return "asyncio"


@pytest.fixture(scope="session")
async def client():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client
