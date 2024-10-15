import os
import sys

# backend ディレクトリをモジュール検索パスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pytest
from httpx import AsyncClient
from backend.app.main import app  # FastAPI アプリケーションをインポート


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def client():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client
