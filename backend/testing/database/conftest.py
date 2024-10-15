import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    """
    See https://anyio.readthedocs.io/en/stable/testing.html#using-async-fixtures-with-higher-scopes
    """
    return "asyncio"
