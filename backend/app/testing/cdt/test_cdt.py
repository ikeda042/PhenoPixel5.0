import pytest
from httpx import AsyncClient
from unittest.mock import patch


@pytest.mark.anyio
async def test_cdt_nagg(client: AsyncClient):
    with patch("CDT.crud.CdtCrudBase.set_ctrl", return_value=None), patch(
        "CDT.crud.CdtCrudBase.analyze_files",
        return_value=[{"filename": "sample.csv", "mean_length": 1.0, "nagg_rate": 0.5}],
    ):
        files = [
            ("ctrl_file", ("ctrl.csv", "ctrl")),
            ("files", ("sample.csv", "sample")),
        ]
        response = await client.post("/api/cdt/nagg", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data[0]["filename"] == "sample.csv"
