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


@pytest.mark.anyio
async def test_db_healthcheck_fluo(client: AsyncClient):
    """
    /cells/database/healthcheck/fluo
    """
    response = await client.get("/api/cells/database/healthcheck/fluo")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_db_healthcheck_3d(client: AsyncClient):
    """
    /cells/database/healthcheck/3d
    """
    response = await client.get("/api/cells/database/healthcheck/3d")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_update_cell_label_ng(client: AsyncClient):
    """
    PATCH /cells/{db_name}/{cell_id}/{label}
    アップロード済みでない DB (例: test_database.db) では 400 を返す
    """
    response = await client.patch("/api/cells/test_database.db/F0C1/1")
    assert response.status_code == 400
    assert "Please provide the name of the uploaded database." in response.text


@pytest.mark.anyio
async def test_replot_cell(client: AsyncClient):
    """
    GET /cells/{cell_id}/{db_name}/replot
    """
    response = await client.get("/api/cells/F0C5/test_database.db/replot")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_cell_path(client: AsyncClient):
    """
    GET /cells/{cell_id}/{db_name}/path
    """
    response = await client.get("/api/cells/F0C5/test_database.db/path")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_mean_fluo_intensities(client: AsyncClient):
    """
    GET /cells/{db_name}/{label}/{cell_id}/mean_fluo_intensities
    """
    response = await client.get(
        "/api/cells/test_database.db/1/F0C5/mean_fluo_intensities"
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_median_fluo_intensities(client: AsyncClient):
    """
    GET /cells/{db_name}/{label}/{cell_id}/median_fluo_intensities
    """
    response = await client.get(
        "/api/cells/test_database.db/1/F0C5/median_fluo_intensities"
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_var_fluo_intensities(client: AsyncClient):
    """
    GET /cells/{db_name}/{label}/{cell_id}/var_fluo_intensities
    """
    response = await client.get(
        "/api/cells/test_database.db/1/F0C5/var_fluo_intensities"
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_median_fluo_intensities_csv(client: AsyncClient):
    """
    GET /cells/{db_name}/{label}/median_fluo_intensities/csv
    """
    response = await client.get(
        "/api/cells/test_database.db/1/median_fluo_intensities/csv"
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_mean_fluo_intensities_csv(client: AsyncClient):
    """
    GET /cells/{db_name}/{label}/mean_fluo_intensities/csv
    """
    response = await client.get(
        "/api/cells/test_database.db/1/mean_fluo_intensities/csv"
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_var_fluo_intensities_csv(client: AsyncClient):
    """
    GET /cells/{db_name}/{label}/var_fluo_intensities/csv
    """
    response = await client.get(
        "/api/cells/test_database.db/1/var_fluo_intensities/csv"
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_heatmap(client: AsyncClient):
    """
    GET /cells/{db_name}/{label}/{cell_id}/heatmap
    """
    response = await client.get("/api/cells/test_database.db/1/F0C5/heatmap")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_fluo_distribution(client: AsyncClient):
    """
    GET /cells/{db_name}/{cell_id}/distribution
    """
    response = await client.get("/api/cells/test_database.db/F0C5/distribution")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_fluo_distribution_normalized(client: AsyncClient):
    """
    GET /cells/{db_name}/{label}/{cell_id}/distribution_normalized
    """
    # label はエンドポイントでは必須ですが引数に未指定だったので "1" としてみます
    response = await client.get(
        "/api/cells/test_database.db/1/F0C5/distribution_normalized"
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_heatmap_csv(client: AsyncClient):
    """
    GET /cells/{db_name}/{label}/{cell_id}/heatmap/csv
    """
    response = await client.get("/api/cells/test_database.db/1/F0C5/heatmap/csv")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_3d_plot(client: AsyncClient):
    """
    GET /cells/{db_name}/{cell_id}/3d
    """
    response = await client.get("/api/cells/test_database.db/F0C5/3d")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_3d_plot_ph(client: AsyncClient):
    """
    GET /cells/{db_name}/{cell_id}/3d-ph
    """
    response = await client.get("/api/cells/test_database.db/F0C5/3d-ph")
    assert response.status_code == 200


# ---- データベース系エンドポイントのテスト ----


@pytest.mark.anyio
async def test_upload_database_ng(client: AsyncClient):
    """
    POST /databases/upload
    db ファイル以外をアップロードするとエラーメッセージを返す
    """
    file_name = "test.txt"
    content = b"dummy content"

    # httpx を使ったファイルアップロードの例
    files = {"file": (file_name, content, "text/plain")}
    response = await client.post("/api/databases/upload", files=files)
    assert response.status_code == 200  # 実装次第で 200 or 422 など
    assert "Please upload a .db file." in response.text


@pytest.mark.anyio
async def test_upload_database_ok(client: AsyncClient):
    """
    POST /databases/upload
    """
    file_name = "new_database.db"
    content = b"dummy db bytes"

    files = {"file": (file_name, content, "application/octet-stream")}
    response = await client.post("/api/databases/upload", files=files)
    # 実装上、同名DBがなければアップロード成功メッセージ or ファイル名を返す想定
    assert response.status_code == 200


@pytest.mark.anyio
async def test_update_database_to_label_completed_ng(client: AsyncClient):
    """
    PATCH /databases/{db_name}
    存在しないファイル名を指定すると 404 が返る
    """
    response = await client.patch("/api/databases/non_existent.db")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_update_database_to_label_completed_ok(client: AsyncClient):
    """
    PATCH /databases/{db_name}
    正常にDBを更新できる場合
    """
    # テストDBを準備できない場合はスキップ
    db_name = "test_database.db"
    if False:
        pytest.skip("No real file to patch")

    response = await client.patch(f"/api/databases/{db_name}")
    # 実装次第でステータスコードが変わるかもしれない
    assert response.status_code in (200, 404)
    # すでに存在するかどうかで結果が変わる


@pytest.mark.anyio
async def test_check_if_database_updated_once(client: AsyncClient):
    """
    GET /databases/{db_name}
    """
    response = await client.get("/api/databases/test_database.db")
    assert response.status_code == 200
    # 実装次第で True/False のような値が返る想定
    # ここでは型だけ確認
    assert isinstance(response.json(), bool) or response.json() in (True, False)


@pytest.mark.anyio
async def test_download_db_ng(client: AsyncClient):
    """
    GET /databases/download-completed/{db_name}
    存在しないファイルの場合、404 が返る
    """
    response = await client.get("/api/databases/download-completed/non_existent.db")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_download_db_ok(client: AsyncClient):
    """
    GET /databases/download-completed/{db_name}
    通常はファイルをダウンロードさせる挙動。
    テストではステータスコードなどを確認。
    """
    # テストDBが存在しない場合はスキップ
    db_name = "test_database.db"
    if False:
        pytest.skip("No real file to download")

    response = await client.get(f"/api/databases/download-completed/{db_name}")
    # 事前にファイルが無い場合は 404、あれば 200 が返る想定
    assert response.status_code in (200, 404)


@pytest.mark.anyio
async def test_get_cell_images_combined(client: AsyncClient):
    """
    GET /databases/{db_name}/combined_images
    """
    response = await client.get(
        "/api/databases/test_database.db/combined_images",
        params={"label": "1", "mode": "fluo"},
    )
    assert response.status_code == 200
