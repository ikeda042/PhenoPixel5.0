import requests
import os

def main():
    # --- 設定項目 ---
    BASE_URL = "http://localhost:8000/api/tlengine"  # FastAPIサーバが動いているURL
    DB_NAME = "my_database_cells.db"            # ダウンロードしたいDB名
    OUTPUT_DIR = "downloaded_timecourses"       # 画像を保存するディレクトリ

    # 出力先フォルダを作成 (存在しなければ)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) データベース内のフィールド一覧を取得
    #    GET /tlengine/databases/{db_name}/fields
    url_fields = f"{BASE_URL}/databases/{DB_NAME}/fields"
    resp_fields = requests.get(url_fields)
    resp_fields.raise_for_status()  # ステータスコードがエラーなら例外
    fields = resp_fields.json()["fields"]

    print(f"Found fields: {fields}")

    # 2) 各フィールドに対してセル番号一覧を取得
    for field in fields:
        url_cellnums = f"{BASE_URL}/databases/{DB_NAME}/fields/{field}/cell_numbers"
        resp_cellnums = requests.get(url_cellnums)
        resp_cellnums.raise_for_status()
        cell_numbers = resp_cellnums.json()["cell_numbers"]

        print(f"Field={field}, cell_numbers={cell_numbers}")

        # 3) 各セル番号について
        for cell_num in cell_numbers:
            # 例: GET /databases/{db_name}/cells/{field}/{cell_number}/timecourse_png/all_channels
            url_timecourse = (
                f"{BASE_URL}/databases/{DB_NAME}/cells/{field}/{cell_num}/timecourse_png/all_channels"
            )

            # 取得 (stream=True でレスポンスを逐次受信)
            print(f"Downloading PNG for field={field}, cell={cell_num} ...")
            resp_png = requests.get(url_timecourse, stream=True)
            if resp_png.status_code != 200:
                print(f"Error or no data: {resp_png.status_code}")
                continue  # スキップ or エラー処理

            # ダウンロード先ファイル名を作成
            filename = f"{DB_NAME}_{field}_{cell_num}_all_channels.png"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # レスポンスデータをファイルに書き込み
            with open(filepath, "wb") as f:
                for chunk in resp_png.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"Saved: {filepath}")

if __name__ == "__main__":
    main()
