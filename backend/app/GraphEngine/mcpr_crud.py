from fastapi import FastAPI, UploadFile, Query
from fastapi.responses import FileResponse
from typing import List
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import cv2
import numpy as np
import asyncio
import concurrent.futures
import time

app = FastAPI()


async def _draw_graph_from_memory(
    csv_file: UploadFile, blank_index: str, timespan_sec: int = 180
) -> List[bytes]:
    """
    CSVファイル(UploadFile)からDataFrameを読み込み、グラフを作成して
    各画像をPNG形式のバイナリデータ(bytes)としてリストで返します。
    """
    # CSVをDataFrameに読み込み
    df = pd.read_csv(csv_file.file, encoding="unicode_escape")

    # 以下、元のロジック
    start_index = 0
    end_index = 0
    temp_detected = False
    for i in range(df.shape[0] - 1):
        if "Temp" in str(df.iloc[i][0]):
            start_index = i
            temp_detected = True
        if (
            type(df.iloc[i + 1][0]) == float
            and type(df.iloc[i][0]) == str
            and temp_detected
        ):
            end_index = i + 1
            break

    data = {}
    for i in range(start_index + 1, end_index):
        data[df.iloc[i][0][0]] = []
    for i in range(start_index + 1, end_index):
        data[df.iloc[i][0][0]].append(
            {df.iloc[i][0]: [float(j) for j in df.iloc[i][1:-1]]}
        )

    x = [i * timespan_sec / 3600 for i in range(len(df.iloc[start_index - 2]) - 2)]

    blank_index = blank_index.replace(" ", "")
    all_keys = [
        key[1:]
        for sublist in [[list(i.keys())[0] for i in data[j]] for j in list(data.keys())]
        for key in sublist
    ]

    if blank_index not in all_keys:
        raise ValueError(f"Key {blank_index} not found in data")

    image_bytes_list = []
    for graph_idx, key_char in enumerate(list(data.keys())):
        time.sleep(1)
        fig = plt.figure(figsize=[5, 5])
        sns.set()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.major.width"] = 1.0
        plt.rcParams["ytick.major.width"] = 1.0
        plt.rcParams["font.size"] = 9
        plt.rcParams["axes.linewidth"] = 1.0
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        plt.xlabel("time(h)")
        plt.ylabel("OD600(-)")
        for i in data[key_char]:
            key = [j for j in i.keys()][0]
            y = i[key]
            plt.scatter(x, y, label=f"{key}", s=6)
        plt.legend(title="Series")
        buf = io.BytesIO()
        fig.savefig(buf, dpi=500, format="png")
        plt.close(fig)
        buf.seek(0)
        image_bytes_list.append(buf.read())

    return image_bytes_list


def _blocking_combine_images_in_memory(image_bytes: List[bytes], per_row: int) -> bytes:
    """
    メモリ上のPNGバイナリデータリストを結合して1枚の画像(bytes)として返す。
    """
    # PNGバイナリ -> numpy配列 (BGR画像) に変換
    images = []
    for img_data in image_bytes:
        arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        images.append(img)

    max_width = max(img.shape[1] for img in images)
    max_height = max(img.shape[0] for img in images)
    num_images = len(images)
    num_rows = num_images // per_row
    if num_images % per_row != 0:
        num_rows += 1

    final_image = np.zeros(
        (num_rows * max_height, per_row * max_width, 3), dtype=np.uint8
    )

    for i, img in enumerate(images):
        top = (i // per_row) * max_height
        left = (i % per_row) * max_width
        final_image[top : top + img.shape[0], left : left + img.shape[1]] = img

    # 結合画像をバイナリにエンコード
    _, encoded_img = cv2.imencode(".png", final_image)
    return encoded_img.tobytes()


async def combine_images_in_memory(image_bytes: List[bytes], per_row: int) -> bytes:
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, _blocking_combine_images_in_memory, image_bytes, per_row
        )
    return result
