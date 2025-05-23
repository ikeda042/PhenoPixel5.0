from fastapi import FastAPI, UploadFile, Query
from fastapi.responses import StreamingResponse
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


def _draw_graph_from_memory(
    csv_file: UploadFile,
    blank_index: str,
    timespan_sec: int = 180,
    lower_OD: float = 0.1,
    upper_OD: float = 0.3,
) -> List[bytes]:
    csv_file.file.seek(0)
    df = pd.read_csv(csv_file.file, encoding="unicode_escape")

    start_index = 0
    end_index = 0
    temp_detected = False
    for i in range(df.shape[0] - 1):
        if "Temp" in str(df.iloc[i][0]):
            start_index = i
            temp_detected = True
        if (
            isinstance(df.iloc[i + 1][0], float)
            and isinstance(df.iloc[i][0], str)
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

        for i_dict in data[key_char]:
            key = list(i_dict.keys())[0]
            y = i_dict[key]

            x_arr = np.array(x)
            y_arr = np.array(y)

            mask = (y_arr >= lower_OD) & (y_arr <= upper_OD)
            x_sub = x_arr[mask]
            y_sub = y_arr[mask]

            mu_value = None
            if len(x_sub) > 1:
                ln_y_sub = np.log(y_sub)
                slope, intercept = np.polyfit(x_sub, ln_y_sub, 1)
                mu_value = slope

            if mu_value is not None:
                plt.scatter(x_arr, y_arr, label=f"{key} (μ={mu_value:.3f}/h)", s=6)
                x_fit = np.linspace(x_sub.min(), x_sub.max(), 50)
                y_fit = np.exp(slope * x_fit + intercept)
                plt.plot(x_fit, y_fit, color="red", lw=1.5)

            else:
                plt.scatter(x_arr, y_arr, label=f"{key} (no fit)", s=6)

        plt.legend(title="Series")
        buf = io.BytesIO()
        fig.savefig(buf, dpi=300, format="png")
        plt.close(fig)
        buf.seek(0)
        image_bytes_list.append(buf.read())

    return image_bytes_list


def _blocking_combine_images_in_memory(image_bytes: List[bytes], per_row: int) -> bytes:
    images = []
    for img_data in image_bytes:
        arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        images.append(img)

    max_width = max(img.shape[1] for img in images)
    max_height = max(img.shape[0] for img in images)

    num_images = len(images)
    num_cols = num_images // per_row
    if num_images % per_row != 0:
        num_cols += 1

    final_image = np.full(
        (per_row * max_height, num_cols * max_width, 3), 255, dtype=np.uint8  # 白
    )

    for i, img in enumerate(images):
        row_idx = i % per_row
        col_idx = i // per_row

        top = row_idx * max_height
        left = col_idx * max_width
        final_image[top : top + img.shape[0], left : left + img.shape[1]] = img

    _, encoded_img = cv2.imencode(".png", final_image)
    return encoded_img.tobytes()


async def combine_images_in_memory(image_bytes: List[bytes], per_row: int) -> bytes:
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, _blocking_combine_images_in_memory, image_bytes, per_row
        )
    return result


@app.post("/upload-and-combine/")
async def upload_and_combine(
    files: List[UploadFile],
    blank_index: str = Query(...),
    per_row: int = Query(2),  # ここで1列あたりの画像数（縦方向の画像数）を指定
):
    """
    複数の CSV を受け取り、グラフ化してまとめて返すエンドポイント
    """
    # まず CSV を順番にグラフ化
    all_image_bytes = []
    for file in files:
        images = _draw_graph_from_memory(file, blank_index)
        all_image_bytes.extend(images)

    # 生成したグラフを一つに結合
    combined_image = await combine_images_in_memory(all_image_bytes, per_row)

    return StreamingResponse(io.BytesIO(combined_image), media_type="image/png")
