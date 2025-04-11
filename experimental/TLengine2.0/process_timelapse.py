import numpy as np
import os
import nd2reader
from PIL import Image
import cv2
import ulid


def process_image(array: np.ndarray) -> np.ndarray:
    """
    画像処理関数：正規化とスケーリングを行う。
    """
    array = array.astype(np.float32)  # Convert to float
    array -= array.min()  # Normalize to 0
    array /= array.max()  # Normalize to 1
    array *= 255  # Scale to 0-255
    return array.astype(np.uint8)


def extract_nd2(file_name: str):
    """
    タイムラプスnd2ファイルをフレームごとにTIFF形式で保存する。
    チャンネル0: ph
    チャンネル1: fluo1
    チャンネル2: fluo2
    """
    base_output_dir = os.path.join(
        "experimental", "TLengine2.0", "output", file_name.split("/")[-1].split(".")[0]
    )
    os.makedirs(base_output_dir, exist_ok=True)

    with nd2reader.ND2Reader(file_name) as images:
        print(f"Available axes: {images.axes}")
        print(f"Sizes: {images.sizes}")

        # (channel, y, x) の順に束ねる
        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        # フィールド(視野)ごとに処理
        images.iter_axes = "v"

        num_fields = images.sizes.get("v", 1)
        num_channels = images.sizes.get("c", 1)
        num_timepoints = images.sizes.get("t", 1)

        for field_idx in range(num_fields):
            # Fieldごとのフォルダを作成
            field_folder = os.path.join(base_output_dir, f"Field_{field_idx + 1}")
            os.makedirs(field_folder, exist_ok=True)

            # チャンネルごとのサブフォルダを作成
            ph_folder = os.path.join(field_folder, "ph")
            fluo1_folder = os.path.join(field_folder, "fluo1")
            fluo2_folder = os.path.join(field_folder, "fluo2")
            os.makedirs(ph_folder, exist_ok=True)
            os.makedirs(fluo1_folder, exist_ok=True)
            os.makedirs(fluo2_folder, exist_ok=True)

            for time_idx in range(num_timepoints):
                images.default_coords.update({"v": field_idx, "t": time_idx})
                image_data = images[0]

                if len(image_data.shape) == 2:
                    # チャンネルが1つのみの場合
                    ph_image = process_image(image_data)
                    tiff_filename = os.path.join(
                        ph_folder, f"time_{time_idx + 1}_channel_0.tif"
                    )
                    Image.fromarray(ph_image).save(tiff_filename)
                    print(f"Saved: {tiff_filename}")
                else:
                    # 複数チャンネルの場合
                    for i in range(image_data.shape[0]):
                        channel_image = process_image(image_data[i])
                        if i == 0:
                            folder = ph_folder
                            channel_name = "ph"
                        elif i == 1:
                            folder = fluo1_folder
                            channel_name = "fluo1"
                        elif i == 2:
                            folder = fluo2_folder
                            channel_name = "fluo2"
                        else:
                            folder = field_folder
                            channel_name = f"channel_{i}"

                        tiff_filename = os.path.join(
                            folder, f"time_{time_idx + 1}_{channel_name}.tif"
                        )
                        Image.fromarray(channel_image).save(tiff_filename)
                        print(f"Saved: {tiff_filename}")


def get_ph_image(file_name: str, field_idx: int, time_idx: int) -> np.ndarray:
    """
    指定したファイル・Field・Timeからph (channel=0) の画像を取得し、
    process_image を通して正規化した numpy.ndarray を返す関数。
    """
    with nd2reader.ND2Reader(file_name) as images:
        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        images.iter_axes = "v"
        images.default_coords.update({"v": field_idx, "t": time_idx})
        image_data = images[0]

        if len(image_data.shape) == 3:
            # channel軸がある場合は channel=0 (ph) を取り出す
            ph_image_data = image_data[0]
        else:
            # channelが1つしかない場合
            ph_image_data = image_data

        return process_image(ph_image_data)


def shift_image(img: np.ndarray, vertical: int, horizontal: int) -> np.ndarray:
    """
    画像を縦(vertical), 横(horizontal)ピクセル数だけずらす。
    符号によって上下左右を決める。
    """
    # 変換行列を作成
    # 正: 下方向 or 右方向, 負: 上方向 or 左方向
    M = np.float32([
        [1, 0, horizontal],  # x方向シフト
        [0, 1, vertical]     # y方向シフト
    ])
    # warpAffineでシフト
    shifted_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted_img


def shift_ph_image(file_name: str, field_idx: int, time_idx: int,
                   vertical_shift: int, horizontal_shift: int) -> np.ndarray:
    """
    指定したファイル・Field・Timeからph画像を取得し、
    指定ピクセル分だけ縦横にずらした画像を返す。
    """
    ph_img = get_ph_image(file_name, field_idx, time_idx)
    shifted_ph = shift_image(ph_img, vertical_shift, horizontal_shift)
    return shifted_ph


if __name__ == "__main__":
    # 実行テスト用のスクリプト
    # まずはND2を分解(必要なら):
    filename = "experimental/TLengine2.0/sk450gen120min-tl.nd2"
    # extract_nd2(filename)  # 既に実行済みならコメントアウトしてOK

    # 任意のField, Time, シフト量を指定
    test_field = 0       # 1つ目のField → Pythonで言えばindex = 0
    test_time = 0        # 1つ目のTime  → Pythonで言えばindex = 0
    shift_vertical = 100  # 下に10ピクセル
    shift_horizontal = -50  # 左に5ピクセル

    # シフトしたph画像を取得
    shifted_ph_image = shift_ph_image(
        filename,
        field_idx=test_field,
        time_idx=test_time,
        vertical_shift=shift_vertical,
        horizontal_shift=shift_horizontal
    )

    # 保存先を指定して保存
    output_path = f"shifted_ph_{test_field}.tif"
    Image.fromarray(shifted_ph_image).save(output_path)
    print(f"Shifted image saved to: {output_path}")
