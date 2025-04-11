import numpy as np
import os
import nd2reader
from PIL import Image
import cv2


def process_image(array):
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
    base_output_dir = "timelapsen2dtotiff"

    os.makedirs(base_output_dir, exist_ok=True)

    with nd2reader.ND2Reader(file_name) as images:
        print(f"Available axes: {images.axes}")
        print(f"Sizes: {images.sizes}")

        # もし 'c' が axes に含まれていれば、 (channel, y, x) の順に束ねる
        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        # フィールド(視野)ごとに取り出す
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
                # ND2Reader で読み込む際の座標指定
                # channel, time のループは下記イテレータにより内側で処理する
                images.default_coords.update(
                    {"v": field_idx, "t": time_idx}
                )
                # ここで取り出す image_data は、(チャンネル数, y, x) の3次元配列になる想定
                image_data = images[0]

                # チャンネルが一つしかない場合は2次元配列になるため、対応
                if len(image_data.shape) == 2:
                    # チャンネル0相当として処理
                    ph_image = process_image(image_data)
                    tiff_filename = os.path.join(
                        ph_folder, f"time_{time_idx + 1}_channel_0.tif"
                    )
                    Image.fromarray(ph_image).save(tiff_filename)
                    print(f"Saved: {tiff_filename}")
                else:
                    # 複数チャンネル (例: 3チャンネル) がある場合
                    for i in range(image_data.shape[0]):
                        channel_image = process_image(image_data[i])
                        # i=0 → ph, i=1 → fluo1, i=2 → fluo2
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
                            # もしチャンネルが3以上の場合に備えるなら、
                            # フォルダ名や処理を追加するか無視するなど設計次第
                            folder = field_folder
                            channel_name = f"channel_{i}"

                        tiff_filename = os.path.join(
                            folder,
                            f"time_{time_idx + 1}_{channel_name}.tif",
                        )
                        Image.fromarray(channel_image).save(tiff_filename)
                        print(f"Saved: {tiff_filename}")


if __name__ == "__main__":
    filename = "experimental/TLengine2.0/sk450gen120min-tl.nd2"
    extract_nd2(filename)
