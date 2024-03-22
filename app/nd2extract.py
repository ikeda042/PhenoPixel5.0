import nd2reader
import numpy as np
from PIL import Image
import os

def extract_nd2(file_name: str):
    """
    nd2ファイルをMultipageTIFFに変換する。
    """
    try:
        os.mkdir("nd2totiff")
    except FileExistsError:
        pass

    with nd2reader.ND2Reader(file_name) as images:
        # 利用可能な軸とサイズをチェック
        print(f"Available axes: {images.axes}")
        print(f"Sizes: {images.sizes}")

        # チャンネル情報があるかどうかに基づいて軸を設定
        images.bundle_axes = 'cyx' if 'c' in images.axes else 'yx'
        images.iter_axes = 'v'

        # チャンネル数を取得（なければデフォルトで1）
        num_channels = images.sizes.get('c', 1)
        print(f"Total images: {len(images)}")
        print(f"Channels: {num_channels}")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        for n, img in enumerate(images):
            if num_channels > 1:
                for channel in range(num_channels):
                    array = np.array(img[channel])
                    array = process_image(array)
                    image = Image.fromarray(array)
                    image.save(f"nd2totiff/image_{n}_channel_{channel}.tif")
            else:
                array = np.array(img)
                array = process_image(array)
                image = Image.fromarray(array)
                image.save(f"nd2totiff/image_{n}.tif")

        all_images = []
        for i in range(len(images)):
            if num_channels > 1:
                for j in range(num_channels):
                    all_images.append(Image.open(f"nd2totiff/image_{i}_channel_{j}.tif"))
            else:
                all_images.append(Image.open(f"nd2totiff/image_{i}.tif"))

        all_images[0].save(
            f"{file_name.split('/')[-1].split('.')[0]}.tif",
            save_all=True,
            append_images=all_images[1:]
        )

def process_image(array):
    """
    画像処理関数：正規化とスケーリングを行う。
    """
    array = array.astype(np.float32)  # Convert to float
    array -= array.min()  # Normalize to 0
    array /= array.max()  # Normalize to 1
    array *= 255  # Scale to 0-255
    return array.astype(np.uint8) 