import numpy as np
import os
import nd2reader
from PIL import Image
import cv2


def correct_drift(reference_image, target_image):
    """
    基準フレームと比較して、対象フレームのドリフトを補正する。
    """
    # ORB（特徴量検出）を用いて、フレーム間の特徴点をマッチング
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(reference_image, None)
    kp2, des2 = orb.detectAndCompute(target_image, None)

    # BFMatcherで特徴点をマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # マッチング点を基に、フレーム間の変換行列を計算
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # 対象フレームを基準フレームに合わせて変換（位置補正）
    aligned_image = cv2.warpAffine(
        target_image, matrix, (target_image.shape[1], target_image.shape[0])
    )

    return aligned_image


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
    """
    base_output_dir = "timelapsen2dtotiff"

    os.makedirs(base_output_dir, exist_ok=True)

    with nd2reader.ND2Reader(file_name) as images:
        print(f"Available axes: {images.axes}")
        print(f"Sizes: {images.sizes}")

        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        images.iter_axes = "v"

        num_fields = images.sizes.get("v", 1)
        num_channels = images.sizes.get("c", 1)
        num_timepoints = images.sizes.get("t", 1)

        for field_idx in range(num_fields):
            field_folder = os.path.join(base_output_dir, f"Field_{field_idx + 1}")
            os.makedirs(field_folder, exist_ok=True)
            base_output_subdir_ph = field_folder + "/ph"
            base_output_subdir_fluo = field_folder + "/fluo"
            os.makedirs(base_output_subdir_ph, exist_ok=True)
            os.makedirs(base_output_subdir_fluo, exist_ok=True)

            for channel_idx in range(num_channels):
                for time_idx in range(num_timepoints):
                    images.default_coords.update(
                        {"v": field_idx, "c": channel_idx, "t": time_idx}
                    )
                    image_data = images[0]

                    if len(image_data.shape) == 3:
                        for i in range(image_data.shape[0]):
                            channel_image = process_image(image_data[i])
                            tiff_filename = os.path.join(
                                (
                                    base_output_subdir_ph
                                    if i == 1
                                    else base_output_subdir_fluo
                                ),
                                f"time_{time_idx + 1}_channel_{i}.tif",
                            )
                            img = Image.fromarray(channel_image)
                            img.save(tiff_filename)
                            print(f"Saved: {tiff_filename}")
                    else:
                        image_data = process_image(image_data)
                        tiff_filename = os.path.join(
                            field_folder, f"time_{time_idx + 1}.tif"
                        )
                        img = Image.fromarray(image_data)
                        img.save(tiff_filename)
                        print(f"Saved: {tiff_filename}")


def create_combined_gif(field_folder: str, output_gif: str):
    """
    Field1のphとfluo画像を左右に並べて時系列順にGIFにする。
    """
    ph_folder = os.path.join(field_folder, "ph")
    fluo_folder = os.path.join(field_folder, "fluo")

    ph_image_files = sorted(
        [
            os.path.join(ph_folder, f)
            for f in os.listdir(ph_folder)
            if f.endswith(".tif")
        ]
    )
    fluo_image_files = sorted(
        [
            os.path.join(fluo_folder, f)
            for f in os.listdir(fluo_folder)
            if f.endswith(".tif")
        ]
    )

    # 画像を読み込む
    ph_images = [Image.open(img_file) for img_file in ph_image_files]
    fluo_images = [Image.open(img_file) for img_file in fluo_image_files]

    # 画像を左右に並べる
    combined_images = []
    print("####################GIF####################")
    for ph_img, fluo_img in zip(ph_images, fluo_images):
        combined_width = ph_img.width + fluo_img.width
        print(ph_img.height, fluo_img.height)
        combined_height = max(ph_img.height, fluo_img.height)
        combined_img = Image.new("RGB", (combined_width, combined_height))
        combined_img.paste(ph_img, (0, 0))
        combined_img.paste(fluo_img, (ph_img.width, 0))
        combined_images.append(combined_img)

    # GIFを保存
    combined_images[0].save(
        output_gif,
        save_all=True,
        append_images=combined_images[1:],
        duration=100,  # 各フレームの表示時間（ミリ秒）
        loop=0,  # 無限ループ
    )

    print(f"GIF saved: {output_gif}")


# テストファイル名
filename = "testdata.nd2"
extract_nd2(filename)

field_folder = "timelapsen2dtotiff/Field_1"
output_gif = "Field_1_ph_fluo_timelapse.gif"
create_combined_gif(field_folder, output_gif)
