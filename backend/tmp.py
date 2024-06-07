import os
from PIL import Image

import nd2reader
import numpy as np
from PIL import Image
import os
import cv2
import numpy as np


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
        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        images.iter_axes = "v"

        # チャンネル数を取得（なければデフォルトで1）
        num_channels = images.sizes.get("c", 1)
        print(f"Total images: {len(images)}")
        print(f"Channels: {num_channels}")
        print("##############################################")

        # チャンネル名を取得（なければデフォルト名を使用）
        channels = images.metadata.get("channels", ["Default"])

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
                    all_images.append(
                        Image.open(f"nd2totiff/image_{i}_channel_{j}.tif")
                    )
            else:
                all_images.append(Image.open(f"nd2totiff/image_{i}.tif"))

        all_images[0].save(
            f"{file_name.split('/')[-1].split('.')[0]}.tif",
            save_all=True,
            append_images=all_images[1:],
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


def extract_tiff(
    tiff_file, fluo_dual_layer: bool = False, singe_layer_mode: bool = True
) -> int:
    folders = [
        folder
        for folder in os.listdir("TempData")
        if os.path.isdir(os.path.join(".", folder))
    ]

    if fluo_dual_layer:
        for i in [i for i in ["Fluo1", "Fluo2", "PH"] if i not in folders]:
            try:
                os.mkdir(f"TempData/{i}")
            except:
                continue
    elif singe_layer_mode:
        for i in [i for i in ["PH"] if i not in folders]:
            try:
                os.mkdir(f"TempData/{i}")
            except:
                continue
    else:
        for i in [i for i in ["Fluo1", "PH"] if i not in folders]:
            try:
                os.mkdir(f"TempData/{i}")
            except:
                continue

    with Image.open(tiff_file) as tiff:
        num_pages = tiff.n_frames
        print("###############################################")
        print(num_pages)
        print("###############################################")
        img_num = 0
        if fluo_dual_layer:
            for i in range(num_pages):
                tiff.seek(i)
                if (i + 2) % 3 == 0:
                    filename = f"TempData/Fluo1/{img_num}.tif"
                elif (i + 2) % 3 == 1:
                    filename = f"TempData/Fluo2/{img_num}.tif"
                    img_num += 1
                else:
                    filename = f"TempData/PH/{img_num}.tif"
                print(filename)
                tiff.save(filename, format="TIFF")
        elif singe_layer_mode:
            for i in range(num_pages):
                tiff.seek(i)
                filename = f"TempData/PH/{img_num}.tif"
                print(filename)
                tiff.save(filename, format="TIFF")
                img_num += 1
        else:
            for i in range(num_pages):
                tiff.seek(i)
                if (i + 1) % 2 == 0:
                    filename = f"TempData/Fluo1/{img_num}.tif"
                    img_num += 1
                else:
                    filename = f"TempData/PH/{img_num}.tif"
                print(filename)
                tiff.save(filename, format="TIFF")

    return num_pages


def get_contour_center(contour):
    # 輪郭のモーメントを計算して重心を求める
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def crop_contours(image, contours, output_size):
    cropped_images = []
    for contour in contours:
        # 各輪郭の中心座標を取得
        cx, cy = get_contour_center(contour)
        # 　中心座標が画像の中心から離れているものを除外
        if cx > 400 and cx < 2000 and cy > 400 and cy < 2000:
            # 切り抜く範囲を計算
            x1 = max(0, cx - output_size[0] // 2)
            y1 = max(0, cy - output_size[1] // 2)
            x2 = min(image.shape[1], cx + output_size[0] // 2)
            y2 = min(image.shape[0], cy + output_size[1] // 2)
            # 画像を切り抜く
            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)
    return cropped_images

import os
import cv2
import numpy as np
from typing import Literal


def init(
    input_filename: str,
    param1: int = 140,
    param2: int = 255,
    image_size: int = 100,
    mode: 
) -> int:

    if fluo_dual_layer_mode:
        set_num = 3
        init_folders = ["Fluo1", "Fluo2", "PH", "frames", "app_data"]
    elif single_layer_mode:
        set_num = 1
        init_folders = ["PH", "frames", "app_data"]
    else:
        set_num = 2
        init_folders = ["Fluo1", "PH", "frames", "app_data"]

    try:
        os.mkdir("TempData")
    except:
        pass

    init_folders = [f"TempData/{d}" for d in init_folders]
    folders = [
        folder
        for folder in os.listdir("TempData")
        if os.path.isdir(os.path.join(".", folder))
    ]
    for i in [i for i in init_folders if i not in folders]:
        try:
            os.mkdir(f"{i}")
        except:
            continue

    # 画像の枚数を取得
    num_tif = extract_tiff(
        input_filename,
        fluo_dual_layer=fluo_dual_layer_mode,
        singe_layer_mode=single_layer_mode,
    )
    # フォルダの作成
    for i in range(num_tif // set_num):
        directories = [
            f"TempData/frames/tiff_{i}",
            f"TempData/frames/tiff_{i}/Cells",
            f"TempData/frames/tiff_{i}/Cells/ph",
            f"TempData/frames/tiff_{i}/Cells/ph_raw",
            f"TempData/frames/tiff_{i}/Cells/fluo1",
            f"TempData/frames/tiff_{i}/Cells/fluo1_adjusted",
            f"TempData/frames/tiff_{i}/Cells/ph_contour",
            f"TempData/frames/tiff_{i}/Cells/fluo1_contour",
            f"TempData/frames/tiff_{i}/Cells/unified_images",
            "ph_contours",
        ]

        for directory in directories:
            try:
                os.mkdir(directory)
            except Exception as e:
                print(e)

        if fluo_dual_layer_mode:
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo2")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo2_adjusted")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo2_contour")
            except Exception as e:
                print(e)

    for k in tqdm(range(num_tif // set_num)):
        print(f"TempData/PH/{k}.tif")
        image_ph = cv2.imread(f"TempData/PH/{k}.tif")
        image_fluo_1 = cv2.imread(f"TempData/Fluo1/{k}.tif")
        if fluo_dual_layer_mode:
            image_fluo_2 = cv2.imread(f"TempData/Fluo2/{k}.tif")
        img_gray = cv2.cvtColor(image_ph, cv2.COLOR_BGR2GRAY)

        # ２値化を行う
        ret, thresh = cv2.threshold(img_gray, param1, param2, cv2.THRESH_BINARY)
        img_canny = cv2.Canny(thresh, 0, 130)

        contours, hierarchy = cv2.findContours(
            img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 細胞の面積で絞り込み
        contours = list(filter(lambda x: cv2.contourArea(x) >= 300, contours))
        # 中心座標が画像の中心から離れているものを除外
        contours = list(
            filter(
                lambda x: cv2.moments(x)["m10"] / cv2.moments(x)["m00"] > 400
                and cv2.moments(x)["m10"] / cv2.moments(x)["m00"] < 1700,
                contours,
            )
        )
        # do the same for y
        contours = list(
            filter(
                lambda x: cv2.moments(x)["m01"] / cv2.moments(x)["m00"] > 400
                and cv2.moments(x)["m01"] / cv2.moments(x)["m00"] < 1700,
                contours,
            )
        )

        output_size = (image_size, image_size)

        if not single_layer_mode:
            cropped_images_fluo_1 = crop_contours(image_fluo_1, contours, output_size)
        if fluo_dual_layer_mode:
            cropped_images_fluo_2 = crop_contours(image_fluo_2, contours, output_size)
        cropped_images_ph = crop_contours(image_ph, contours, output_size)

        image_ph_copy = image_ph.copy()
        cv2.drawContours(image_ph_copy, contours, -1, (0, 255, 0), 3)
        cv2.imwrite(f"ph_contours/{k}.png", image_ph_copy)
        n = 0
        if fluo_dual_layer_mode:
            for j, ph, fluo1, fluo2 in zip(
                [i for i in range(len(cropped_images_ph))],
                cropped_images_ph,
                cropped_images_fluo_1,
                cropped_images_fluo_2,
            ):
                if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                    cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                    cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/fluo1/{n}.png", fluo1)
                    cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/fluo2/{n}.png", fluo2)
                    brightness_factor_fluo1 = 255 / np.max(fluo1)
                    image_fluo1_brightened = cv2.convertScaleAbs(
                        fluo1, alpha=brightness_factor_fluo1, beta=0
                    )
                    cv2.imwrite(
                        f"TempData/frames/tiff_{k}/Cells/fluo_adjusted/{n}.png",
                        image_fluo1_brightened,
                    )
                    brightness_factor_fluo2 = 255 / np.max(fluo2)
                    image_fluo2_brightened = cv2.convertScaleAbs(
                        fluo2, alpha=brightness_factor_fluo2, beta=0
                    )
                    cv2.imwrite(
                        f"TempData/frames/tiff_{k}/Cells/fluo_adjusted/{n}.png",
                        image_fluo2_brightened,
                    )
                    n += 1
        elif single_layer_mode:
            for j, ph in zip(
                [i for i in range(len(cropped_images_ph))], cropped_images_ph
            ):
                if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                    cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                    n += 1
        else:
            for j, ph, fluo1 in zip(
                [i for i in range(len(cropped_images_ph))],
                cropped_images_ph,
                cropped_images_fluo_1,
            ):
                if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                    cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                    cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/fluo1/{n}.png", fluo1)
                    brightness_factor_fluo1 = 255 / np.max(fluo1)
                    image_fluo1_brightened = cv2.convertScaleAbs(
                        fluo1, alpha=brightness_factor_fluo1, beta=0
                    )
                    cv2.imwrite(
                        f"TempData/frames/tiff_{k}/Cells/fluo_adjusted/{n}.png",
                        image_fluo1_brightened,
                    )
                    n += 1

    return num_tif


# file = "backend/sk328gen120min.nd2"
# extract_nd2(file)
# extract_tiff("nd2totiff/image_0_channel_0.tif", fluo_dual_layer=True)
