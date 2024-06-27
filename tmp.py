import os
from PIL import Image
import nd2reader
import numpy as np
import cv2
from typing import Literal
from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
import cv2
from tqdm import tqdm
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, BLOB, select, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update


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
        print(
            "########################################################################################################################################################################################"
        )

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


def streamlined_save_tiff(tiff, mode, num_pages):
    img_num = 0
    folder_map = {
        "dual": lambda i: (
            "Fluo1" if (i + 2) % 3 == 0 else "Fluo2" if (i + 2) % 3 == 1 else "PH",
            1 if (i + 2) % 3 == 1 else 0,
        ),
        "single": lambda i: ("PH", 1),
        "normal": lambda i: (
            "Fluo1" if (i + 1) % 2 == 0 else "PH",
            1 if (i + 1) % 2 == 0 else 0,
        ),
    }
    if mode not in folder_map:
        raise ValueError("Invalid mode")

    for i in range(num_pages):
        tiff.seek(i)
        folder, increment = folder_map[mode](i)
        filename = f"TempData/{folder}/{img_num}.tif"
        tiff.save(filename, format="TIFF")
        img_num += increment


def extract_tiff(
    tiff_file_path: str, mode: Literal["normal", "single", "dual"] = "normal"
) -> int:
    if not os.path.exists("TempData"):
        os.mkdir("TempData")

    folders = [
        folder
        for folder in os.listdir("TempData")
        if os.path.isdir(os.path.join(".", folder))
    ]

    mode_to_folders = {
        "dual": ["Fluo1", "Fluo2", "PH"],
        "single": ["PH"],
        "normal": ["Fluo1", "PH"],
    }

    folders_to_create = mode_to_folders.get(mode, [])

    for folder in folders_to_create:
        if folder not in folders:
            try:
                os.mkdir(f"TempData/{folder}")
            except:
                continue

    if mode not in mode_to_folders:
        raise ValueError("Invalid mode")

    with Image.open(tiff_file_path) as tiff:
        num_pages = tiff.n_frames
        streamlined_save_tiff(tiff, mode, num_pages)
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


def init(
    input_filename: str,
    image_size: int = 100,
    mode: Literal["normal", "single", "dual"] = "normal",
    param1: int = 85,
    param2: int = 255,
) -> int:
    if mode == "dual":
        set_num = 3
        init_folders = ["Fluo1", "Fluo2", "PH", "frames", "app_data"]
    elif mode == "single":
        set_num = 1
        init_folders = ["PH", "frames", "app_data"]
    elif mode == "normal":
        set_num = 2
        init_folders = ["Fluo1", "PH", "frames", "app_data"]
    else:
        raise ValueError("Invalid mode")

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
        mode=mode,
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

        if mode == "dual":
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

    for k in range(num_tif // set_num):
        print(f"TempData/PH/{k}.tif")
        image_ph = cv2.imread(f"TempData/PH/{k}.tif")
        image_fluo_1 = cv2.imread(f"TempData/Fluo1/{k}.tif")
        if mode == "dual":
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

        if not mode == "single":
            cropped_images_fluo_1 = crop_contours(image_fluo_1, contours, output_size)
        if mode == "dual":
            cropped_images_fluo_2 = crop_contours(image_fluo_2, contours, output_size)
        cropped_images_ph = crop_contours(image_ph, contours, output_size)

        image_ph_copy = image_ph.copy()
        cv2.drawContours(image_ph_copy, contours, -1, (0, 255, 0), 3)
        cv2.imwrite(f"ph_contours/{k}.png", image_ph_copy)
        n = 0
        if mode == "dual":
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
        elif mode == "single":
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


def unify_images_ndarray2(image1, image2, image3, output_name):
    combined_width = image1.shape[1] + image2.shape[1] + image3.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0], image3.shape[0])

    canvas = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Image 1
    canvas[: image1.shape[0], : image1.shape[1]] = image1

    # Image 2
    offset_x_image2 = image1.shape[1]
    canvas[: image2.shape[0], offset_x_image2 : offset_x_image2 + image2.shape[1]] = (
        image2
    )

    # Image 3
    offset_x_image3 = offset_x_image2 + image2.shape[1]
    canvas[: image3.shape[0], offset_x_image3 : offset_x_image3 + image3.shape[1]] = (
        image3
    )

    cv2.imwrite(f"{output_name}.png", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def unify_images_ndarray(image1, image2, output_name):
    combined_width = image1.shape[1] + image2.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0])

    canvas = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    canvas[: image1.shape[0], : image1.shape[1], :] = image1
    canvas[: image2.shape[0], image1.shape[1] :, :] = image2
    cv2.imwrite(f"{output_name}.png", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def unify_images_ndarray6(image1, image2, image3, image4, image5, image6, output_name):
    # すべての画像の幅と高さを取得
    widths = [
        image1.shape[1],
        image2.shape[1],
        image3.shape[1],
        image4.shape[1],
        image5.shape[1],
        image6.shape[1],
    ]
    heights = [
        image1.shape[0],
        image2.shape[0],
        image3.shape[0],
        image4.shape[0],
        image5.shape[0],
        image6.shape[0],
    ]

    # 最大の幅と高さを決定
    max_width = max(widths)
    max_height = max(heights)

    # キャンバスのサイズを決定（3行2列）
    canvas_width = max_width * 2
    canvas_height = max_height * 3

    # キャンバスを作成
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 各画像をキャンバスに配置
    images = [image1, image2, image3, image4, image5, image6]
    for i, img in enumerate(images):
        row = i // 2
        col = i % 2
        canvas[
            row * max_height : (row + 1) * max_height,
            col * max_width : (col + 1) * max_width,
            :,
        ] = cv2.resize(img, (max_width, max_height))

    # 画像を保存
    cv2.imwrite(f"{output_name}.png", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


Base = declarative_base()


class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB)
    img_fluo2 = Column(BLOB)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)


def image_process(
    input_filename: str = "data.tif",
    param1: int = 80,
    param2: int = 255,
    image_size: int = 100,
    draw_scale_bar: bool = True,
    fluo_dual_layer_mode: bool = True,
    single_layer_mode: bool = False,
) -> None:
    engine = create_engine(f'sqlite:///{input_filename.split(".")[0]}.db', echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    num_tif = init(
        input_filename=input_filename,
        param1=param1,
        param2=param2,
        image_size=image_size,
        mode=(
            "dual"
            if fluo_dual_layer_mode
            else "single" if single_layer_mode else "normal"
        ),
    )
    print(num_tif)
    print(
        "Processing images...\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n"
    )
    iter_n = num_tif // 3 if not single_layer_mode else num_tif
    for k in tqdm(range(0, iter_n)):
        for j in range(len(os.listdir(f"TempData/frames/tiff_{k}/Cells/ph/"))):
            cell_id: str = f"F{k}C{j}"
            img_ph = cv2.imread(f"TempData/frames/tiff_{k}/Cells/ph/{j}.png")
            if not single_layer_mode:
                img_fluo1 = cv2.imread(f"TempData/frames/tiff_{k}/Cells/fluo1/{j}.png")

            img_ph_gray = cv2.cvtColor(img_ph, cv2.COLOR_BGR2GRAY)
            if not single_layer_mode:
                img_fluo1_gray = cv2.cvtColor(img_fluo1, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(img_ph_gray, param1, param2, cv2.THRESH_BINARY)
            img_canny = cv2.Canny(thresh, 0, 150)
            contours_raw, hierarchy = cv2.findContours(
                img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Filter out contours with small area
            contours = list(filter(lambda x: cv2.contourArea(x) >= 300, contours_raw))
            # Check if the center of the contour is not too far from the center of the image
            contours = list(
                filter(
                    lambda x: abs(
                        cv2.moments(x)["m10"] / cv2.moments(x)["m00"] - image_size / 2
                    )
                    < 10,
                    contours,
                )
            )
            # do the same for y
            contours = list(
                filter(
                    lambda x: abs(
                        cv2.moments(x)["m01"] / cv2.moments(x)["m00"] - image_size / 2
                    )
                    < 10,
                    contours,
                )
            )

            if not single_layer_mode:
                cv2.drawContours(img_fluo1, contours, -1, (0, 255, 0), 1)
                cv2.imwrite(
                    f"TempData/frames/tiff_{k}/Cells/fluo1_contour/{j}.png", img_fluo1
                )
            cv2.drawContours(img_ph, contours, -1, (0, 255, 0), 1)
            cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/ph_contour/{j}.png", img_ph)

            if fluo_dual_layer_mode:
                img_fluo2 = cv2.imread(f"TempData/frames/tiff_{k}/Cells/fluo2/{j}.png")
                print(f"empData/frames/tiff_{k}/Cells/fluo2/{j}.png")
                img_fluo2_gray = cv2.cvtColor(img_fluo2, cv2.COLOR_BGR2GRAY)
                cv2.drawContours(img_fluo2, contours, -1, (0, 255, 0), 1)
                cv2.imwrite(
                    f"TempData/frames/tiff_{k}/Cells/fluo2_contour/{j}.png", img_fluo2
                )

            if contours != []:
                if draw_scale_bar:
                    image_ph_copy = img_ph.copy()
                    if not single_layer_mode:
                        image_fluo1_copy = img_fluo1.copy()

                    pixel_per_micro_meter = 0.0625
                    # want to draw a scale bar of 20% of the image width at the bottom right corner. (put some mergins so that the scale bar is not too close to the edge)
                    # scale bar length in pixels
                    scale_bar_length = int(image_size * 0.2)
                    scale_bar_size = scale_bar_length * pixel_per_micro_meter
                    # scale bar thickness in pixels
                    scale_bar_thickness = int(2 * (image_size / 100))
                    # scale bar mergins from the edge of the image
                    scale_bar_mergins = int(10 * (image_size / 100))
                    # scale bar color
                    scale_bar_color = (255, 255, 255)
                    # scale bar text color
                    scale_bar_text_color = (255, 255, 255)
                    # draw scale bar for the both image_ph and image_fluo and the scale bar should be Rectangle
                    # scale bar for image_ph
                    cv2.rectangle(
                        image_ph_copy,
                        (
                            image_size - scale_bar_mergins - scale_bar_length,
                            image_size - scale_bar_mergins,
                        ),
                        (
                            image_size - scale_bar_mergins,
                            image_size - scale_bar_mergins - scale_bar_thickness,
                        ),
                        scale_bar_color,
                        -1,
                    )
                    # cv2.putText(image_ph_copy,f"{round(scale_bar_size,2)} µm",(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins-2*scale_bar_thickness),cv2.FONT_HERSHEY_SIMPLEX,0.2,scale_bar_text_color,1,cv2.LINE_AA)
                    # scale bar for image_fluo
                    if not single_layer_mode:
                        cv2.rectangle(
                            image_fluo1_copy,
                            (
                                image_size - scale_bar_mergins - scale_bar_length,
                                image_size - scale_bar_mergins,
                            ),
                            (
                                image_size - scale_bar_mergins,
                                image_size - scale_bar_mergins - scale_bar_thickness,
                            ),
                            scale_bar_color,
                            -1,
                        )
                    # cv2.putText(image_fluo_copy,f"{round(scale_bar_size,2)} µm",(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins-2*scale_bar_thickness),cv2.FONT_HERSHEY_SIMPLEX,0.2,scale_bar_text_color,1,cv2.LINE_AA)
                    if fluo_dual_layer_mode:
                        image_fluo2_copy = img_fluo2.copy()
                        cv2.rectangle(
                            image_fluo2_copy,
                            (
                                image_size - scale_bar_mergins - scale_bar_length,
                                image_size - scale_bar_mergins,
                            ),
                            (
                                image_size - scale_bar_mergins,
                                image_size - scale_bar_mergins - scale_bar_thickness,
                            ),
                            scale_bar_color,
                            -1,
                        )
                        unify_images_ndarray2(
                            image1=image_ph_copy,
                            image2=image_fluo1_copy,
                            image3=image_fluo2_copy,
                            output_name=f"TempData/frames/tiff_{k}/Cells/unified_images/{j}",
                        )
                        unify_images_ndarray2(
                            image1=image_ph_copy,
                            image2=image_fluo1_copy,
                            image3=image_fluo2_copy,
                            output_name=f"TempData/app_data/{cell_id}",
                        )
                    elif single_layer_mode:
                        cv2.imwrite(
                            f"TempData/frames/tiff_{k}/Cells/unified_images/{j}.png",
                            image_ph_copy,
                        )
                        cv2.imwrite(f"TempData/app_data/{cell_id}.png", image_ph_copy)
                    else:
                        unify_images_ndarray(
                            image1=image_ph_copy,
                            image2=image_fluo1_copy,
                            output_name=f"TempData/frames/tiff_{k}/Cells/unified_images/{j}",
                        )
                        unify_images_ndarray(
                            image1=image_ph_copy,
                            image2=image_fluo1_copy,
                            output_name=f"TempData/app_data/{cell_id}",
                        )

                with Session() as session:
                    perimeter = cv2.arcLength(contours[0], closed=True)
                    area = cv2.contourArea(contour=contours[0])
                    image_ph_data = cv2.imencode(".png", img_ph_gray)[1].tobytes()
                    if not single_layer_mode:
                        image_fluo1_data = cv2.imencode(".png", img_fluo1_gray)[
                            1
                        ].tobytes()
                    if fluo_dual_layer_mode:
                        image_fluo2_data = cv2.imencode(".png", img_fluo2_gray)[
                            1
                        ].tobytes()
                    contour = pickle.dumps(contours[0])
                    center_x, center_y = get_contour_center(contours[0])
                    print(center_x, center_y)
                    if fluo_dual_layer_mode:
                        cell = Cell(
                            cell_id=cell_id,
                            label_experiment="",
                            perimeter=perimeter,
                            area=area,
                            img_ph=image_ph_data,
                            img_fluo1=image_fluo1_data,
                            img_fluo2=image_fluo2_data,
                            contour=contour,
                            center_x=center_x,
                            center_y=center_y,
                        )
                    elif single_layer_mode:
                        cell = Cell(
                            cell_id=cell_id,
                            label_experiment="",
                            perimeter=perimeter,
                            area=area,
                            img_ph=image_ph_data,
                            contour=contour,
                            center_x=center_x,
                            center_y=center_y,
                        )
                    else:
                        cell = Cell(
                            cell_id=cell_id,
                            label_experiment="",
                            perimeter=perimeter,
                            area=area,
                            img_ph=image_ph_data,
                            img_fluo1=image_fluo1_data,
                            contour=contour,
                            center_x=center_x,
                            center_y=center_y,
                        )
                    if session.query(Cell).filter_by(cell_id=cell_id).first() is None:
                        session.add(cell)
                        session.commit()


class AsyncCellCRUD:
    def __init__(self, db_name: str):
        self.DATABASE_URL = f"sqlite+aiosqlite://./{db_name}.db"
        self.engine = create_async_engine(self.DATABASE_URL, echo=True)
        self.AsyncSessionLocal = sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self, Base):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def read_all_cell_ids(self, Cell):
        async with self.AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(select(Cell.cell_id))
                cell_ids = result.scalars().all()
                return cell_ids

    async def read_cell(self, Cell, cell_id: str):
        async with self.AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(select(Cell).filter_by(cell_id=cell_id))
                cell = result.scalars().first()
                return cell

    async def update_cell(self, Cell, cell_id: str, **kwargs):
        async with self.AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    update(Cell).where(Cell.cell_id == cell_id).values(**kwargs)
                )
                await session.commit()


# file = "sk328gen120min.nd2"
# extract_nd2(file)
# init("sk328gen120min.tif", mode="dual")
# image_process(
#     "sk328gen120min.tif",
#     param1=85,
#     param2=255,
#     image_size=100,
#     fluo_dual_layer_mode=True,
#     single_layer_mode=False,
# )
