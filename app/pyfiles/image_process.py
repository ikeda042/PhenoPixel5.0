from .database import Cell, Base
from tqdm import tqdm
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import cv2
import numpy as np
import os
from PIL import Image


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


def init(
    input_filename: str,
    param1: int = 140,
    param2: int = 255,
    image_size: int = 100,
    fluo_dual_layer_mode: bool = True,
    single_layer_mode: bool = False,
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
    folders = [
        "Cells",
        "Cells/ph",
        "Cells/ph_raw",
        "Cells/fluo1",
        "Cells/fluo1_adjusted",
        "Cells/ph_contour",
        "Cells/fluo1_contour",
        "Cells/unified_images",
        "ph_contours",
    ]

    if fluo_dual_layer_mode:
        folders.extend(["Cells/fluo2", "Cells/fluo2_adjusted", "Cells/fluo2_contour"])

    for i in tqdm(range(num_tif // set_num)):
        for folder in folders:
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/{folder}")
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
                cropped_images_fluo_1 = crop_contours(
                    image_fluo_1, contours, output_size
                )
            if fluo_dual_layer_mode:
                cropped_images_fluo_2 = crop_contours(
                    image_fluo_2, contours, output_size
                )
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
                        cv2.imwrite(
                            f"TempData/frames/tiff_{k}/Cells/fluo1/{n}.png", fluo1
                        )
                        cv2.imwrite(
                            f"TempData/frames/tiff_{k}/Cells/fluo2/{n}.png", fluo2
                        )
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
                        cv2.imwrite(
                            f"TempData/frames/tiff_{k}/Cells/fluo1/{n}.png", fluo1
                        )
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
    """
    Combine 3 images into a single image.( for trile channel mode)
    """
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
    """
    Combine 2 images into a single image.( for dual channel mode)
    """
    combined_width = image1.shape[1] + image2.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0])

    canvas = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    canvas[: image1.shape[0], : image1.shape[1], :] = image1
    canvas[: image2.shape[0], image1.shape[1] :, :] = image2
    cv2.imwrite(f"{output_name}.png", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


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
        fluo_dual_layer_mode=fluo_dual_layer_mode,
        single_layer_mode=single_layer_mode,
    )
    for k in tqdm(range(0, num_tif // 3)):
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
