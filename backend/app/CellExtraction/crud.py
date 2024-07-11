import os
import numpy as np
from PIL import Image
from typing import Literal
import nd2reader
import os
import cv2
import numpy as np
import cv2
import numpy as np
import asyncio


class SyncChores:
    @staticmethod
    def process_image(array):
        """
        画像処理関数：正規化とスケーリングを行う。
        """
        array = array.astype(np.float32)  # Convert to float
        array -= array.min()  # Normalize to 0
        array /= array.max()  # Normalize to 1
        array *= 255  # Scale to 0-255
        return array.astype(np.uint8)

    @staticmethod
    def save_images(images, file_name, num_channels):
        """
        画像を保存し、MultipageTIFFとして出力する。
        """
        all_images = []
        for i, img in enumerate(images):
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

    @staticmethod
    def extract_nd2(file_name: str):
        """
        nd2ファイルをMultipageTIFFに変換する。
        """
        os.makedirs("nd2totiff", exist_ok=True)

        with nd2reader.ND2Reader(file_name) as images:
            print(f"Available axes: {images.axes}")
            print(f"Sizes: {images.sizes}")

            images.bundle_axes = "cyx" if "c" in images.axes else "yx"
            images.iter_axes = "v"

            num_channels = images.sizes.get("c", 1)
            print(f"Total images: {len(images)}")
            print(f"Channels: {num_channels}")
            print("##############################################")

            for n, img in enumerate(images):
                if num_channels > 1:
                    for channel in range(num_channels):
                        array = np.array(img[channel])
                        array = SyncChores.process_image(array)
                        image = Image.fromarray(array)
                        image.save(f"nd2totiff/image_{n}_channel_{channel}.tif")
                else:
                    array = np.array(img)
                    array = SyncChores.process_image(array)
                    image = Image.fromarray(array)
                    image.save(f"nd2totiff/image_{n}.tif")

            SyncChores.save_images(images, file_name, num_channels)
        SyncChores.cleanup("nd2totiff")
        num_tiff = SyncChores.extract_tiff(
            f"./{file_name.split('/')[-1].split('.')[0]}.tif"
        )
        os.remove(f"./{file_name.split('/')[-1].split('.')[0]}.tif")
        return num_tiff

    @staticmethod
    def extract_tiff(
        tiff_path: str,
        mode: Literal["single_layer", "dual_layer", "triple_layer"] = "dual_layer",
    ) -> int:
        os.makedirs("TempData", exist_ok=True)
        folders = [
            folder
            for folder in os.listdir("TempData")
            if os.path.isdir(os.path.join("TempData", folder))
        ]

        layers = {
            "triple_layer": ["Fluo1", "Fluo2", "PH"],
            "single_layer": ["PH"],
            "dual_layer": ["Fluo1", "PH"],
        }

        for layer in layers.get(mode, []):
            os.makedirs(f"TempData/{layer}", exist_ok=True)

        with Image.open(tiff_path) as tiff:
            num_pages = tiff.n_frames
            img_num = 0

            layer_map = {
                "triple_layer": [(0, "PH"), (1, "Fluo1"), (2, "Fluo2")],
                "single_layer": [(0, "PH")],
                "dual_layer": [(0, "PH"), (1, "Fluo1")],
            }

            for i in range(num_pages):
                tiff.seek(i)
                layer_idx = i % len(layer_map[mode])
                layer = layer_map[mode][layer_idx][1]
                filename = f"TempData/{layer}/{img_num}.tif"
                print(filename)
                tiff.save(filename, format="TIFF")
                if layer_idx == len(layer_map[mode]) - 1:
                    img_num += 1

        return num_pages

    @staticmethod
    def cleanup(directory: str):
        """
        指定されたディレクトリを削除する。
        """
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(directory)

    @staticmethod
    def get_contour_center(contour):
        # 輪郭のモーメントを計算して重心を求める
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    @staticmethod
    def crop_contours(image, contours, output_size):
        cropped_images = []
        for contour in contours:
            # 各輪郭の中心座標を取得
            cx, cy = SyncChores.get_contour_center(contour)
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

    @staticmethod
    def init(
        input_filename: str,
        num_tiff: int,
        param1: int = 85,
        image_size: int = 200,
        mode: Literal["single_layer", "dual_layer", "triple_layer"] = "dual_layer",
    ) -> int:

        if mode == "triple_layer":
            set_num = 3
            init_folders = ["Fluo1", "Fluo2", "PH", "frames", "app_data"]
        elif mode == "single_layer":
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
        # フォルダの作成
        for i in range(num_tiff // set_num):
            try:
                os.mkdir(f"TempData/frames/tiff_{i}")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/ph")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/ph_raw")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo1")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo1_adjusted")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/ph_contour")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/fluo1_contour")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"TempData/frames/tiff_{i}/Cells/unified_images")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"ph_contours")
            except Exception as e:
                print(e)

            if mode == "triple_layer":
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

        for k in range(num_tiff // set_num):
            image_ph = cv2.imread(f"TempData/PH/{k}.tif")
            image_fluo_1 = cv2.imread(f"TempData/Fluo1/{k}.tif")
            if mode == "triple_layer":
                image_fluo_2 = cv2.imread(f"TempData/Fluo2/{k}.tif")
            img_gray = cv2.cvtColor(image_ph, cv2.COLOR_BGR2GRAY)

            # ２値化を行う
            ret, thresh = cv2.threshold(img_gray, param1, 255, cv2.THRESH_BINARY)
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

            cropped_images_ph = SyncChores.crop_contours(
                image_ph, contours, output_size
            )
            if mode == "triple_layer" or mode == "dual_layer":
                cropped_images_fluo_1 = SyncChores.crop_contours(
                    image_fluo_1, contours, output_size
                )
            if mode == "triple_layer":
                cropped_images_fluo_2 = SyncChores.crop_contours(
                    image_fluo_2, contours, output_size
                )

            image_ph_copy = image_ph.copy()
            cv2.drawContours(image_ph_copy, contours, -1, (0, 255, 0), 3)
            cv2.imwrite(f"ph_contours/{k}.png", image_ph_copy)
            n = 0
            if mode == "triple_layer":
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
            elif mode == "single_layer":
                for j, ph in zip(
                    [i for i in range(len(cropped_images_ph))], cropped_images_ph
                ):
                    if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                        cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                        n += 1
            elif mode == "dual_layer":
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
        return num_tiff


class CellExtraction:
    def __init__(
        self,
        nd2_path: str = "/Users/leeyunosuke/Documents/PhenoPixel5.0/sk326tri30min.nd2",
        mode: str = "dual_layer",
    ):
        self.nd2_path = nd2_path
        self.file_prefix = self.nd2_path.split("/")[-1].split(".")[0]
        self.mode = mode

    async def run_in_thread(self, func, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

    async def main(self):
        chores = SyncChores()
        num_tiff = await self.run_in_thread(chores.extract_nd2, self.nd2_path)
        await self.run_in_thread(
            chores.init, f"{self.file_prefix}.nd2", 24, 85, 200, self.mode
        )


if __name__ == "__main__":
    asyncio.run(CellExtraction().main())


# from .initialize import init
# from .unify_images import unify_images_ndarray2, unify_images_ndarray
# from .database import Cell, Base
# from .calc_center import get_contour_center
# import os
# import cv2
# from tqdm import tqdm
# import pickle
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from typing import Literal
# from database import get_session, Cell


# def image_process(
#     input_filename: str,
#     param1: int = 80,
#     image_size: int = 100,
#     mode: Literal["single_layer", "dual_layer", "triple_layer"] = "dual_layer",
# ) -> None:
#     engine = create_engine(f'sqlite:///{input_filename.split(".")[0]}.db', echo=False)
#     Base.metadata.create_all(engine)
#     Session = sessionmaker(bind=engine)
#     num_tif = init(
#         input_filename=input_filename,
#         param1=param1,
#         param2=param2,
#         image_size=image_size,
#         fluo_dual_layer_mode=fluo_dual_layer_mode,
#         single_layer_mode=single_layer_mode,
#     )
#     print(num_tif)
#     print(
#         "Processing images...\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n+\n"
#     )
#     iter_n = num_tif // 3 if not single_layer_mode else num_tif
#     for k in tqdm(range(0, iter_n)):
#         for j in range(len(os.listdir(f"TempData/frames/tiff_{k}/Cells/ph/"))):
#             cell_id: str = f"F{k}C{j}"
#             img_ph = cv2.imread(f"TempData/frames/tiff_{k}/Cells/ph/{j}.png")
#             if not single_layer_mode:
#                 img_fluo1 = cv2.imread(f"TempData/frames/tiff_{k}/Cells/fluo1/{j}.png")

#             img_ph_gray = cv2.cvtColor(img_ph, cv2.COLOR_BGR2GRAY)
#             if not single_layer_mode:
#                 img_fluo1_gray = cv2.cvtColor(img_fluo1, cv2.COLOR_BGR2GRAY)

#             ret, thresh = cv2.threshold(img_ph_gray, param1, param2, cv2.THRESH_BINARY)
#             img_canny = cv2.Canny(thresh, 0, 150)
#             contours_raw, hierarchy = cv2.findContours(
#                 img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
#             )
#             # Filter out contours with small area
#             contours = list(filter(lambda x: cv2.contourArea(x) >= 300, contours_raw))
#             # Check if the center of the contour is not too far from the center of the image
#             contours = list(
#                 filter(
#                     lambda x: abs(
#                         cv2.moments(x)["m10"] / cv2.moments(x)["m00"] - image_size / 2
#                     )
#                     < 10,
#                     contours,
#                 )
#             )
#             # do the same for y
#             contours = list(
#                 filter(
#                     lambda x: abs(
#                         cv2.moments(x)["m01"] / cv2.moments(x)["m00"] - image_size / 2
#                     )
#                     < 10,
#                     contours,
#                 )
#             )

#             if not single_layer_mode:
#                 cv2.drawContours(img_fluo1, contours, -1, (0, 255, 0), 1)
#                 cv2.imwrite(
#                     f"TempData/frames/tiff_{k}/Cells/fluo1_contour/{j}.png", img_fluo1
#                 )
#             cv2.drawContours(img_ph, contours, -1, (0, 255, 0), 1)
#             cv2.imwrite(f"TempData/frames/tiff_{k}/Cells/ph_contour/{j}.png", img_ph)

#             if fluo_dual_layer_mode:
#                 img_fluo2 = cv2.imread(f"TempData/frames/tiff_{k}/Cells/fluo2/{j}.png")
#                 print(f"empData/frames/tiff_{k}/Cells/fluo2/{j}.png")
#                 img_fluo2_gray = cv2.cvtColor(img_fluo2, cv2.COLOR_BGR2GRAY)
#                 cv2.drawContours(img_fluo2, contours, -1, (0, 255, 0), 1)
#                 cv2.imwrite(
#                     f"TempData/frames/tiff_{k}/Cells/fluo2_contour/{j}.png", img_fluo2
#                 )

#             if contours != []:
#                 if draw_scale_bar:
#                     image_ph_copy = img_ph.copy()
#                     if not single_layer_mode:
#                         image_fluo1_copy = img_fluo1.copy()

#                     pixel_per_micro_meter = 0.0625
#                     # want to draw a scale bar of 20% of the image width at the bottom right corner. (put some mergins so that the scale bar is not too close to the edge)
#                     # scale bar length in pixels
#                     scale_bar_length = int(image_size * 0.2)
#                     scale_bar_size = scale_bar_length * pixel_per_micro_meter
#                     # scale bar thickness in pixels
#                     scale_bar_thickness = int(2 * (image_size / 100))
#                     # scale bar mergins from the edge of the image
#                     scale_bar_mergins = int(10 * (image_size / 100))
#                     # scale bar color
#                     scale_bar_color = (255, 255, 255)
#                     # scale bar text color
#                     scale_bar_text_color = (255, 255, 255)
#                     # draw scale bar for the both image_ph and image_fluo and the scale bar should be Rectangle
#                     # scale bar for image_ph
#                     cv2.rectangle(
#                         image_ph_copy,
#                         (
#                             image_size - scale_bar_mergins - scale_bar_length,
#                             image_size - scale_bar_mergins,
#                         ),
#                         (
#                             image_size - scale_bar_mergins,
#                             image_size - scale_bar_mergins - scale_bar_thickness,
#                         ),
#                         scale_bar_color,
#                         -1,
#                     )
#                     # cv2.putText(image_ph_copy,f"{round(scale_bar_size,2)} µm",(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins-2*scale_bar_thickness),cv2.FONT_HERSHEY_SIMPLEX,0.2,scale_bar_text_color,1,cv2.LINE_AA)
#                     # scale bar for image_fluo
#                     if not single_layer_mode:
#                         cv2.rectangle(
#                             image_fluo1_copy,
#                             (
#                                 image_size - scale_bar_mergins - scale_bar_length,
#                                 image_size - scale_bar_mergins,
#                             ),
#                             (
#                                 image_size - scale_bar_mergins,
#                                 image_size - scale_bar_mergins - scale_bar_thickness,
#                             ),
#                             scale_bar_color,
#                             -1,
#                         )
#                     # cv2.putText(image_fluo_copy,f"{round(scale_bar_size,2)} µm",(image_size-scale_bar_mergins-scale_bar_length,image_size-scale_bar_mergins-2*scale_bar_thickness),cv2.FONT_HERSHEY_SIMPLEX,0.2,scale_bar_text_color,1,cv2.LINE_AA)
#                     if fluo_dual_layer_mode:
#                         image_fluo2_copy = img_fluo2.copy()
#                         cv2.rectangle(
#                             image_fluo2_copy,
#                             (
#                                 image_size - scale_bar_mergins - scale_bar_length,
#                                 image_size - scale_bar_mergins,
#                             ),
#                             (
#                                 image_size - scale_bar_mergins,
#                                 image_size - scale_bar_mergins - scale_bar_thickness,
#                             ),
#                             scale_bar_color,
#                             -1,
#                         )
#                         unify_images_ndarray2(
#                             image1=image_ph_copy,
#                             image2=image_fluo1_copy,
#                             image3=image_fluo2_copy,
#                             output_name=f"TempData/frames/tiff_{k}/Cells/unified_images/{j}",
#                         )
#                         unify_images_ndarray2(
#                             image1=image_ph_copy,
#                             image2=image_fluo1_copy,
#                             image3=image_fluo2_copy,
#                             output_name=f"TempData/app_data/{cell_id}",
#                         )
#                     elif single_layer_mode:
#                         cv2.imwrite(
#                             f"TempData/frames/tiff_{k}/Cells/unified_images/{j}.png",
#                             image_ph_copy,
#                         )
#                         cv2.imwrite(f"TempData/app_data/{cell_id}.png", image_ph_copy)
#                     else:
#                         unify_images_ndarray(
#                             image1=image_ph_copy,
#                             image2=image_fluo1_copy,
#                             output_name=f"TempData/frames/tiff_{k}/Cells/unified_images/{j}",
#                         )
#                         unify_images_ndarray(
#                             image1=image_ph_copy,
#                             image2=image_fluo1_copy,
#                             output_name=f"TempData/app_data/{cell_id}",
#                         )

#                 with Session() as session:
#                     perimeter = cv2.arcLength(contours[0], closed=True)
#                     area = cv2.contourArea(contour=contours[0])
#                     image_ph_data = cv2.imencode(".png", img_ph_gray)[1].tobytes()
#                     if not single_layer_mode:
#                         image_fluo1_data = cv2.imencode(".png", img_fluo1_gray)[
#                             1
#                         ].tobytes()
#                     if fluo_dual_layer_mode:
#                         image_fluo2_data = cv2.imencode(".png", img_fluo2_gray)[
#                             1
#                         ].tobytes()
#                     contour = pickle.dumps(contours[0])
#                     center_x, center_y = get_contour_center(contours[0])
#                     print(center_x, center_y)
#                     if fluo_dual_layer_mode:
#                         cell = Cell(
#                             cell_id=cell_id,
#                             label_experiment="",
#                             perimeter=perimeter,
#                             area=area,
#                             img_ph=image_ph_data,
#                             img_fluo1=image_fluo1_data,
#                             img_fluo2=image_fluo2_data,
#                             contour=contour,
#                             center_x=center_x,
#                             center_y=center_y,
#                         )
#                     elif single_layer_mode:
#                         cell = Cell(
#                             cell_id=cell_id,
#                             label_experiment="",
#                             perimeter=perimeter,
#                             area=area,
#                             img_ph=image_ph_data,
#                             contour=contour,
#                             center_x=center_x,
#                             center_y=center_y,
#                         )
#                     else:
#                         cell = Cell(
#                             cell_id=cell_id,
#                             label_experiment="",
#                             perimeter=perimeter,
#                             area=area,
#                             img_ph=image_ph_data,
#                             img_fluo1=image_fluo1_data,
#                             contour=contour,
#                             center_x=center_x,
#                             center_y=center_y,
#                         )
#                     if session.query(Cell).filter_by(cell_id=cell_id).first() is None:
#                         session.add(cell)
#                         session.commit()
