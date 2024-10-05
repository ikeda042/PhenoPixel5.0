import numpy as np
import os
import nd2reader
from PIL import Image
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from fastapi.responses import JSONResponse
import io


class SyncChores:
    @staticmethod
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
    def extract_timelapse_nd2(file_name: str):
        """
        タイムラプスnd2ファイルをフレームごとにTIFF形式で保存する。(チャンネル0がFluo画像、チャンネル1がPH画像)
        """
        base_output_dir = "uploaded_files/"
        os.makedirs("TimelapseParserTemp", exist_ok=True)
        with nd2reader.ND2Reader(base_output_dir + file_name) as images:
            print(f"Available axes: {images.axes}")
            print(f"Sizes: {images.sizes}")

            images.bundle_axes = "cyx" if "c" in images.axes else "yx"
            images.iter_axes = "v"

            num_fields = images.sizes.get("v", 1)
            num_channels = images.sizes.get("c", 1)
            num_timepoints = images.sizes.get("t", 1)

            for field_idx in range(num_fields):
                field_folder = os.path.join(
                    "TimelapseParserTemp/", f"Field_{field_idx + 1}"
                )
                os.makedirs(field_folder, exist_ok=True)
                base_output_subdir_ph = field_folder + "/ph"
                base_output_subdir_fluo = field_folder + "/fluo"
                os.makedirs(base_output_subdir_ph, exist_ok=True)
                os.makedirs(base_output_subdir_fluo, exist_ok=True)

                reference_image_ph = None
                reference_image_fluo = None

                for channel_idx in range(num_channels):
                    for time_idx in range(num_timepoints):
                        images.default_coords.update(
                            {"v": field_idx, "c": channel_idx, "t": time_idx}
                        )
                        image_data = images[0]

                        if len(image_data.shape) == 3:
                            for i in range(image_data.shape[0]):
                                channel_image = SyncChores.process_image(image_data[i])

                                if i == 1:  # 位相差画像 (ph)
                                    if time_idx == 0:
                                        reference_image_ph = (
                                            channel_image  # 基準フレーム設定
                                        )
                                    if reference_image_ph is not None and time_idx > 0:
                                        channel_image = SyncChores.correct_drift(
                                            reference_image_ph, channel_image
                                        )  # ドリフト補正

                                    tiff_filename = os.path.join(
                                        base_output_subdir_ph,
                                        f"time_{time_idx + 1}_channel_{i}.tif",
                                    )

                                else:  # 蛍光画像 (fluo)
                                    if time_idx == 0:
                                        reference_image_fluo = (
                                            channel_image  # 基準フレーム設定
                                        )
                                    if (
                                        reference_image_fluo is not None
                                        and time_idx > 0
                                    ):
                                        channel_image = SyncChores.correct_drift(
                                            reference_image_fluo, channel_image
                                        )  # ドリフト補正

                                    tiff_filename = os.path.join(
                                        base_output_subdir_fluo,
                                        f"time_{time_idx + 1}_channel_{i}.tif",
                                    )

                                img = Image.fromarray(channel_image)
                                img.save(tiff_filename)
                                print(f"Saved: {tiff_filename}")
                        else:
                            image_data = SyncChores.process_image(image_data)

                            if time_idx == 0:
                                reference_image_ph = (
                                    image_data  # 最初のフレームを基準に設定
                                )

                            if reference_image_ph is not None and time_idx > 0:
                                image_data = SyncChores.correct_drift(
                                    reference_image_ph, image_data
                                )  # ドリフト補正

                            tiff_filename = os.path.join(
                                field_folder, f"time_{time_idx + 1}.tif"
                            )
                            img = Image.fromarray(image_data)
                            img.save(tiff_filename)
                            print(f"Saved: {tiff_filename}")

    @staticmethod
    def create_combined_gif(
        field_folder: str, resize_factor: float = 0.5
    ) -> io.BytesIO:
        """
        Field1のphとfluo画像を左右に並べて時系列順にGIFを作成し、バイトバッファとして返す。
        画像は指定されたリサイズ比率で縮小される。

        :param field_folder: 画像が含まれているフォルダのパス
        :param resize_factor: 画像のリサイズ比率（デフォルトは50%）
        :return: GIFのバイトバッファ
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

        # 画像をリサイズして左右に並べる
        combined_images = []
        print("####################GIF####################")
        for ph_img, fluo_img in zip(ph_images, fluo_images):
            # 画像をリサイズ
            ph_img_resized = ph_img.resize(
                (int(ph_img.width * resize_factor), int(ph_img.height * resize_factor))
            )
            fluo_img_resized = fluo_img.resize(
                (
                    int(fluo_img.width * resize_factor),
                    int(fluo_img.height * resize_factor),
                )
            )

            # リサイズ後の画像を結合
            combined_width = ph_img_resized.width + fluo_img_resized.width
            combined_height = max(ph_img_resized.height, fluo_img_resized.height)
            combined_img = Image.new("RGB", (combined_width, combined_height))
            combined_img.paste(ph_img_resized, (0, 0))
            combined_img.paste(fluo_img_resized, (ph_img_resized.width, 0))
            combined_images.append(combined_img)

        # バイトバッファにGIFを保存
        gif_buffer = io.BytesIO()
        combined_images[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=combined_images[1:],
            duration=100,  # 各フレームの表示時間（ミリ秒）
            loop=0,  # 無限ループ
        )

        # バッファの位置を先頭に戻す
        gif_buffer.seek(0)

        return gif_buffer


class AsyncChores:
    def __init__(self):
        self.executor = ThreadPoolExecutor()

    async def correct_drift(self, reference_image, target_image):
        """
        correct_driftを非同期で実行。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            partial(SyncChores.correct_drift, reference_image, target_image),
        )

    async def process_image(self, array):
        """
        process_imageを非同期で実行。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, partial(SyncChores.process_image, array)
        )

    async def extract_timelapse_nd2(self, file_name: str):
        """
        extract_timelapse_nd2を非同期で実行。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, partial(SyncChores.extract_timelapse_nd2, file_name)
        )

    async def shutdown(self):
        """
        スレッドプールのシャットダウンを非同期で行う。
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.executor.shutdown)

    async def create_combined_gif(self, field_folder: str) -> io.BytesIO:
        """
        create_combined_gifを非同期で実行。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, partial(SyncChores.create_combined_gif, field_folder)
        )


class TimelapseEngineCrudBase:
    def __init__(self, nd2_path: str):
        self.nd2_path = nd2_path

    async def get_nd2_filenames(self) -> list[str]:
        return [i for i in os.listdir("uploaded_files") if i.endswith("_timelapse.nd2")]

    async def delete_nd2_file(self, file_path: str):
        filename = filename.split("/")[-1]
        await asyncio.to_thread(os.remove, f"uploaded_files/{filename}")
        return True

    async def main(self):
        await AsyncChores().extract_timelapse_nd2(self.nd2_path)
        return JSONResponse(content={"message": "Timelapse extracted"})

    async def create_combined_gif(self, field: str):
        return await AsyncChores().create_combined_gif("TimelapseParserTemp/" + field)
