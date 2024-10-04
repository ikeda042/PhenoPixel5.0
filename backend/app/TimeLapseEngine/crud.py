import numpy as np
import os
import nd2reader
from PIL import Image
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


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
                                channel_image = SyncChores.process_image(image_data[i])
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
                            image_data = SyncChores.process_image(image_data)
                            tiff_filename = os.path.join(
                                field_folder, f"time_{time_idx + 1}.tif"
                            )
                            img = Image.fromarray(image_data)
                            img.save(tiff_filename)
                            print(f"Saved: {tiff_filename}")


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


class TimelapseEngineCrudBase:
    def __init__(self, nd2_path: str):
        self.nd2_path = nd2_path

    async def get_nd2_filenames(self) -> list[str]:
        return [i for i in os.listdir("uploaded_files") if i.endswith("_timelapse.nd2")]
