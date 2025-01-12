import asyncio
import os
import io
import nd2reader
from PIL import Image
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from fastapi.responses import JSONResponse
import shutil


class SyncChores:
    @staticmethod
    def correct_drift(reference_image, target_image):
        """
        ORB + BFMatcher + 座標反転してからアフィン変換を推定。
        """
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(reference_image, None)
        kp2, des2 = orb.detectAndCompute(target_image, None)

        if des1 is None or des2 is None:
            print("Descriptor is None, skipping drift correction.")
            return target_image

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 10:  # マッチングが少なすぎる場合、補正を行わない
            print("Insufficient matches, skipping drift correction.")
            return target_image

        # 特徴点の座標を取得し、x軸とy軸を反転させる
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        height, width = reference_image.shape[:2]
        # x座標とy座標をそれぞれ反転
        src_pts[:, :, 0] = width - src_pts[:, :, 0]  # x座標を反転
        src_pts[:, :, 1] = height - src_pts[:, :, 1]  # y座標を反転
        dst_pts[:, :, 0] = width - dst_pts[:, :, 0]
        dst_pts[:, :, 1] = height - dst_pts[:, :, 1]

        # RANSACで部分アフィン行列を推定
        matrix, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if matrix is not None:
            aligned_image = cv2.warpAffine(
                target_image, matrix, (target_image.shape[1], target_image.shape[0])
            )
            return aligned_image
        else:
            print("Matrix estimation failed, skipping drift correction.")
            return target_image

    @staticmethod
    def process_image(array):
        """
        シンプルな正規化＆8bit変換
        """
        array = array.astype(np.float32)  # Convert to float
        array_min = array.min()
        array_max = array.max()
        array -= array_min
        diff = array_max - array_min
        if diff == 0:
            # 全てが同値の場合
            return (array * 0).astype(np.uint8)
        array /= diff
        array *= 255
        return array.astype(np.uint8)

    @classmethod
    def _drift_and_save(cls, reference_image, target_image, save_path):
        """
        並列用: ドリフト補正して保存する小分け関数
        """
        corrected = cls.correct_drift(reference_image, target_image)
        Image.fromarray(corrected).save(save_path)
        return save_path

    @classmethod
    def extract_timelapse_nd2(cls, file_name: str):
        """
        タイムラプスnd2ファイルをFieldごと・時系列ごとにTIFFで保存。
        (チャンネル0 -> fluo, チャンネル1 -> ph の想定)
        """
        base_output_dir = "uploaded_files/"
        # 出力用フォルダを作り直す
        if os.path.exists("TimelapseParserTemp"):
            shutil.rmtree("TimelapseParserTemp")
        os.makedirs("TimelapseParserTemp", exist_ok=True)

        nd2_fullpath = os.path.join(base_output_dir, file_name)
        with nd2reader.ND2Reader(nd2_fullpath) as images:
            # ★★★ ここがポイント： iter_axes をオフにする(または None にする)
            images.iter_axes = []

            # 必要に応じて bundle_axes を設定 (c,y,x) など
            images.bundle_axes = "yxc"  # 例: y,x,c

            print(f"Available axes: {images.axes}")
            print(f"Sizes: {images.sizes}")

            num_fields = images.sizes.get("v", 1)
            num_channels = images.sizes.get("c", 1)
            num_timepoints = images.sizes.get("t", 1)

            for field_idx in range(num_fields):
                field_folder = os.path.join(
                    "TimelapseParserTemp", f"Field_{field_idx + 1}"
                )
                os.makedirs(field_folder, exist_ok=True)
                base_output_subdir_ph = os.path.join(field_folder, "ph")
                base_output_subdir_fluo = os.path.join(field_folder, "fluo")
                os.makedirs(base_output_subdir_ph, exist_ok=True)
                os.makedirs(base_output_subdir_fluo, exist_ok=True)

                reference_image_ph = None
                reference_image_fluo = None

                # 並列タスクの蓄積先
                tasks = []
                with ThreadPoolExecutor() as executor:
                    for channel_idx in range(num_channels):
                        for time_idx in range(num_timepoints):
                            # ここで手動で指定する：v=field_idx, c=channel_idx, t=time_idx
                            # 2Dフレームを取り出す
                            frame_data = images.get_frame_2D(
                                v=field_idx, c=channel_idx, t=time_idx
                            )
                            # frame_data.shape は通常 (y, x)

                            # とりあえず channel_idx == 1 を ph (位相差) とし、
                            # それ以外を fluo とする例
                            channel_image = cls.process_image(frame_data)

                            if channel_idx == 1:  # ph
                                if time_idx == 0:
                                    reference_image_ph = channel_image
                                    tiff_filename = os.path.join(
                                        base_output_subdir_ph,
                                        f"time_{time_idx + 1}.tif",
                                    )
                                    Image.fromarray(channel_image).save(tiff_filename)
                                    print(f"Saved: {tiff_filename}")
                                else:
                                    tasks.append(
                                        executor.submit(
                                            cls._drift_and_save,
                                            reference_image_ph,
                                            channel_image,
                                            os.path.join(
                                                base_output_subdir_ph,
                                                f"time_{time_idx + 1}.tif",
                                            ),
                                        )
                                    )
                            else:  # fluo
                                if time_idx == 0:
                                    reference_image_fluo = channel_image
                                    tiff_filename = os.path.join(
                                        base_output_subdir_fluo,
                                        f"time_{time_idx + 1}.tif",
                                    )
                                    Image.fromarray(channel_image).save(tiff_filename)
                                    print(f"Saved: {tiff_filename}")
                                else:
                                    tasks.append(
                                        executor.submit(
                                            cls._drift_and_save,
                                            reference_image_fluo,
                                            channel_image,
                                            os.path.join(
                                                base_output_subdir_fluo,
                                                f"time_{time_idx + 1}.tif",
                                            ),
                                        )
                                    )

                    # すべての並列タスク完了を待つ
                    for future in as_completed(tasks):
                        try:
                            saved_path = future.result()
                            print(f"Saved (parallel): {saved_path}")
                        except Exception as e:
                            print(f"Error in parallel task: {e}")

    @classmethod
    def create_combined_gif(
        cls, field_folder: str, resize_factor: float = 0.5
    ) -> io.BytesIO:
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

        ph_images = [Image.open(img_file) for img_file in ph_image_files]
        fluo_images = [Image.open(img_file) for img_file in fluo_image_files]

        combined_images = []
        for ph_img, fluo_img in zip(ph_images, fluo_images):
            ph_img_resized = ph_img.resize(
                (int(ph_img.width * resize_factor), int(ph_img.height * resize_factor))
            )
            fluo_img_resized = fluo_img.resize(
                (
                    int(fluo_img.width * resize_factor),
                    int(fluo_img.height * resize_factor),
                )
            )

            combined_width = ph_img_resized.width + fluo_img_resized.width
            combined_height = max(ph_img_resized.height, fluo_img_resized.height)
            combined_img = Image.new("RGB", (combined_width, combined_height))
            combined_img.paste(ph_img_resized, (0, 0))
            combined_img.paste(fluo_img_resized, (ph_img_resized.width, 0))
            combined_images.append(combined_img)

        if not combined_images:
            raise ValueError("No images found to create combined GIF.")

        gif_buffer = io.BytesIO()
        combined_images[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=combined_images[1:],
            duration=100,
            loop=0,
        )
        gif_buffer.seek(0)
        return gif_buffer


class AsyncChores:
    """
    SyncChores の処理を非同期にラップするクラス
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor()

    async def correct_drift(self, reference_image, target_image):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            partial(SyncChores.correct_drift, reference_image, target_image),
        )

    async def process_image(self, array):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, partial(SyncChores.process_image, array)
        )

    async def extract_timelapse_nd2(self, file_name: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, partial(SyncChores.extract_timelapse_nd2, file_name)
        )

    async def shutdown(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.executor.shutdown)

    async def create_combined_gif(self, field_folder: str) -> io.BytesIO:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, partial(SyncChores.create_combined_gif, field_folder)
        )


class TimelapseEngineCrudBase:
    """
    ND2 -> TIFF(タイムラプス分割) -> GIF作成 といった一連の流れを実装するクラスの例
    """

    def __init__(self, nd2_path: str):
        self.nd2_path = nd2_path

    async def get_nd2_filenames(self) -> list[str]:
        return [i for i in os.listdir("uploaded_files") if i.endswith("_timelapse.nd2")]

    async def delete_nd2_file(self, file_path: str):
        filename = file_path.split("/")[-1]
        await asyncio.to_thread(os.remove, f"uploaded_files/{filename}")
        return True

    async def main(self):
        # ND2ファイルを tiff に分割保存 (ドリフト補正込み)
        await AsyncChores().extract_timelapse_nd2(self.nd2_path)
        return JSONResponse(content={"message": "Timelapse extracted"})

    async def create_combined_gif(self, field: str):
        # 指定したFieldフォルダからGIFを生成
        return await AsyncChores().create_combined_gif("TimelapseParserTemp/" + field)

    async def get_fields_of_nd2(self) -> list[str]:
        """
        ND2ファイルを軽く読み込み、'v' 軸のサイズから Field 数を取得して
        ["Field_1", "Field_2", ...] のように返す。
        """
        base_output_dir = "uploaded_files/"
        nd2_fullpath = os.path.join(base_output_dir, self.nd2_path)
        if not os.path.exists(nd2_fullpath):
            return []

        with nd2reader.ND2Reader(nd2_fullpath) as images:
            num_fields = images.sizes.get("v", 1)
            return [f"Field_{i+1}" for i in range(num_fields)]
