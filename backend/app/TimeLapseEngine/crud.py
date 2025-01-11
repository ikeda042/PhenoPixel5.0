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


class SyncChores:
    # ---- 高速化策 1: ORB と BFMatcher のインスタンス使い回し + パラメータ最適化 ----
    # nfeatures を減らし、計算量を抑える
    orb = cv2.ORB_create(nfeatures=500)
    # FLANNベースのバイナリ特徴量用IndexParams
    # BFMatcherから FLANNベースに変更する例 (必要がなければBFMatcherのままでもOK)
    index_params = dict(
        algorithm=6, table_number=6, key_size=12, multi_probe_level=1  # FLANN_INDEX_LSH
    )
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    @classmethod
    def correct_drift(cls, reference_image, target_image, scale=0.5):
        """
        画像を scale 倍にリサイズしてからORB特徴抽出 & マッチングし、
        求めたアフィン変換をオリジナルサイズに適用する。
        """
        # まずはリサイズ（整数にしないとcv2.resizeでエラーになることがあるので注意）
        new_h = int(reference_image.shape[0] * scale)
        new_w = int(reference_image.shape[1] * scale)
        if new_h < 2 or new_w < 2:
            # あまりに小さくなるなら、補正を諦めてそのまま返す
            return target_image

        ref_small = cv2.resize(
            reference_image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        tgt_small = cv2.resize(
            target_image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        kp1, des1 = cls.orb.detectAndCompute(ref_small, None)
        kp2, des2 = cls.orb.detectAndCompute(tgt_small, None)

        if des1 is None or des2 is None:
            print("Descriptor is None, skipping drift correction.")
            return target_image

        # ---- 高速化策2: FLANNベースに変更 (BFMatcherのままでもOK) ----
        # matches = cls.bf.match(des1, des2)
        matches = cls.flann.match(des1, des2)

        if len(matches) < 10:  # マッチングが少なすぎる場合、補正を行わない
            print("Insufficient matches, skipping drift correction.")
            return target_image

        # 特徴点の座標を取得
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # RANSACで部分アフィン行列を推定
        matrix, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )
        if matrix is not None:
            # 求めた変換行列をオリジナルサイズに合わせる
            # 例: 平行移動成分に scale の逆数をかける
            #     拡大縮小のスケールも戻す（warpAffineに与えるときはフルサイズにするため）
            scale_matrix = np.array(
                [[1 / scale, 0, 0], [0, 1 / scale, 0]], dtype=np.float32
            )
            # 2x3のアフィン行列同士の乗算
            full_matrix = scale_matrix @ np.vstack([matrix, [0, 0, 1]])

            # オリジナルサイズで変換
            aligned_image = cv2.warpAffine(
                target_image,
                full_matrix[:2, :],
                (target_image.shape[1], target_image.shape[0]),
            )
            return aligned_image
        else:
            print("Matrix estimation failed, skipping drift correction.")
            return target_image

    @classmethod
    def process_image(cls, array):
        """
        画像処理関数：正規化とスケーリングを行う。
        """
        array = array.astype(np.float32)  # Convert to float
        array_min = array.min()
        array_max = array.max()
        if array_max - array_min == 0:
            return (array * 0).astype(np.uint8)
        array -= array_min
        array /= array_max - array_min
        array *= 255
        return array.astype(np.uint8)

    @classmethod
    def extract_timelapse_nd2(cls, file_name: str):
        """
        タイムラプスnd2ファイルをフレームごとにTIFF形式で保存する。(チャンネル0がFluo画像、チャンネル1がPH画像)
        高速化策 3: ThreadPoolExecutorを使った並列処理(読み込み＆ドリフト補正など)の例。
        """
        base_output_dir = "uploaded_files/"
        os.makedirs("TimelapseParserTemp", exist_ok=True)

        nd2_fullpath = os.path.join(base_output_dir, file_name)
        with nd2reader.ND2Reader(nd2_fullpath) as images:
            print(f"Available axes: {images.axes}")
            print(f"Sizes: {images.sizes}")

            images.bundle_axes = "cyx" if "c" in images.axes else "yx"

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

                # ---- 並列化のためのリストを用意 ----
                tasks = []
                with ThreadPoolExecutor() as executor:
                    for channel_idx in range(num_channels):
                        for time_idx in range(num_timepoints):
                            # Frame取得
                            image_data = images.get_frame_2D(
                                v=field_idx, c=channel_idx, t=time_idx
                            )

                            if len(image_data.shape) == 3:
                                # 3D（Zスタックなど）の場合
                                for i in range(image_data.shape[0]):
                                    channel_image = cls.process_image(image_data[i])
                                    # ドリフト補正を並列で実行するタスクを追加
                                    if i == 1:  # PH
                                        if time_idx == 0:
                                            reference_image_ph = channel_image
                                            # time=0のものはそのまま保存
                                            tiff_filename = os.path.join(
                                                base_output_subdir_ph,
                                                f"time_{time_idx + 1}_channel_{i}.tif",
                                            )
                                            Image.fromarray(channel_image).save(
                                                tiff_filename
                                            )
                                            print(f"Saved: {tiff_filename}")
                                        else:
                                            # 並列タスクに登録
                                            tasks.append(
                                                executor.submit(
                                                    cls._drift_and_save,
                                                    reference_image_ph,
                                                    channel_image,
                                                    os.path.join(
                                                        base_output_subdir_ph,
                                                        f"time_{time_idx + 1}_channel_{i}.tif",
                                                    ),
                                                )
                                            )
                                    else:  # fluo
                                        if time_idx == 0:
                                            reference_image_fluo = channel_image
                                            tiff_filename = os.path.join(
                                                base_output_subdir_fluo,
                                                f"time_{time_idx + 1}_channel_{i}.tif",
                                            )
                                            Image.fromarray(channel_image).save(
                                                tiff_filename
                                            )
                                            print(f"Saved: {tiff_filename}")
                                        else:
                                            tasks.append(
                                                executor.submit(
                                                    cls._drift_and_save,
                                                    reference_image_fluo,
                                                    channel_image,
                                                    os.path.join(
                                                        base_output_subdir_fluo,
                                                        f"time_{time_idx + 1}_channel_{i}.tif",
                                                    ),
                                                )
                                            )
                            else:
                                # 2D の場合
                                channel_image = cls.process_image(image_data)
                                if time_idx == 0:
                                    reference_image_ph = channel_image
                                    tiff_filename = os.path.join(
                                        field_folder, f"time_{time_idx + 1}.tif"
                                    )
                                    Image.fromarray(channel_image).save(tiff_filename)
                                    print(f"Saved: {tiff_filename}")
                                else:
                                    # 並列タスクに登録
                                    tasks.append(
                                        executor.submit(
                                            cls._drift_and_save,
                                            reference_image_ph,
                                            channel_image,
                                            os.path.join(
                                                field_folder, f"time_{time_idx + 1}.tif"
                                            ),
                                        )
                                    )

                    # ---- 並列タスクを完了させる ----
                    for future in as_completed(tasks):
                        try:
                            saved_path = future.result()
                            print(f"Saved (parallel): {saved_path}")
                        except Exception as e:
                            print(f"Error in parallel task: {e}")

    @classmethod
    def _drift_and_save(cls, reference_image, target_image, save_path):
        """
        並列用: ドリフト補正して保存する小分け関数
        """
        corrected = cls.correct_drift(reference_image, target_image, scale=0.5)
        Image.fromarray(corrected).save(save_path)
        return save_path

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
            raise ValueError(
                "No images found to create combined GIF. Check ph/fluo folders."
            )

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
    def __init__(self, nd2_path: str):
        self.nd2_path = nd2_path

    async def get_nd2_filenames(self) -> list[str]:
        return [i for i in os.listdir("uploaded_files") if i.endswith("_timelapse.nd2")]

    async def delete_nd2_file(self, file_path: str):
        filename = file_path.split("/")[-1]
        await asyncio.to_thread(os.remove, f"uploaded_files/{filename}")
        return True

    async def main(self):
        # ND2ファイルを tiff に分割保存
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
