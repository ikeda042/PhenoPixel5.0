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
import re

# 追加
import pickle
from typing import Literal
import aiofiles
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import BLOB, Column, FLOAT, Integer, String
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import select
import ulid

Base = declarative_base()


# 既存の DB テーブル定義 (細胞情報を保存するための例)
class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)


async def get_session(dbname: str):
    engine = create_async_engine(f"sqlite+aiosqlite:///{dbname}?timeout=30", echo=False)
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session


async def create_database(dbname: str):
    engine = create_async_engine(f"sqlite+aiosqlite:///{dbname}?timeout=30", echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


class SyncChores:
    """
    既存の補助的クラス。
    """

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

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        height, width = reference_image.shape[:2]
        # x座標とy座標をそれぞれ反転
        src_pts[:, :, 0] = width - src_pts[:, :, 0]
        src_pts[:, :, 1] = height - src_pts[:, :, 1]
        dst_pts[:, :, 0] = width - dst_pts[:, :, 0]
        dst_pts[:, :, 1] = height - dst_pts[:, :, 1]

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
        if os.path.exists("TimelapseParserTemp"):
            shutil.rmtree("TimelapseParserTemp")
        os.makedirs("TimelapseParserTemp", exist_ok=True)

        nd2_fullpath = os.path.join(base_output_dir, file_name)
        with nd2reader.ND2Reader(nd2_fullpath) as images:
            images.iter_axes = []
            images.bundle_axes = "yxc"

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

                tasks = []
                with ThreadPoolExecutor() as executor:
                    for channel_idx in range(num_channels):
                        for time_idx in range(num_timepoints):
                            frame_data = images.get_frame_2D(
                                v=field_idx, c=channel_idx, t=time_idx
                            )
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

        def extract_time_index(path: str) -> int:
            match = re.search(r"time_(\d+)\.tif", path)
            return int(match.group(1)) if match else 0

        ph_image_files = [
            os.path.join(ph_folder, f)
            for f in os.listdir(ph_folder)
            if f.endswith(".tif")
        ]
        ph_image_files = sorted(ph_image_files, key=extract_time_index)

        fluo_image_files = [
            os.path.join(fluo_folder, f)
            for f in os.listdir(fluo_folder)
            if f.endswith(".tif")
        ]
        fluo_image_files = sorted(fluo_image_files, key=extract_time_index)

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
    ND2 -> TIFF(タイムラプス分割) -> GIF作成 といった流れを実装するクラスの例
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
        # ND2ファイルを TIFF に分割保存 (ドリフト補正込み)
        await AsyncChores().extract_timelapse_nd2(self.nd2_path)
        return JSONResponse(content={"message": "Timelapse extracted"})

    async def create_combined_gif(self, field: str):
        # 指定した Field フォルダから GIF を生成
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

    # ▼ ここから追加：extract_cellsメソッドの例 ▼
    async def extract_cells(
        self,
        field: str,
        dbname: str = "cells_timelapse.db",
        param1: int = 130,
        min_area: int = 300,
    ):
        """
        指定した Field (例: "Field_1") 内の全タイムポイントについて、
        ph画像・fluo画像を読み込み、輪郭抽出→セル情報抽出→DBに保存する。

        :param field: "Field_1" や "Field_2" のように指定
        :param dbname: SQLite の保存先DBファイル名
        :param param1: ２値化用の閾値
        :param min_area: セルとみなす最小面積
        """

        # DB作成 (すでに存在する場合は追記 or 先に消して作り直す等、用途に合わせて)
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{dbname}?timeout=30", echo=False
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)  # Cell テーブルを作成

        async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        # フォルダのパスを組み立て
        ph_folder = os.path.join("TimelapseParserTemp", field, "ph")
        fluo_folder = os.path.join("TimelapseParserTemp", field, "fluo")

        # time_{数字}.tif の数字部分を抽出してソートするための関数
        def extract_time_index(filename: str) -> int:
            match = re.search(r"time_(\d+)\.tif", filename)
            return int(match.group(1)) if match else 0

        # ph, fluo のTIFファイルリストを取得
        ph_files = [f for f in os.listdir(ph_folder) if f.endswith(".tif")]
        fluo_files = [f for f in os.listdir(fluo_folder) if f.endswith(".tif")]
        ph_files = sorted(ph_files, key=extract_time_index)
        fluo_files = sorted(fluo_files, key=extract_time_index)

        if not ph_files or not fluo_files:
            print("No TIFF files found for ph or fluo.")
            return

        # タイムポイント数だけループ
        for i, (ph_file, fluo_file) in enumerate(zip(ph_files, fluo_files)):
            ph_path = os.path.join(ph_folder, ph_file)
            fluo_path = os.path.join(fluo_folder, fluo_file)

            # 画像をOpenCVで読み込み
            ph_img = cv2.imread(ph_path, cv2.IMREAD_COLOR)
            fluo_img = cv2.imread(fluo_path, cv2.IMREAD_COLOR)

            if ph_img is None or fluo_img is None:
                print(f"Skipping because image not found: {ph_path}, {fluo_path}")
                continue

            # (1) 位相差をグレースケールに変換して閾値処理
            ph_gray = cv2.cvtColor(ph_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(ph_gray, param1, 255, cv2.THRESH_BINARY)
            # (2) Cannyなどで輪郭検出
            edges = cv2.Canny(thresh, 0, 150)
            contours, hierarchy = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # 面積フィルタ
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            # セル個数ループ
            async with async_session() as session:
                for c_idx, contour in enumerate(contours):
                    # セル情報計算
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    center_x = M["m10"] / M["m00"]
                    center_y = M["m01"] / M["m00"]

                    # IDなど好きなように付与
                    cell_id = f"field{field}_time{i+1}_cell{c_idx+1}"

                    # ph / fluo それぞれのグレースケール
                    ph_gray_encode = cv2.imencode(".png", ph_gray)[1].tobytes()
                    fluo_gray = cv2.cvtColor(fluo_img, cv2.COLOR_BGR2GRAY)
                    fluo_gray_encode = cv2.imencode(".png", fluo_gray)[1].tobytes()

                    # DB に格納 (Cell テーブル)
                    cell_obj = Cell(
                        cell_id=cell_id,
                        label_experiment=field,  # 例: Field名を実験ラベルとして登録
                        manual_label=-1,  # まだ未分類などの意味で -1
                        perimeter=perimeter,
                        area=area,
                        img_ph=ph_gray_encode,
                        img_fluo1=fluo_gray_encode,
                        img_fluo2=None,  # 今回は fluo2未使用とする
                        contour=pickle.dumps(contour),
                        center_x=center_x,
                        center_y=center_y,
                    )

                    # 既存チェックは適宜行う
                    # 例: 同じ cell_id が既にあるかどうかを確認してから追加
                    existing = await session.execute(
                        select(Cell).filter_by(cell_id=cell_id)
                    )
                    if existing.scalar() is None:
                        session.add(cell_obj)

                await session.commit()

        print("Cell extraction finished.")
        return

    # ▲ ここまでが追加メソッド ▲
