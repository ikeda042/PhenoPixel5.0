import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
import io
import math
import os
import pickle
import re
import shutil
from functools import partial
from PIL import Image, ImageDraw, ImageFont
from typing import Literal
import cv2
import nd2reader
import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import BLOB, Column, FLOAT, Integer, String, delete, update, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import select
import os
from CellDBConsole.crud import AsyncChores as CellDBAsyncChores
from ulid import ULID


def get_ulid() -> str:
    return str(ULID())


Base = declarative_base()


class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(String)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)

    # --- 新規追加: field, time, cell の3カラム ---
    field = Column(String, nullable=True)  # 例: "Field_1"
    time = Column(Integer, nullable=True)  # 例: 1, 2, 3, ...
    cell = Column(Integer, nullable=True)  # 例: 同一タイム内のセル番号
    base_cell_id = Column(String, nullable=True)

    # 死細胞判定用カラム
    is_dead = Column(Integer, nullable=True)

    # GIFを保存するためのカラム
    gif_ph = Column(BLOB, nullable=True)
    gif_fluo1 = Column(BLOB, nullable=True)
    gif_fluo2 = Column(BLOB, nullable=True)


@asynccontextmanager
async def get_session(dbname: str):
    """
    セッションを非同期コンテキストマネージャとして返す関数。
    """
    engine = create_async_engine(
        f"sqlite+aiosqlite:///timelapse_databases/{dbname}?timeout=30", echo=False
    )
    AsyncSessionLocal = sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    async with AsyncSessionLocal() as session:
        yield session


async def create_database(dbname: str):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///timelapse_databases/{dbname}?timeout=30", echo=True
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(
            lambda sync_conn: sync_conn.execute(text("PRAGMA journal_mode=WAL;"))
        )
    return engine


class SyncChores:
    """
    既存の補助的クラス。各種静的メソッドで画像処理を行う。
    """

    @staticmethod
    def correct_drift(reference_image, target_image):
        """
        ORB + BFMatcher + 座標反転してからアフィン変換を推定する例。
        特徴点ベースのため、細胞分裂で大きく見え方が変化した場合は、
        十分に対応しきれない可能性あり。
        """
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(reference_image, None)
        kp2, des2 = orb.detectAndCompute(target_image, None)

        if des1 is None or des2 is None:
            print("Descriptor is None, skipping drift correction.")
            return target_image

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # マッチングがあまりに少ない場合は補正を行わない
        if len(matches) < 300:
            print("Insufficient matches, skipping drift correction.")
            return target_image

        # ORB特徴点の座標リスト
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 画像の縦横サイズ
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

    # --- [ECC 法追加箇所] ---
    @staticmethod
    def correct_drift_ecc(reference_image, target_image, warp_mode=cv2.MOTION_AFFINE):
        """
        ECC (Enhanced Correlation Coefficient) によるドリフト補正。
        同一フレームの全画素が持つ情報を相関最大化するようにアライメントするため、
        細胞が多く写っている/分裂してもある程度頑健に補正が可能。

        warp_mode (int):
          - cv2.MOTION_TRANSLATION: 平行移動のみ
          - cv2.MOTION_EUCLIDEAN: 回転＋平行移動
          - cv2.MOTION_AFFINE: アフィン変換
          - cv2.MOTION_HOMOGRAPHY: 射影変換

        細胞が大きく回転するシーンが少なければ MOTION_TRANSLATION や MOTION_AFFINE が軽量かつ安定しやすい。
        """
        # ECC ではグレースケール画像が前提
        ref_gray = (
            reference_image
            if len(reference_image.shape) == 2
            else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        )
        tgt_gray = (
            target_image
            if len(target_image.shape) == 2
            else cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        )

        # 初期変換行列を設定
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # 終了条件 (反復回数と閾値)
        number_of_iterations = 100
        termination_eps = 1e-4
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            number_of_iterations,
            termination_eps,
        )

        try:
            # ECC の計算を実行
            cc, warp_matrix = cv2.findTransformECC(
                ref_gray, tgt_gray, warp_matrix, warp_mode, criteria
            )
        except cv2.error as e:
            print(f"[ECC] Alignment failed with OpenCV error: {e}")
            return target_image

        # warp_mode に応じて変換
        h, w = reference_image.shape[:2]
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # homography なら 3x3 行列を warpPerspective で適用
            aligned = cv2.warpPerspective(
                target_image,
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )
        else:
            # それ以外は warpAffine で適用
            aligned = cv2.warpAffine(
                target_image,
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )

        return aligned

    # --- [ECC 法追加箇所] ---

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
            # 全てが同値の場合は真っ黒にする
            return (array * 0).astype(np.uint8)
        array /= diff
        array *= 255
        return array.astype(np.uint8)

    @classmethod
    def extract_timelapse_nd2(cls, file_name: str, drift_method: str = "orb"):
        """
        タイムラプス nd2 ファイルを Field ごと・時系列ごとに TIFF で保存（補正後を上書き保存）。
        チャンネル 0 -> ph, チャンネル 1 -> fluo1, チャンネル 2 -> fluo2 の想定。

        drift_method (str): "orb" or "ecc"
          - "orb": ORB + BFMatcher の既存ロジック
          - "ecc": ECC による画素相関ベースのアライメント
        """
        base_output_dir = "uploaded_files/"
        # 既存の作業用フォルダがあれば削除し、再作成
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
            num_channels = images.sizes.get("c", 1)  # 3 チャンネルを想定
            num_timepoints = images.sizes.get("t", 1)

            # Field(視野)ごとに処理
            for field_idx in range(num_fields):
                field_folder = os.path.join(
                    "TimelapseParserTemp", f"Field_{field_idx + 1}"
                )
                os.makedirs(field_folder, exist_ok=True)

                # チャンネルごとにサブディレクトリを作成
                base_output_subdir_ph = os.path.join(field_folder, "ph")
                base_output_subdir_fluo1 = os.path.join(field_folder, "fluo1")
                base_output_subdir_fluo2 = os.path.join(field_folder, "fluo2")
                os.makedirs(base_output_subdir_ph, exist_ok=True)
                os.makedirs(base_output_subdir_fluo1, exist_ok=True)
                os.makedirs(base_output_subdir_fluo2, exist_ok=True)

                # 3 チャンネル分の参照画像を保存する辞書
                reference_images = {
                    "ph": None,
                    "fluo1": None,
                    "fluo2": None,
                }

                # 並列実行のためのキュー
                tasks = []
                with ThreadPoolExecutor() as executor:
                    for channel_idx in range(num_channels):
                        for time_idx in range(num_timepoints):
                            # 2次元フレームを取得
                            frame_data = images.get_frame_2D(
                                v=field_idx, c=channel_idx, t=time_idx
                            )
                            channel_image = cls.process_image(frame_data)

                            # 保存先のパスを作成＆参照画像ラベルを決定
                            if channel_idx == 0:
                                channel_label = "ph"
                                save_dir = base_output_subdir_ph
                            elif channel_idx == 1:
                                channel_label = "fluo1"
                                save_dir = base_output_subdir_fluo1
                            else:  # channel_idx == 2
                                channel_label = "fluo2"
                                save_dir = base_output_subdir_fluo2

                            tiff_filename = os.path.join(
                                save_dir, f"time_{time_idx + 1}.tif"
                            )

                            # 1 フレーム目は参照画像をセット (ドリフト補正をスキップ)
                            if time_idx == 0:
                                reference_images[channel_label] = channel_image
                                Image.fromarray(channel_image).save(tiff_filename)
                                print(
                                    f"Saved ({channel_label}, first): {tiff_filename}"
                                )
                            else:
                                # 2フレーム目以降は drift 補正 (ORB or ECC 選択)
                                if drift_method == "ecc":
                                    drift_func = cls.correct_drift_ecc
                                else:
                                    drift_func = cls.correct_drift

                                tasks.append(
                                    executor.submit(
                                        cls._drift_and_save,
                                        reference_images[channel_label],
                                        channel_image,
                                        tiff_filename,
                                        drift_func,
                                    )
                                )

                    # 並列タスク完了待ち
                    for future in as_completed(tasks):
                        try:
                            saved_path = future.result()
                            print(f"Saved (parallel): {saved_path}")
                        except Exception as e:
                            print(f"Error in parallel task: {e}")

    @classmethod
    def _drift_and_save(cls, reference_image, target_image, save_path, drift_func):
        """
        並列用: ドリフト補正して上書き保存する小分け関数。
        drift_func: 使用するドリフト補正関数 (correct_drift or correct_drift_ecc)
        """
        corrected = drift_func(reference_image, target_image)
        Image.fromarray(corrected).save(save_path)
        return save_path

    @classmethod
    def create_combined_gif(
        cls, field_folder: str, resize_factor: float = 0.5
    ) -> io.BytesIO:
        """
        ph, fluo1, fluo2 の 3 つを横に並べて GIF をメモリ上 (BytesIO) に作成して返す。
        全チャンネル必須という想定。
        """

        ph_folder = os.path.join(field_folder, "ph")
        fluo1_folder = os.path.join(field_folder, "fluo1")
        fluo2_folder = os.path.join(field_folder, "fluo2")

        # 時間インデックスを取り出すための正規表現
        def extract_time_index(path: str) -> int:
            match = re.search(r"time_(\d+)\.tif", path)
            return int(match.group(1)) if match else 0

        # ph, fluo1, fluo2 でそれぞれのファイルをソート
        ph_image_files = sorted(
            [
                os.path.join(ph_folder, f)
                for f in os.listdir(ph_folder)
                if f.endswith(".tif")
            ],
            key=extract_time_index,
        )
        fluo1_image_files = sorted(
            [
                os.path.join(fluo1_folder, f)
                for f in os.listdir(fluo1_folder)
                if f.endswith(".tif")
            ],
            key=extract_time_index,
        )
        fluo2_image_files = sorted(
            [
                os.path.join(fluo2_folder, f)
                for f in os.listdir(fluo2_folder)
                if f.endswith(".tif")
            ],
            key=extract_time_index,
        )

        # 画像として読み込み
        ph_images = [Image.open(img_file) for img_file in ph_image_files]
        fluo1_images = [Image.open(img_file) for img_file in fluo1_image_files]
        fluo2_images = [Image.open(img_file) for img_file in fluo2_image_files]

        if not ph_images or not fluo1_images or not fluo2_images:
            raise ValueError("Not all channel images found to create combined GIF.")

        combined_images = []
        for ph_img, fluo1_img, fluo2_img in zip(ph_images, fluo1_images, fluo2_images):
            # リサイズ（thumbnailを使用してアスペクト比を保ちながらリサイズ）
            ph_img.thumbnail(
                (int(ph_img.width * resize_factor), int(ph_img.height * resize_factor))
            )
            fluo1_img.thumbnail(
                (
                    int(fluo1_img.width * resize_factor),
                    int(fluo1_img.height * resize_factor),
                )
            )
            fluo2_img.thumbnail(
                (
                    int(fluo2_img.width * resize_factor),
                    int(fluo2_img.height * resize_factor),
                )
            )

            # 横に並べて合成 (3 チャンネル)
            combined_width = ph_img.width + fluo1_img.width + fluo2_img.width
            combined_height = max(ph_img.height, fluo1_img.height, fluo2_img.height)
            combined_img = Image.new("RGB", (combined_width, combined_height))

            # 貼り付け
            offset_x = 0
            combined_img.paste(ph_img, (offset_x, 0))
            offset_x += ph_img.width
            combined_img.paste(fluo1_img, (offset_x, 0))
            offset_x += fluo1_img.width
            combined_img.paste(fluo2_img, (offset_x, 0))

            combined_images.append(combined_img)

        if not combined_images:
            raise ValueError("No images found to create combined GIF.")

        # GIF をメモリに書き込む
        gif_buffer = io.BytesIO()
        combined_images[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=combined_images[1:],
            duration=100,
            loop=0,
            optimize=True,  # GIFの最適化
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

    async def correct_drift_ecc(self, reference_image, target_image):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            partial(SyncChores.correct_drift_ecc, reference_image, target_image),
        )

    async def process_image(self, array):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, partial(SyncChores.process_image, array)
        )

    async def extract_timelapse_nd2(self, file_name: str, drift_method: str = "orb"):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            partial(SyncChores.extract_timelapse_nd2, file_name, drift_method),
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

    async def main(self, drift_method: str = "orb"):
        # ND2ファイルを TIFF に分割保存 (ドリフト補正込み) - drift_method を指定
        await AsyncChores().extract_timelapse_nd2(
            self.nd2_path, drift_method=drift_method
        )
        return JSONResponse(
            content={"message": f"Timelapse extracted with {drift_method}."}
        )

    async def create_combined_gif(self, field: str):
        # 指定した Field フォルダから 3ch GIF を生成
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

    async def extract_cells(
        self,
        field: str,
        dbname: str,
        param1: int = 90,
        min_area: int = 300,
        crop_size: int = 200,
    ):
        db_path = f"timelapse_databases/{dbname}"

        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}?timeout=30", echo=False
        )
        async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # 3チャンネルの TIFF ディレクトリ
        ph_folder = os.path.join("TimelapseParserTemp", field, "ph")
        fluo1_folder = os.path.join("TimelapseParserTemp", field, "fluo1")
        fluo2_folder = os.path.join("TimelapseParserTemp", field, "fluo2")

        # 時間インデックスを抜き出すヘルパー関数
        def extract_time_index(filename: str) -> int:
            match = re.search(r"time_(\d+)\.tif", filename)
            return int(match.group(1)) if match else 0

        # ph, fluo1, fluo2 ファイルリストを取得
        ph_files = (
            sorted(os.listdir(ph_folder), key=extract_time_index)
            if os.path.exists(ph_folder)
            else []
        )
        fluo1_files = (
            sorted(os.listdir(fluo1_folder), key=extract_time_index)
            if os.path.exists(fluo1_folder)
            else []
        )
        fluo2_files = (
            sorted(os.listdir(fluo2_folder), key=extract_time_index)
            if os.path.exists(fluo2_folder)
            else []
        )

        if not ph_files:
            print("No TIFF files found for ph.")
            return

        total_frames = len(ph_files)
        output_size = (crop_size, crop_size)

        # 前フレームで追跡していた細胞の中心座標管理用
        active_cells = {}
        next_cell_idx = 1
        base_ids = {}

        # 各フレームの処理
        for i, ph_file in enumerate(ph_files):
            ph_time_idx = extract_time_index(ph_file)
            ph_path = os.path.join(ph_folder, ph_file)

            # fluo1パスの取得
            fluo1_path = None
            candidates_fluo1 = [
                f for f in fluo1_files if extract_time_index(f) == ph_time_idx
            ]
            if candidates_fluo1:
                fluo1_path = os.path.join(fluo1_folder, candidates_fluo1[0])

            # fluo2パスの取得
            fluo2_path = None
            candidates_fluo2 = [
                f for f in fluo2_files if extract_time_index(f) == ph_time_idx
            ]
            if candidates_fluo2:
                fluo2_path = os.path.join(fluo2_folder, candidates_fluo2[0])

            ph_img = cv2.imread(ph_path, cv2.IMREAD_COLOR)
            fluo1_img = cv2.imread(fluo1_path, cv2.IMREAD_COLOR) if fluo1_path else None
            fluo2_img = cv2.imread(fluo2_path, cv2.IMREAD_COLOR) if fluo2_path else None

            if ph_img is None:
                print(f"Skipping because ph image not found: {ph_path}")
                continue

            height, width = ph_img.shape[:2]

            # ph のみを使って閾値処理 & 輪郭検出
            ph_gray = cv2.cvtColor(ph_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(ph_gray, param1, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(thresh, 0, 130)
            contours, hierarchy = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            new_active_cells = {}

            # セッションを開いてDBに書き込み
            async with async_session() as session:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # 画像の上下左右をそれぞれ10%除外
                    x_min = int(width * 0.1)
                    x_max = int(width * 0.9)
                    y_min = int(height * 0.1)
                    y_max = int(height * 0.9)

                    if not (x_min < cx < x_max and y_min < cy < y_max):
                        continue

                    assigned_cell_idx = None
                    min_dist = float("inf")
                    distance_threshold = 50

                    # 前フレームの細胞中心との距離を見て同一セルかどうか判定
                    for prev_idx, (px, py) in active_cells.items():
                        dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                        if dist < distance_threshold and dist < min_dist:
                            min_dist = dist
                            assigned_cell_idx = prev_idx

                    # 新規のセルとして割り当て
                    if assigned_cell_idx is None:
                        assigned_cell_idx = next_cell_idx
                        next_cell_idx += 1

                    # 今フレームで追跡対象登録
                    if assigned_cell_idx not in new_active_cells:
                        new_active_cells[assigned_cell_idx] = (cx, cy)

                    # クロップ領域を算出
                    x1 = max(0, cx - output_size[0] // 2)
                    y1 = max(0, cy - output_size[1] // 2)
                    x2 = min(ph_img.shape[1], cx + output_size[0] // 2)
                    y2 = min(ph_img.shape[0], cy + output_size[1] // 2)

                    # クロップが指定サイズに満たない場合はスキップ
                    if (y2 - y1) != output_size[1] or (x2 - x1) != output_size[0]:
                        continue

                    cropped_ph = ph_img[y1:y2, x1:x2]
                    cropped_ph_gray = cv2.cvtColor(cropped_ph, cv2.COLOR_BGR2GRAY)
                    ph_gray_encode = cv2.imencode(".png", cropped_ph_gray)[1].tobytes()

                    # fluo1
                    fluo1_gray_encode = None
                    if fluo1_img is not None:
                        cropped_fluo1 = fluo1_img[y1:y2, x1:x2]
                        cropped_fluo1_gray = cv2.cvtColor(
                            cropped_fluo1, cv2.COLOR_BGR2GRAY
                        )
                        fluo1_gray_encode = cv2.imencode(".png", cropped_fluo1_gray)[
                            1
                        ].tobytes()

                    # fluo2
                    fluo2_gray_encode = None
                    if fluo2_img is not None:
                        cropped_fluo2 = fluo2_img[y1:y2, x1:x2]
                        cropped_fluo2_gray = cv2.cvtColor(
                            cropped_fluo2, cv2.COLOR_BGR2GRAY
                        )
                        fluo2_gray_encode = cv2.imencode(".png", cropped_fluo2_gray)[
                            1
                        ].tobytes()

                    # 輪郭をクロップ座標系へシフト
                    contour_shifted = contour.copy()
                    contour_shifted[:, :, 0] -= x1
                    contour_shifted[:, :, 1] -= y1

                    # 毎フレームごとにユニークな cell_id を生成
                    new_ulid = get_ulid()

                    # もしこのセル番号の base_cell_id が未登録なら、新しく登録
                    if assigned_cell_idx not in base_ids:
                        base_ids[assigned_cell_idx] = new_ulid

                    # DB 保存用のオブジェクト作成
                    cell_obj = Cell(
                        cell_id=new_ulid,
                        label_experiment=field,
                        manual_label="N/A",
                        perimeter=perimeter,
                        area=area,
                        img_ph=ph_gray_encode,
                        img_fluo1=fluo1_gray_encode,
                        img_fluo2=fluo2_gray_encode,
                        contour=pickle.dumps(contour_shifted),
                        center_x=cx,
                        center_y=cy,
                        field=field,
                        time=i + 1,
                        cell=assigned_cell_idx,
                        base_cell_id=base_ids[assigned_cell_idx],
                        is_dead=0,
                        gif_ph=None,
                        gif_fluo1=None,
                        gif_fluo2=None,
                    )

                    # 重複チェック (念のため)
                    existing = await session.execute(
                        select(Cell).filter_by(cell_id=new_ulid, time=i + 1)
                    )
                    if existing.scalar() is None:
                        session.add(cell_obj)

                # フレーム内処理が終わったらコミット
                await session.commit()

            # 次のフレーム用に active_cells を更新
            active_cells = new_active_cells

        # 最初のフレームに存在しない細胞を削除
        async with async_session() as session:
            first_frame = 1

            # 最初のフレームに存在する細胞の一覧を取得する
            subquery = select(Cell.cell).where(
                Cell.field == field,
                Cell.time == first_frame,
            )
            result = await session.execute(subquery)
            cells_with_first_frame = [row[0] for row in result]

            # 最初のフレームに存在しない細胞を削除する
            delete_stmt = (
                delete(Cell)
                .where(Cell.field == field)
                .where(~Cell.cell.in_(cells_with_first_frame))
            )
            await session.execute(delete_stmt)
            await session.commit()

        print("Cell extraction finished (with cropping).")
        print("Removed cells that did not appear in every frame.")

        """
        ここから、各 base セルに対して GIF を作成して BLOB として保存する処理
        self.create_gif_for_cell を使う
        """
        async with async_session() as session:
            # time=1 のレコードが「ベース」として扱われる想定
            base_cells = await session.execute(select(Cell).where(Cell.time == 1))
            base_cells = base_cells.scalars().all()
            base_cell_ids = [cell.cell_id for cell in base_cells]

            print(base_cell_ids)

            for base_id in base_cell_ids:
                print(base_id)
                for channel in ["ph", "fluo1", "fluo2"]:
                    cells = await session.execute(
                        select(Cell).filter_by(cell_id=base_id)
                    )
                    cell = cells.scalar()

                    gif_buffer = await self.create_gif_for_cell(
                        field=cell.field,
                        cell_number=cell.cell,
                        dbname=dbname,
                        channel=channel,
                        duration_ms=200,
                    )
                    gif_binary = gif_buffer.getvalue()

                    if channel == "ph":
                        cell.gif_ph = gif_binary
                    elif channel == "fluo1":
                        cell.gif_fluo1 = gif_binary
                    else:
                        cell.gif_fluo2 = gif_binary

                    await session.commit()

        return

    async def create_gif_for_cell(
        self,
        field: str,
        cell_number: int,
        dbname: str,
        channel: str = "ph",
        duration_ms: int = 200,
    ):
        if channel not in ["ph", "fluo1", "fluo2"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid channel. Use 'ph', 'fluo1', or 'fluo2'.",
            )

        engine = create_async_engine(
            f"sqlite+aiosqlite:///timelapse_databases/{dbname}?timeout=30", echo=False
        )
        async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        # session の中で処理を行う
        async with async_session() as session:
            # cell_number と field が一致するレコードを1フレーム目で検索
            result = await session.execute(
                select(Cell).filter_by(field=field, cell=cell_number, time=1)
            )
            cell = result.scalar()
            if cell is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for field={field}, cell={cell_number}",
                )

            # 既に GIF が作成済みならそれを返す
            if channel == "ph" and cell.gif_ph:
                return io.BytesIO(cell.gif_ph)
            elif channel == "fluo1" and cell.gif_fluo1:
                return io.BytesIO(cell.gif_fluo1)
            elif channel == "fluo2" and cell.gif_fluo2:
                return io.BytesIO(cell.gif_fluo2)

            # GIF 用フレーム作成
            frames = []
            result = await session.execute(
                select(Cell)
                .filter_by(field=field, cell=cell_number)
                .order_by(Cell.time)
            )
            cells = result.scalars().all()

            if not cells:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for field={field}, cell={cell_number}",
                )

            for row in cells:
                if channel == "ph":
                    img_binary = row.img_ph
                elif channel == "fluo1":
                    img_binary = row.img_fluo1
                else:  # channel == "fluo2"
                    img_binary = row.img_fluo2

                if img_binary is None:
                    continue

                np_img_gray = cv2.imdecode(
                    np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if np_img_gray is None:
                    continue

                # OpenCVで扱うために BGR 画像に変換
                np_img_color = cv2.cvtColor(np_img_gray, cv2.COLOR_GRAY2BGR)

                # 輪郭描画
                if row.contour is not None:
                    try:
                        contours = pickle.loads(row.contour)
                        if not isinstance(contours, list):
                            contours = [contours]

                        for c in contours:
                            c = np.array(c, dtype=np.float32)
                            if len(c.shape) == 2:
                                # OpenCV の drawContours の仕様に合わせて [N,1,2] に整形
                                c = c[:, np.newaxis, :]

                            # create_gif_for_cell では特にリサイズを行っていないため、
                            # スケールは (1.0, 1.0) のままとする
                            c = c.astype(np.int32)
                            cv2.drawContours(np_img_color, [c], -1, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"[WARN] Contour parse error: {e}")

                # OpenCV BGR → PIL RGB
                np_img_rgb = cv2.cvtColor(np_img_color, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(np_img_rgb)
                frames.append(pil_img)

            if not frames:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No valid frames found for field={field}, "
                        f"cell={cell_number}, channel={channel}"
                    ),
                )

            gif_buffer = io.BytesIO()
            frames[0].save(
                gif_buffer,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )
            gif_buffer.seek(0)
            return gif_buffer

    async def create_gif_for_cells(
        self,
        field: str,
        dbname: str,
        channel: Literal["ph", "fluo1", "fluo2"] = "ph",
        duration_ms: int = 200,
    ):
        # channel 入力チェック
        if channel not in ["ph", "fluo1", "fluo2"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid channel. Use 'ph', 'fluo1', or 'fluo2'.",
            )

        engine = create_async_engine(
            f"sqlite+aiosqlite:///timelapse_databases/{dbname}?timeout=30", echo=False
        )
        async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        async with async_session() as session:
            all_cells_query = select(Cell).order_by(
                Cell.time.asc(), Cell.cell.asc(), Cell.time == 1
            )
            if field != "all":
                all_cells_query = all_cells_query.filter_by(field=field)
            result = await session.execute(all_cells_query)
            all_cells = result.scalars().all()

        if not all_cells:
            raise HTTPException(
                status_code=404,
                detail=f"No cells found for field={field}",
            )

        unique_times = sorted(list(set(c.time for c in all_cells)))
        unique_cells = sorted(
            list(set(c.cell for c in all_cells if c.cell is not None))
        )

        if not unique_cells:
            raise HTTPException(
                status_code=404,
                detail=f"No valid cell numbering found for field={field}",
            )

        num_cells = len(unique_cells)
        rows = int(math.floor(math.sqrt(num_cells)))
        cols = int(math.ceil(num_cells / rows))

        cell_positions = {}
        for i, cell_num in enumerate(unique_cells):
            row_idx = i // cols
            col_idx = i % cols
            cell_positions[cell_num] = (row_idx, col_idx)

        first_valid_img_size = None
        frames_for_gif = []

        for t in unique_times:
            cell_to_image = {}
            for cell_num in unique_cells:
                matched = [c for c in all_cells if c.cell == cell_num and c.time == t]
                if not matched:
                    # 空画像(真っ黒 200x200)
                    cell_to_image[cell_num] = Image.new("RGB", (200, 200), (0, 0, 0))
                    continue

                row = matched[0]
                if channel == "ph":
                    img_binary = row.img_ph
                elif channel == "fluo1":
                    img_binary = row.img_fluo1
                else:
                    img_binary = row.img_fluo2

                if img_binary is None:
                    cell_to_image[cell_num] = Image.new("RGB", (200, 200), (0, 0, 0))
                    continue

                np_img_gray = cv2.imdecode(
                    np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if np_img_gray is None:
                    cell_to_image[cell_num] = Image.new("RGB", (200, 200), (0, 0, 0))
                    continue

                # ここでカラー変換
                np_img_color = cv2.cvtColor(np_img_gray, cv2.COLOR_GRAY2BGR)
                original_h, original_w = np_img_gray.shape[:2]

                if first_valid_img_size is None:
                    first_valid_img_size = (original_w, original_h)

                target_w, target_h = first_valid_img_size

                # リサイズ
                if (original_w, original_h) != (target_w, target_h):
                    scale_x = target_w / original_w
                    scale_y = target_h / original_h
                    np_img_color = cv2.resize(
                        np_img_color, (target_w, target_h), interpolation=cv2.INTER_AREA
                    )
                else:
                    scale_x, scale_y = 1.0, 1.0

                # 輪郭描画
                if row.contour is not None:
                    try:
                        contours = pickle.loads(row.contour)
                        if not isinstance(contours, list):
                            contours = [contours]

                        for c in contours:
                            c = np.array(c, dtype=np.float32)
                            if len(c.shape) == 2:
                                c = c[:, np.newaxis, :]

                            c[:, :, 0] *= scale_x
                            c[:, :, 1] *= scale_y
                            c = c.astype(np.int32)
                            cv2.drawContours(np_img_color, [c], -1, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"[WARN] Contour parse error: {e}")

                # OpenCV BGR → PIL RGB
                np_img_rgb = cv2.cvtColor(np_img_color, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(np_img_rgb)
                cell_to_image[cell_num] = pil_img

            if not first_valid_img_size:
                raise HTTPException(
                    status_code=404,
                    detail="No valid images found to determine size.",
                )

            w, h = first_valid_img_size
            mosaic_w = cols * w
            mosaic_h = rows * h

            mosaic = Image.new("RGB", (mosaic_w, mosaic_h), (0, 0, 0))
            for cell_num, pil_img in cell_to_image.items():
                r_idx, c_idx = cell_positions[cell_num]
                mosaic.paste(pil_img, (c_idx * w, r_idx * h))

            frames_for_gif.append(mosaic)

        if not frames_for_gif:
            raise HTTPException(
                status_code=404,
                detail=f"No frames created for field={field}",
            )

        gif_buffer = io.BytesIO()
        frames_for_gif[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=frames_for_gif[1:],
            duration=duration_ms,
            loop=0,
        )
        gif_buffer.seek(0)
        return gif_buffer


class TimelapseDatabaseCrud:
    def __init__(self, dbname: str):
        self.dbname = dbname

    async def get_cells_by_field(self, field: str) -> list[Cell]:
        async with get_session(self.dbname) as session:
            result = await session.execute(select(Cell).filter_by(field=field))
            return result.scalars().all()

    async def get_cell_by_id(self, cell_id: str) -> Cell:
        async with get_session(self.dbname) as session:
            result = await session.execute(select(Cell).filter_by(cell_id=cell_id))
            return result.scalar_one()

    async def get_cells_by_cell_number(
        self, field: str, cell_number: int
    ) -> list[Cell]:
        async with get_session(self.dbname) as session:
            result = await session.execute(
                select(Cell).filter_by(field=field, cell=cell_number)
            )
            return result.scalars().all()

    async def get_cells_gif_by_cell_number(
        self,
        field: str,
        cell_number: int,
        channel: str,
        draw_contour: bool = True,
        draw_frame_number: bool = True,  # 追加
    ) -> io.BytesIO:
        """
        指定した field, cell_number, channel のセル画像を順番に GIF 化して返す。
        draw_contour=True の場合は DB に格納されている contour を描画した画像を返す。
        draw_frame_number=True の場合はフレーム左上にフレーム番号を描画する。
        """
        # セッションからデータ取得
        async with get_session(self.dbname) as session:
            result = await session.execute(
                select(Cell)
                .filter_by(field=field, cell=cell_number)
                .order_by(Cell.time)
            )
            cells: list[Cell] = result.scalars().all()

        if not cells:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for field={field}, cell={cell_number}",
            )

        frames = []
        for i, row in enumerate(cells):
            # チャネルごとの画像バイナリを取得
            if channel == "ph":
                img_binary = row.img_ph
            elif channel == "fluo1":
                img_binary = row.img_fluo1
            else:
                img_binary = row.img_fluo2

            if img_binary is None:
                continue

            # バイナリ -> NumPy 配列 (グレースケール)
            np_img = cv2.imdecode(
                np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if np_img is None:
                continue

            # contour 描画フラグが立っている場合
            if draw_contour and row.contour is not None:
                try:
                    # pickle で保存されている contour 情報を想定
                    contour_data = pickle.loads(row.contour)
                    # グレースケール -> BGR に変換
                    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
                    # 単一/複数の輪郭を想定して描画
                    cv2.drawContours(
                        bgr_img,
                        [np.array(contour_data, dtype=np.int32)],
                        -1,
                        (0, 255, 0),  # 緑色
                        1,
                    )
                    # Pillow 用に BGR -> RGB に変換
                    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                except Exception:
                    # contour のフォーマットが合わない場合などはエラーを無視して通常表示
                    pil_img = Image.fromarray(np_img)
            else:
                # 通常のグレースケール画像をそのまま PIL に渡す
                pil_img = Image.fromarray(np_img)

            # フレーム番号描画フラグが True の場合
            if draw_frame_number:
                draw = ImageDraw.Draw(pil_img)
                # 既定のフォントを使用 (適宜変更可能)
                font = ImageFont.load_default()
                # 左上に i+1 の番号を描画
                draw.text((5, 5), f"Frame: {i+1}", font=font, fill=(255, 255, 255))

            frames.append(pil_img)

        if not frames:
            raise HTTPException(
                status_code=404,
                detail=f"No valid frames found for field={field}, cell={cell_number}, channel={channel}",
            )

        # GIF バッファ作成
        gif_buffer = io.BytesIO()
        frames[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
        gif_buffer.seek(0)
        return gif_buffer

    async def get_database_names(self) -> list[str]:
        return [i for i in os.listdir("timelapse_databases") if i.endswith(".db")]

    async def get_fields_of_db(self) -> list[str]:
        async with get_session(self.dbname) as session:
            result = await session.execute(select(Cell.field).distinct())
            return [row[0] for row in result]

    async def get_cell_numbers_of_field(self, field: str) -> list[int]:
        async with get_session(self.dbname) as session:
            result = await session.execute(
                select(Cell.cell).filter_by(field=field).distinct()
            )
            return [row[0] for row in result]

    async def update_manual_label(self, base_cell_id: str, label: str):
        async with get_session(self.dbname) as session:
            result = await session.execute(
                update(Cell)
                .where(Cell.base_cell_id == base_cell_id)
                .values(manual_label=label)
            )
            await session.commit()
            return result.rowcount

    async def update_dead_status(self, base_cell_id: str, is_dead: int):
        async with get_session(self.dbname) as session:
            result = await session.execute(
                update(Cell)
                .where(Cell.base_cell_id == base_cell_id)
                .values(is_dead=is_dead)
            )
            await session.commit()
            return result.rowcount

    async def get_contour_areas_by_cell_number(
        self, field: str, cell_number: int
    ) -> list[float]:
        """
        指定した field, cell_number のセルをタイム順に取得し、
        各フレームに対して contour の面積を算出したリストを返す。
        複数輪郭が入っている場合は合計面積を返すようにしている。
        """
        async with get_session(self.dbname) as session:
            result = await session.execute(
                select(Cell)
                .filter_by(field=field, cell=cell_number)
                .order_by(Cell.time)
            )
            cells: list[Cell] = result.scalars().all()

        if not cells:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for field={field}, cell={cell_number}",
            )
        areas = []
        for row in cells:
            contour_area = 0.0
            if row.contour is not None:
                try:
                    contour_data = pickle.loads(row.contour)
                    # 配列かつ多輪郭のリストの場合
                    if (
                        isinstance(contour_data, list)
                        and contour_data
                        and isinstance(contour_data[0], (list, np.ndarray))
                    ):
                        for cnt in contour_data:
                            contour_area += cv2.contourArea(
                                np.array(cnt, dtype=np.int32)
                            )
                    else:
                        # 単一輪郭の場合
                        contour_area = cv2.contourArea(
                            np.array(contour_data, dtype=np.int32)
                        )
                except Exception:
                    contour_area = 0.0
            areas.append(contour_area)

        return areas

    async def replot_cell(
        self,
        field: str,
        cell_number: int,
        channel: str,
        degree: int,
    ) -> io.BytesIO:
        """
        指定した field, cell_number, channel の全フレームを取得し、
        replot で生成した画像を GIF 化して返すサンプル関数。
        キャッシュをクリアした上で再度 replot を行う。
        """

        # データベースから frame の一覧を time 昇順に取得
        async with get_session(self.dbname) as session:
            result = await session.execute(
                select(Cell)
                .filter_by(field=field, cell=cell_number)
                .order_by(Cell.time)
            )
            cells = result.scalars().all()

        if not cells:
            raise HTTPException(
                status_code=404,
                detail=f"No cells found for field={field}, cell_number={cell_number}",
            )

        frames: list[Image.Image] = []

        for i, cell in enumerate(cells):
            # チャネルごとの画像バイナリを取得 (PH 以外を想定)
            if channel == "ph":
                image_fluo_raw = cell.img_ph
            elif channel == "fluo1":
                image_fluo_raw = cell.img_fluo1
            else:
                image_fluo_raw = cell.img_fluo2

            if not image_fluo_raw:
                raise HTTPException(
                    status_code=404,
                    detail=f"No {channel} data found for field={field}, cell={cell_number} (frame index={i})",
                )

            if not cell.contour:
                raise HTTPException(
                    status_code=404,
                    detail=f"No contour data found for field={field}, cell={cell_number} (frame index={i})",
                )

            # replot 呼び出し前にキャッシュをクリア
            if hasattr(CellDBAsyncChores.replot, "cache_clear"):
                CellDBAsyncChores.replot.cache_clear()

            # replot 関数を実行し、返ってきた画像を io.BytesIO として受け取る
            buf = await CellDBAsyncChores.replot(image_fluo_raw, cell.contour, degree)
            buf.seek(0)

            # Pillow Image として開き、frames に追加
            frames.append(Image.open(buf))

        if not frames:
            raise HTTPException(
                status_code=404,
                detail=f"No frames were generated for field={field}, cell={cell_number}",
            )

        # GIF 用の BytesIO を作成
        gif_buf = io.BytesIO()

        # 先頭のフレームに残りを連結して GIF を書き込み
        frames[0].save(
            gif_buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=200,
        )

        gif_buf.seek(0)
        return gif_buf
