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

import pickle
from sqlalchemy import BLOB, Column, FLOAT, Integer, String
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import select
import math
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

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

    # --- 新規追加: field, time, cell の3カラム ---
    field = Column(String, nullable=True)  # 例: "Field_1"
    time = Column(Integer, nullable=True)  # 例: 1, 2, 3, ...
    cell = Column(Integer, nullable=True)  # 例: 同一タイム内のセル番号


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


import os
import re
import cv2
import io
import shutil
import numpy as np
import nd2reader
from PIL import Image
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed


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

        # マッチングがあまりに少ない場合は補正を行わない
        if len(matches) < 10:
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
    def extract_timelapse_nd2(cls, file_name: str):
        """
        タイムラプス nd2 ファイルを Field ごと・時系列ごとに TIFF で保存（補正後を上書き保存）。
        (チャンネル0 -> fluo, チャンネル1 -> ph の想定)
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
            num_channels = images.sizes.get("c", 1)
            num_timepoints = images.sizes.get("t", 1)

            # Field(視野)ごとに処理
            for field_idx in range(num_fields):
                field_folder = os.path.join(
                    "TimelapseParserTemp", f"Field_{field_idx + 1}"
                )
                os.makedirs(field_folder, exist_ok=True)
                base_output_subdir_ph = os.path.join(field_folder, "ph")
                base_output_subdir_fluo = os.path.join(field_folder, "fluo")
                os.makedirs(base_output_subdir_ph, exist_ok=True)
                os.makedirs(base_output_subdir_fluo, exist_ok=True)

                # ph, fluo それぞれの参照画像を初期化
                reference_image_ph = None
                reference_image_fluo = None

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

                            # 保存先のパスを作成
                            if channel_idx == 1:  # ph チャンネル
                                tiff_filename = os.path.join(
                                    base_output_subdir_ph,
                                    f"time_{time_idx + 1}.tif",
                                )
                                if time_idx == 0:
                                    # 1フレーム目：参照画像をセット
                                    reference_image_ph = channel_image
                                    # ※もし1フレーム目も補正したい場合は下の行をコメントアウトして、
                                    #   'corrected_image = cls.correct_drift(...)' を呼び出すように変更してください
                                    corrected_image = channel_image
                                    Image.fromarray(corrected_image).save(tiff_filename)
                                    print(f"Saved (ph, first): {tiff_filename}")
                                else:
                                    # 2フレーム目以降は drift 補正して上書き保存
                                    tasks.append(
                                        executor.submit(
                                            cls._drift_and_save,
                                            reference_image_ph,
                                            channel_image,
                                            tiff_filename,
                                        )
                                    )
                            else:  # fluo チャンネル
                                tiff_filename = os.path.join(
                                    base_output_subdir_fluo,
                                    f"time_{time_idx + 1}.tif",
                                )
                                if time_idx == 0:
                                    # 1フレーム目：参照画像をセット
                                    reference_image_fluo = channel_image
                                    # ※もし1フレーム目も補正したい場合は下の行をコメントアウトして、
                                    #   'corrected_image = cls.correct_drift(...)' を呼び出すように変更してください
                                    corrected_image = channel_image
                                    Image.fromarray(corrected_image).save(tiff_filename)
                                    print(f"Saved (fluo, first): {tiff_filename}")
                                else:
                                    # 2フレーム目以降は drift 補正して上書き保存
                                    tasks.append(
                                        executor.submit(
                                            cls._drift_and_save,
                                            reference_image_fluo,
                                            channel_image,
                                            tiff_filename,
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
    def _drift_and_save(cls, reference_image, target_image, save_path):
        """
        並列用: ドリフト補正して上書き保存する小分け関数。
        """
        corrected = cls.correct_drift(reference_image, target_image)
        Image.fromarray(corrected).save(save_path)
        return save_path

    @classmethod
    def create_combined_gif(
        cls, field_folder: str, resize_factor: float = 0.5
    ) -> io.BytesIO:
        """
        ph と fluo を横並びに合成した GIF をメモリ上 (BytesIO) に作成して返す。
        """
        ph_folder = os.path.join(field_folder, "ph")
        fluo_folder = os.path.join(field_folder, "fluo")

        # 時間インデックスを取り出すための正規表現
        def extract_time_index(path: str) -> int:
            match = re.search(r"time_(\d+)\.tif", path)
            return int(match.group(1)) if match else 0

        # ph と fluo でそれぞれのファイルをソート
        ph_image_files = sorted(
            [
                os.path.join(ph_folder, f)
                for f in os.listdir(ph_folder)
                if f.endswith(".tif")
            ],
            key=extract_time_index,
        )
        fluo_image_files = sorted(
            [
                os.path.join(fluo_folder, f)
                for f in os.listdir(fluo_folder)
                if f.endswith(".tif")
            ],
            key=extract_time_index,
        )

        # 画像として読み込み
        ph_images = [Image.open(img_file) for img_file in ph_image_files]
        fluo_images = [Image.open(img_file) for img_file in fluo_image_files]

        combined_images = []
        for ph_img, fluo_img in zip(ph_images, fluo_images):
            # リサイズ
            ph_img_resized = ph_img.resize(
                (int(ph_img.width * resize_factor), int(ph_img.height * resize_factor))
            )
            fluo_img_resized = fluo_img.resize(
                (
                    int(fluo_img.width * resize_factor),
                    int(fluo_img.height * resize_factor),
                )
            )

            # 横に並べて合成
            combined_width = ph_img_resized.width + fluo_img_resized.width
            combined_height = max(ph_img_resized.height, fluo_img_resized.height)
            combined_img = Image.new("RGB", (combined_width, combined_height))
            combined_img.paste(ph_img_resized, (0, 0))
            combined_img.paste(fluo_img_resized, (ph_img_resized.width, 0))
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

    async def extract_cells(
        self,
        field: str,
        dbname: str,
        param1: int = 90,
        min_area: int = 300,
        crop_size: int = 200,
    ):
        """
        指定した Field (例: "Field_1") 内の全タイムポイントについて、
        ph画像・fluo画像を読み込み、輪郭抽出 & セルクロップ → DBに保存する。

        前の time におけるセル中心と十分近い座標ならば同一セル (cell) と判定し、
        同じ cell カラム値を割り当てることで、タイムラプス上での同一細胞トラッキングを行う。
        """

        # DB作成 (すでに存在する場合は追記／用途に応じて)
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{dbname}?timeout=30", echo=False
        )
        async with engine.begin() as conn:
            await conn.run_sync(
                Base.metadata.create_all
            )  # Cell テーブル（拡張含む）を作成

        async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        # TimelapseParserTemp/
        #   Field_1/
        #     ph/time_1.tif, time_2.tif, ...
        #     fluo/time_1.tif, time_2.tif, ...
        ph_folder = os.path.join("TimelapseParserTemp", field, "ph")
        fluo_folder = os.path.join("TimelapseParserTemp", field, "fluo")

        def extract_time_index(filename: str) -> int:
            """
            'time_数字.tif' の数字部分を抽出して int に変換する。
            マッチしなければ 0 を返す。
            """
            match = re.search(r"time_(\d+)\.tif", filename)
            return int(match.group(1)) if match else 0

        ph_files = [f for f in os.listdir(ph_folder) if f.endswith(".tif")]
        fluo_files = [f for f in os.listdir(fluo_folder) if f.endswith(".tif")]
        ph_files = sorted(ph_files, key=extract_time_index)
        fluo_files = sorted(fluo_files, key=extract_time_index)

        if not ph_files or not fluo_files:
            print("No TIFF files found for ph or fluo.")
            return

        output_size = (crop_size, crop_size)

        # 前の time から引き継ぐアクティブなセルたち (cell_idx -> (cx, cy))
        active_cells = {}
        # 次に割り当てるセル番号（globalなIDとして利用する想定）
        next_cell_idx = 1

        # タイムポイントごとにループ
        for i, (ph_file, fluo_file) in enumerate(zip(ph_files, fluo_files)):
            ph_path = os.path.join(ph_folder, ph_file)
            fluo_path = os.path.join(fluo_folder, fluo_file)

            ph_img = cv2.imread(ph_path, cv2.IMREAD_COLOR)
            fluo_img = cv2.imread(fluo_path, cv2.IMREAD_COLOR)

            if ph_img is None or fluo_img is None:
                print(f"Skipping because image not found: {ph_path}, {fluo_path}")
                continue

            # (1) 位相差をグレースケールに → 閾値処理
            ph_gray = cv2.cvtColor(ph_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(ph_gray, param1, 255, cv2.THRESH_BINARY)

            # (2) Canny で輪郭検出
            edges = cv2.Canny(thresh, 0, 130)
            contours, hierarchy = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # (3) 面積フィルタリング
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            # 新しい time (i) におけるセル中心座標一覧をためる
            new_active_cells = {}  # 更新後の active_cells を入れるための辞書

            async with async_session() as session:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # (例) 中心が [400, 2000] の範囲にない場合はスキップ
                    if not (400 < cx < 2000 and 400 < cy < 2000):
                        continue

                    # (4) 前の time のセルと近いかどうか判定 → 同一セルならセル番号を再利用
                    assigned_cell_idx = None
                    min_dist = float("inf")
                    distance_threshold = (
                        40  # この閾値より小さければ「同じセル」とみなす
                    )

                    for prev_idx, (px, py) in active_cells.items():
                        dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                        if dist < distance_threshold and dist < min_dist:
                            min_dist = dist
                            assigned_cell_idx = prev_idx

                    # 前の time のセルと対応付かなければ新規セル扱い
                    if assigned_cell_idx is None:
                        assigned_cell_idx = next_cell_idx
                        next_cell_idx += 1

                    # new_active_cells に登録しておく（次の time の比較用）
                    # 同じ assigned_cell_idx が複数 contour に割り当てられないように、
                    # もしすでに new_active_cells に同じ ID が入っていたら、
                    # より中心が近いほうを優先する等のロジックを加味することもあるが、
                    # ここでは単純化して上書きなしの想定とする
                    if assigned_cell_idx not in new_active_cells:
                        new_active_cells[assigned_cell_idx] = (cx, cy)
                    else:
                        # すでに割り当て済みの場合、どちらかに統合するロジックを入れる等は運用次第
                        pass

                    # クロップ領域設定
                    x1 = max(0, cx - output_size[0] // 2)
                    y1 = max(0, cy - output_size[1] // 2)
                    x2 = min(ph_img.shape[1], cx + output_size[0] // 2)
                    y2 = min(ph_img.shape[0], cy + output_size[1] // 2)

                    cropped_ph = ph_img[y1:y2, x1:x2]  # BGR
                    cropped_fluo = fluo_img[y1:y2, x1:x2]  # BGR

                    # クロップサイズが合わなければスキップ
                    if (
                        cropped_ph.shape[0] != output_size[1]
                        or cropped_ph.shape[1] != output_size[0]
                    ):
                        continue

                    cropped_ph_gray = cv2.cvtColor(cropped_ph, cv2.COLOR_BGR2GRAY)
                    cropped_fluo_gray = cv2.cvtColor(cropped_fluo, cv2.COLOR_BGR2GRAY)

                    ph_gray_encode = cv2.imencode(".png", cropped_ph_gray)[1].tobytes()
                    fluo_gray_encode = cv2.imencode(".png", cropped_fluo_gray)[
                        1
                    ].tobytes()

                    # DB保存用の一意なID (例: field名 + グローバルcell_idx)
                    # time も入れたい場合は適宜ご調整ください
                    cell_id = f"{field}_cell{assigned_cell_idx}"

                    cell_obj = Cell(
                        cell_id=cell_id,
                        label_experiment=field,
                        manual_label=-1,
                        perimeter=perimeter,
                        area=area,
                        img_ph=ph_gray_encode,
                        img_fluo1=fluo_gray_encode,
                        img_fluo2=None,
                        contour=pickle.dumps(contour),
                        center_x=cx,
                        center_y=cy,
                        field=field,
                        time=i + 1,
                        cell=assigned_cell_idx,  # 同一セル tracking の肝
                    )

                    # 既に同一 cell_id があるかチェック（不要なら省略しても可）
                    existing = await session.execute(
                        select(Cell).filter_by(
                            cell_id=cell_id,
                            time=i + 1,
                        )
                    )
                    if existing.scalar() is None:
                        session.add(cell_obj)

                await session.commit()

            # タイム i の処理が終わったら、active_cells を更新
            active_cells = new_active_cells

        print("Cell extraction finished (with cropping).")
        return

    async def create_gif_for_cell(
        self,
        field: str,
        cell_number: int,
        dbname: str,
        channel: str = "ph",  # "ph", "fluo1", "fluo2"
        duration_ms: int = 200,
    ):
        if channel not in ["ph", "fluo1", "fluo2"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid channel. Use 'ph', 'fluo1', or 'fluo2'.",
            )

        engine = create_async_engine(
            f"sqlite+aiosqlite:///{dbname}?timeout=30", echo=False
        )
        async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        frames = []
        async with async_session() as session:
            # 例: "field" カラムと "cell" カラムを持つテーブルがある想定
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

                np_img = cv2.imdecode(
                    np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if np_img is None:
                    continue

                pil_img = Image.fromarray(np_img)
                frames.append(pil_img)

        if not frames:
            raise HTTPException(
                status_code=404,
                detail=f"No valid frames found for field={field}, cell={cell_number}, channel={channel}",
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
        channel: str = "ph",  # "ph", "fluo1", or "fluo2"
        duration_ms: int = 200,
    ):
        """
        指定した field 内に含まれる全セルについて、タイムごとに 200x200 のクロップ画像を並べ、
        できるだけ正方形に近いグリッド (n x m) でレイアウトしたフレームを時系列順に繋いで
        1つの GIF にして返す関数。

        - field: 例 "Field_1" など
        - dbname: 使用するDBファイル名 (例: "example.db")
        - channel: "ph", "fluo1", "fluo2" のいずれか
        - duration_ms: GIF のフレーム間隔 (ミリ秒)

        戻り値:
            GIF バイナリが格納された BytesIO オブジェクト
        """
        # channel 入力チェック
        if channel not in ["ph", "fluo1", "fluo2"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid channel. Use 'ph', 'fluo1', or 'fluo2'.",
            )

        # データベースへの接続準備
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{dbname}?timeout=30", echo=False
        )
        async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        async with async_session() as session:
            # 指定フィールドに該当するセルをすべて取得 (time順, cell順)
            all_cells_query = (
                select(Cell)
                .filter_by(field=field)
                .order_by(Cell.time.asc(), Cell.cell.asc())
            )
            result = await session.execute(all_cells_query)
            all_cells = result.scalars().all()

        if not all_cells:
            raise HTTPException(
                status_code=404,
                detail=f"No cells found for field={field}",
            )

        # タイムとセル番号をすべて取得
        unique_times = sorted(list(set(c.time for c in all_cells)))
        unique_cells = sorted(
            list(set(c.cell for c in all_cells if c.cell is not None))
        )

        if not unique_cells:
            raise HTTPException(
                status_code=404,
                detail=f"No valid cell numbering found for field={field}",
            )

        # 画像配置用の行数・列数を「できるだけ正方形に近い」形に計算
        # 例: セル数=200 → sqrt(200)=14.14 → 行=14, 列=15 など
        import math

        num_cells = len(unique_cells)
        rows = int(math.floor(math.sqrt(num_cells)))
        cols = int(math.ceil(num_cells / rows))

        # DBに保存された画像はすべて200x200である想定だが、
        # 念のため 1枚取り出してサイズを確認し、サイズが揃っていない場合は
        # リサイズするよう実装 (ここでは全画像200x200である前提でほぼ動作)
        first_valid_img_size = None

        # GIF用フレームを作る
        frames_for_gif = []

        for t in unique_times:
            # タイム t における全セル画像を channel から取り出して並べる
            # cell -> 画像(PIL) の辞書を作り、存在しなければ真っ黒画像にする
            cell_to_image = {}
            for cell_num in unique_cells:
                # セル(cell_num) かつ time(t) を探す
                matched = [c for c in all_cells if c.cell == cell_num and c.time == t]
                if not matched:
                    # 該当がなければ空画像を用意 (200x200 の真っ黒)
                    cell_to_image[cell_num] = Image.new("L", (200, 200), color=0)
                    continue

                # 先頭1つを取り出し (同じセル&タイムで複数行あるケースは想定外として)
                row = matched[0]
                if channel == "ph":
                    img_binary = row.img_ph
                elif channel == "fluo1":
                    img_binary = row.img_fluo1
                else:
                    img_binary = row.img_fluo2

                if img_binary is None:
                    # 画像が無い場合は真っ黒
                    cell_to_image[cell_num] = Image.new("L", (200, 200), color=0)
                    continue

                # バイナリ→numpy→PIL
                np_img = cv2.imdecode(
                    np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if np_img is None:
                    cell_to_image[cell_num] = Image.new("L", (200, 200), color=0)
                    continue

                pil_img = Image.fromarray(np_img)

                # サイズチェック＆リサイズ(必要なら)
                if first_valid_img_size is None:
                    first_valid_img_size = pil_img.size
                if pil_img.size != first_valid_img_size:
                    pil_img = pil_img.resize(first_valid_img_size)

                cell_to_image[cell_num] = pil_img

            # rows x cols の配置でモザイクを作成
            if not first_valid_img_size:
                # 一度も有効画像が出なかった場合
                raise HTTPException(
                    status_code=404,
                    detail="No valid images found to determine size.",
                )

            w, h = first_valid_img_size
            mosaic_w = cols * w
            mosaic_h = rows * h

            mosaic = Image.new("L", (mosaic_w, mosaic_h), color=0)

            # セルを行・列順に敷き詰める
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < num_cells:
                        cell_num = unique_cells[idx]
                        tile_img = cell_to_image[cell_num]
                        mosaic.paste(tile_img, (c * w, r * h))
                        idx += 1
                    else:
                        # セル総数が grid に満たない場合はそのまま空でOK
                        pass

            frames_for_gif.append(mosaic)

        # Frames が空ならエラー
        if not frames_for_gif:
            raise HTTPException(
                status_code=404,
                detail=f"No frames created for field={field}",
            )

        # GIF をメモリバッファに保存
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
