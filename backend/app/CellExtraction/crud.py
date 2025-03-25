import asyncio
import os
import pickle
from typing import Literal

import aiofiles
import cv2
import nd2reader
import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import BLOB, Column, FLOAT, Integer, String, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import select
import ulid


def get_ulid() -> str:
    return str(ulid.new())


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
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)
    user_id = Column(String, nullable=True)


async def get_session(dbname: str):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{dbname}?timeout=30",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session


async def create_database(dbname: str):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{dbname}?timeout=30",
        echo=True,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(
            lambda sync_conn: sync_conn.execute(text("PRAGMA journal_mode=WAL;"))
        )
    return engine


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
    def save_images(images, file_name, num_channels, ulid):
        """
        画像を保存し、MultipageTIFFとして出力する。
        """
        all_images = []
        for i, img in enumerate(images):
            if num_channels > 1:
                for j in range(num_channels):
                    all_images.append(
                        Image.open(f"nd2totiff{ulid}/image_{i}_channel_{j}.tif")
                    )
            else:
                all_images.append(Image.open(f"nd2totiff{ulid}/image_{i}.tif"))

        all_images[0].save(
            f"{file_name.split('/')[-1].split('.')[0]}.tif",
            save_all=True,
            append_images=all_images[1:],
        )

    @staticmethod
    def get_contour_center(contour):
        M = cv2.moments(contour)
        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]
        return center_x, center_y

    @staticmethod
    def extract_nd2(file_name: str, mode: str, ulid: str, reverse: bool = False) -> int:
        """
        nd2ファイルをMultipageTIFFに変換する。
        """
        temp_dir = f"nd2totiff{ulid}"
        os.makedirs(temp_dir, exist_ok=True)

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
                        if reverse:
                            channel = num_channels - channel - 1
                        image.save(f"{temp_dir}/image_{n}_channel_{channel}.tif")
                else:
                    array = np.array(img)
                    array = SyncChores.process_image(array)
                    image = Image.fromarray(array)
                    image.save(f"{temp_dir}/image_{n}.tif")
            SyncChores.save_images(images, file_name, num_channels, ulid=ulid)
        SyncChores.cleanup(temp_dir)
        num_tiff = SyncChores.extract_tiff(
            tiff_path=f"./{file_name.split('/')[-1].split('.')[0]}.tif",
            ulid=ulid,
            mode=mode,
            reverse=reverse,
        )
        os.remove(f"./{file_name.split('/')[-1].split('.')[0]}.tif")
        return num_tiff

    @staticmethod
    def extract_tiff(
        tiff_path: str,
        ulid: str,
        mode: Literal["single_layer", "dual_layer", "triple_layer"] = "dual_layer",
        reverse: bool = False,
    ) -> int:
        temp_dir = f"TempData{ulid}"
        os.makedirs(temp_dir, exist_ok=True)
        folders = [
            folder
            for folder in os.listdir(temp_dir)
            if os.path.isdir(os.path.join(temp_dir, folder))
        ]

        layers = {
            "triple_layer": ["Fluo1", "Fluo2", "PH"],
            "single_layer": ["PH"],
            "dual_layer": ["Fluo1", "PH"],
        }

        for layer in layers.get(mode, []):
            os.makedirs(f"{temp_dir}/{layer}", exist_ok=True)

        with Image.open(tiff_path) as tiff:
            num_pages = tiff.n_frames
            img_num = 0

            layer_map = {
                "triple_layer": [(0, "PH"), (1, "Fluo1"), (2, "Fluo2")],
                "single_layer": [(0, "PH")],
                "dual_layer": (
                    [(0, "PH"), (1, "Fluo1")]
                    if not reverse
                    else [(1, "PH"), (0, "Fluo1")]
                ),
            }

            for i in range(num_pages):
                tiff.seek(i)
                layer_idx = i % len(layer_map[mode])
                layer = layer_map[mode][layer_idx][1]
                filename = f"{temp_dir}/{layer}/{img_num}.tif"
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
        ulid: str,
        param1: int = 130,
        image_size: int = 200,
        mode: Literal["single_layer", "dual_layer", "triple_layer"] = "dual_layer",
    ) -> int:
        temp_dir = f"TempData{ulid}"
        print(f"Initializing {temp_dir}")
        print("}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
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
            os.mkdir(f"{temp_dir}")
        except:
            pass

        init_folders = [f"{temp_dir}/{d}" for d in init_folders]
        folders = [
            folder
            for folder in os.listdir(f"{temp_dir}")
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
                os.mkdir(f"{temp_dir}/frames/tiff_{i}")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"{temp_dir}/frames/tiff_{i}/Cells")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"{temp_dir}/frames/tiff_{i}/Cells/ph")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"{temp_dir}/frames/tiff_{i}/Cells/fluo1")
            except Exception as e:
                print(e)
            try:
                os.mkdir(f"ph_contours{ulid}")
            except Exception as e:
                print(e)

            if mode == "triple_layer":
                try:
                    os.mkdir(f"{temp_dir}/frames/tiff_{i}/Cells/fluo2")
                except Exception as e:
                    print(e)
                try:
                    os.mkdir(f"{temp_dir}/frames/tiff_{i}/Cells/fluo2_adjusted")
                except Exception as e:
                    print(e)
                try:
                    os.mkdir(f"{temp_dir}/frames/tiff_{i}/Cells/fluo2_contour")
                except Exception as e:
                    print(e)
        loop_num = num_tiff // set_num if mode != "single_layer" else num_tiff
        for k in range(loop_num):
            image_ph = cv2.imread(f"{temp_dir}/PH/{k}.tif")
            print(num_tiff)
            if mode == "dual_layer" or mode == "triple_layer":
                image_fluo_1 = cv2.imread(f"{temp_dir}/Fluo1/{k}.tif")
            if mode == "triple_layer":
                image_fluo_2 = cv2.imread(f"{temp_dir}/Fluo2/{k}.tif")
            img_gray = cv2.cvtColor(image_ph, cv2.COLOR_BGR2GRAY)

            # ２値化を行う
            ret, thresh = cv2.threshold(img_gray, param1, 255, cv2.THRESH_BINARY)
            img_canny = cv2.Canny(thresh, 0, 130)
            contours, hierarchy = cv2.findContours(
                img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
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
            cv2.imwrite(f"ph_contours{ulid}/{k}.png", image_ph_copy)
            n = 0
            if mode == "triple_layer":
                for ph, fluo1, fluo2 in zip(
                    cropped_images_ph, cropped_images_fluo_1, cropped_images_fluo_2
                ):
                    if (
                        len(ph) == output_size[0]
                        and len(ph[0]) == output_size[1]
                        and len(fluo1) == output_size[0]
                        and len(fluo1[0]) == output_size[1]
                        and len(fluo2) == output_size[0]
                        and len(fluo2[0]) == output_size[1]
                    ):
                        cv2.imwrite(f"{temp_dir}/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                        cv2.imwrite(
                            f"{temp_dir}/frames/tiff_{k}/Cells/fluo1/{n}.png", fluo1
                        )
                        cv2.imwrite(
                            f"{temp_dir}/frames/tiff_{k}/Cells/fluo2/{n}.png", fluo2
                        )
                        n += 1

            elif mode == "single_layer":
                for ph in zip(cropped_images_ph):
                    if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                        cv2.imwrite(f"{temp_dir}/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                        n += 1
            elif mode == "dual_layer":
                for ph, fluo1 in zip(cropped_images_ph, cropped_images_fluo_1):
                    if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                        cv2.imwrite(f"{temp_dir}/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                        cv2.imwrite(
                            f"{temp_dir}/frames/tiff_{k}/Cells/fluo1/{n}.png", fluo1
                        )
                        n += 1
        return num_tiff


class ExtractionCrudBase:
    def __init__(
        self,
        nd2_path: str,
        mode: str = "dual_layer",
        param1: int = 130,
        image_size: int = 200,
        reverse_layers: bool = False,
        user_id: str | None = None,
    ):
        self.nd2_path = nd2_path
        self.nd2_path = self.nd2_path.replace("\\", "/")
        self.file_prefix = self.nd2_path.split("/")[-1].split(".")[0]
        self.mode = mode
        self.param1 = param1
        self.image_size = image_size
        self.reverse_layers = reverse_layers
        self.ulid = get_ulid()
        self.user_id = user_id

    async def load_image(self, path):
        async with aiofiles.open(path, mode="rb") as f:
            data = await f.read()
        img_array = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    def process_image(self, img_ph, img_fluo1=None, img_fluo2=None):
        img_ph_gray = cv2.cvtColor(img_ph, cv2.COLOR_BGR2GRAY)
        img_fluo1_gray = img_fluo2_gray = None
        if img_fluo1 is not None:
            img_fluo1_gray = cv2.cvtColor(img_fluo1, cv2.COLOR_BGR2GRAY)
        if img_fluo2 is not None:
            img_fluo2_gray = cv2.cvtColor(img_fluo2, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img_ph_gray, self.param1, 255, cv2.THRESH_BINARY)
        img_canny = cv2.Canny(thresh, 0, 150)
        contours_raw, hierarchy = cv2.findContours(
            img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = list(filter(lambda x: cv2.contourArea(x) >= 300, contours_raw))

        contours = list(
            i
            for i in contours
            if SyncChores.get_contour_center(i)[0] - img_ph.shape[1] // 2 < 3
            and SyncChores.get_contour_center(i)[1] - img_ph.shape[0] // 2 < 3
        )
        contour = contours[0] if contours else None
        return contour, img_ph_gray, img_fluo1_gray, img_fluo2_gray

    async def process_cell(self, dbname, i, j, user_id: str | None = None):
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{dbname}?timeout=30", echo=False
        )
        async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
        async with async_session() as session:
            cell_id = f"F{i}C{j}"
            img_ph = await self.load_image(
                f"TempData{self.ulid}/frames/tiff_{i}/Cells/ph/{j}.png"
            )
            img_fluo1 = img_fluo2 = None
            if self.mode != "single_layer":
                img_fluo1 = await self.load_image(
                    f"TempData{self.ulid}/frames/tiff_{i}/Cells/fluo1/{j}.png"
                )

            contour, img_ph_gray, img_fluo1_gray, img_fluo2_gray = self.process_image(
                img_ph, img_fluo1
            )

            if contour is not None:
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                center_x, center_y = SyncChores.get_contour_center(contour)
                if (
                    abs(center_x - img_ph.shape[1] // 2) < 3
                    and abs(center_y - img_ph.shape[0] // 2) < 3
                ):
                    img_ph_data = cv2.imencode(".png", img_ph_gray)[1].tobytes()
                    img_fluo1_data = img_fluo2_data = None
                    if self.mode != "single_layer":
                        img_fluo1_data = cv2.imencode(".png", img_fluo1_gray)[
                            1
                        ].tobytes()
                    if self.mode == "triple_layer":
                        img_fluo2 = await self.load_image(
                            f"TempData{self.ulid}/frames/tiff_{i}/Cells/fluo2/{j}.png"
                        )
                        img_fluo2_gray = cv2.cvtColor(img_fluo2, cv2.COLOR_BGR2GRAY)
                        img_fluo2_data = cv2.imencode(".png", img_fluo2_gray)[
                            1
                        ].tobytes()
                    cv2.drawContours(img_ph, [contour], -1, (0, 255, 0), 1, cv2.LINE_AA)
                    cell = Cell(
                        cell_id=cell_id,
                        label_experiment="",
                        manual_label="N/A",
                        perimeter=perimeter,
                        area=area,
                        img_ph=img_ph_data,
                        img_fluo1=img_fluo1_data,
                        img_fluo2=img_fluo2_data,
                        contour=pickle.dumps(contour),
                        center_x=center_x,
                        center_y=center_y,
                        user_id=user_id,
                    )
                    existing_cell = await session.execute(
                        select(Cell).filter_by(cell_id=cell_id)
                    )
                    existing_cell = existing_cell.scalar()
                    if existing_cell is None:
                        session.add(cell)
                        await session.commit()

    async def main(self):
        chores = SyncChores()
        dbname = (
            f"databases/{self.file_prefix}-uploaded.db"
            if self.mode != "single_layer"
            else f"databases/{self.file_prefix}-single_layer-uploaded.db"
        )

        try:
            await asyncio.to_thread(os.remove, dbname)
        except FileNotFoundError:
            pass

        num_tiff = await asyncio.to_thread(
            chores.extract_nd2, self.nd2_path, self.mode, self.ulid, self.reverse_layers
        )

        await asyncio.to_thread(
            chores.init,
            f"{self.file_prefix}.nd2",
            num_tiff,
            self.ulid,
            self.param1,
            self.image_size,
            self.mode,
        )

        iter_n = {
            "triple_layer": num_tiff // 3,
            "single_layer": num_tiff,
            "dual_layer": num_tiff // 2,
        }

        await create_database(dbname)
        engine = create_async_engine(f"sqlite+aiosqlite:///{dbname}?timeout=30")

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        tasks = []
        for i in range(iter_n[self.mode]):
            cell_path = f"TempData{self.ulid}/frames/tiff_{i}/Cells/ph/"
            cell_files = await asyncio.to_thread(os.listdir, cell_path)
            for j in range(len(cell_files)):
                tasks.append(self.process_cell(dbname, i, j, self.user_id))

        await asyncio.gather(*tasks)
        await asyncio.to_thread(SyncChores.cleanup, f"TempData{self.ulid}")
        return num_tiff, self.ulid

    async def get_nd2_filenames(self) -> list[str]:
        return [
            i
            for i in os.listdir("uploaded_files")
            if i.endswith(".nd2") and not i.endswith("timelapse.nd2")
        ]

    async def delete_nd2_file(self, filename: str) -> bool:
        filename = filename.split("/")[-1]
        await asyncio.to_thread(os.remove, f"uploaded_files/{filename}")
        return True

    async def get_ph_contours(self, frame_num: int, ulid: str) -> StreamingResponse:
        filepath = f"ph_contours{ulid}/{frame_num}.png"
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found")
        return StreamingResponse(open(filepath, "rb"), media_type="image/png")

    async def get_ph_contours_num(self, ulid: str) -> int:
        return len(os.listdir(f"ph_contours{ulid}"))
