import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
import io
import math
import os
import pickle
import re
import shutil
from functools import partial
from PIL import Image, ImageDraw, ImageFont
from typing import Literal, Any
import cv2
import nd2reader
import numpy as np
from PIL import Image
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import BLOB, Column, FLOAT, Integer, String, delete, update, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import select
import random
from scipy.optimize import linear_sum_assignment
from CellDBConsole.crud import AsyncChores as CellDBAsyncChores

def get_ulid() -> str:
    """Return a fake ULID using random digits."""
    # NOTE: This is a placeholder implementation
    return "".join(str(random.randint(0, 9)) for _ in range(16))


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
    @staticmethod
    def correct_drift(reference_image, target_image):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(reference_image, None)
        kp2, des2 = orb.detectAndCompute(target_image, None)

        if des1 is None or des2 is None:
            print("Descriptor is None, skipping drift correction.")
            return target_image

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 300:
            print("Insufficient matches, skipping drift correction.")
            return target_image

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        height, width = reference_image.shape[:2]
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
    def correct_drift_ecc(reference_image, target_image, warp_mode=cv2.MOTION_AFFINE):
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

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        number_of_iterations = 100
        termination_eps = 1e-4
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            number_of_iterations,
            termination_eps,
        )

        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_gray, tgt_gray, warp_matrix, warp_mode, criteria
            )
        except cv2.error as e:
            print(f"[ECC] Alignment failed with OpenCV error: {e}")
            return target_image

        h, w = reference_image.shape[:2]
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(
                target_image,
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )
        else:
            aligned = cv2.warpAffine(
                target_image,
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )

        return aligned

    @staticmethod
    def calc_shift_by_phase_correlation(ref_img: np.ndarray, target_img: np.ndarray) -> tuple[int, int]:
        ref_float = ref_img.astype(np.float32)
        tgt_float = target_img.astype(np.float32)
        ref_float -= np.mean(ref_float)
        tgt_float -= np.mean(tgt_float)

        (shift_x, shift_y), _ = cv2.phaseCorrelate(ref_float, tgt_float)
        vertical_shift = -int(round(shift_y))
        horizontal_shift = -int(round(shift_x))
        return (vertical_shift, horizontal_shift)

    @staticmethod
    def correct_drift_phase_correlation(reference_image, target_image):
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

        vertical_shift, horizontal_shift = SyncChores.calc_shift_by_phase_correlation(
            ref_gray, tgt_gray
        )

        h, w = reference_image.shape[:2]
        M = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
        aligned = cv2.warpAffine(target_image, M, (w, h))

        return aligned

    @staticmethod
    def estimate_transform_orb(reference_image, target_image):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(reference_image, None)
        kp2, des2 = orb.detectAndCompute(target_image, None)

        if des1 is None or des2 is None:
            print("Descriptor is None, skipping drift calculation.")
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 300:
            print("Insufficient matches, skipping drift calculation.")
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        height, width = reference_image.shape[:2]
        src_pts[:, :, 0] = width - src_pts[:, :, 0]
        src_pts[:, :, 1] = height - src_pts[:, :, 1]
        dst_pts[:, :, 0] = width - dst_pts[:, :, 0]
        dst_pts[:, :, 1] = height - dst_pts[:, :, 1]

        matrix, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )
        return matrix

    @staticmethod
    def estimate_transform_ecc(reference_image, target_image, warp_mode=cv2.MOTION_AFFINE):
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

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        number_of_iterations = 100
        termination_eps = 1e-4
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            number_of_iterations,
            termination_eps,
        )

        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, tgt_gray, warp_matrix, warp_mode, criteria
            )
        except cv2.error as e:
            print(f"[ECC] Alignment failed with OpenCV error: {e}")
            return None

        return warp_matrix

    @staticmethod
    def apply_transform(image, transform, method):
        if transform is None:
            return image
        h, w = image.shape[:2]
        if method == "phase":
            vertical_shift, horizontal_shift = transform
            M = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
            return cv2.warpAffine(image, M, (w, h))
        elif method == "orb":
            return cv2.warpAffine(image, transform, (w, h))
        elif method == "ecc":
            if transform.shape == (2, 3):
                return cv2.warpAffine(
                    image,
                    transform,
                    (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                )
            else:
                return cv2.warpPerspective(
                    image,
                    transform,
                    (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                )
        else:
            return image

    @staticmethod
    def process_image(array):
        array = array.astype(np.float32)
        array_min = array.min()
        array_max = array.max()
        array -= array_min
        diff = array_max - array_min
        if diff == 0:
            return (array * 0).astype(np.uint8)
        array /= diff
        array *= 255
        return array.astype(np.uint8)

    @classmethod
    def extract_timelapse_nd2(cls, file_name: str, drift_method: str = "phase"):
        base_output_dir = "uploaded_files/"
        if os.path.exists("TimelapseParserTemp"):
            shutil.rmtree("TimelapseParserTemp")
        os.makedirs("TimelapseParserTemp", exist_ok=True)

        nd2_fullpath = os.path.join(base_output_dir, file_name)
        with nd2reader.ND2Reader(nd2_fullpath) as images:
            images.iter_axes = []
            images.bundle_axes = "yxc"

            num_fields = images.sizes.get("v", 1)
            num_channels = images.sizes.get("c", 1)
            num_timepoints = images.sizes.get("t", 1)

            for field_idx in range(num_fields):
                field_folder = os.path.join(
                    "TimelapseParserTemp", f"Field_{field_idx + 1}"
                )
                os.makedirs(field_folder, exist_ok=True)

                base_output_subdir_ph = os.path.join(field_folder, "ph")
                base_output_subdir_fluo1 = os.path.join(field_folder, "fluo1")
                base_output_subdir_fluo2 = os.path.join(field_folder, "fluo2")
                os.makedirs(base_output_subdir_ph, exist_ok=True)
                os.makedirs(base_output_subdir_fluo1, exist_ok=True)
                os.makedirs(base_output_subdir_fluo2, exist_ok=True)

                reference_ph = None
                drift_transforms: dict[int, Any] = {}

                # first process phase contrast (channel 0) to calculate drifts
                for time_idx in range(num_timepoints):
                    frame_data = images.get_frame_2D(v=field_idx, c=0, t=time_idx)
                    channel_image = cls.process_image(frame_data)
                    tiff_filename = os.path.join(
                        base_output_subdir_ph, f"time_{time_idx + 1}.tif"
                    )

                    if time_idx == 0:
                        reference_ph = channel_image
                        drift_transforms[time_idx] = None
                        Image.fromarray(channel_image).save(tiff_filename)
                    else:
                        if drift_method == "ecc":
                            transform = cls.estimate_transform_ecc(reference_ph, channel_image)
                        elif drift_method == "orb":
                            transform = cls.estimate_transform_orb(reference_ph, channel_image)
                        else:
                            transform = cls.calc_shift_by_phase_correlation(
                                reference_ph if len(reference_ph.shape)==2 else cv2.cvtColor(reference_ph, cv2.COLOR_BGR2GRAY),
                                channel_image if len(channel_image.shape)==2 else cv2.cvtColor(channel_image, cv2.COLOR_BGR2GRAY),
                            )

                        drift_transforms[time_idx] = transform
                        corrected = cls.apply_transform(channel_image, transform, drift_method)
                        Image.fromarray(corrected).save(tiff_filename)

                # process remaining channels using the same drift transforms
                for channel_idx in range(1, num_channels):
                    if channel_idx == 1:
                        channel_label = "fluo1"
                        save_dir = base_output_subdir_fluo1
                    else:
                        channel_label = "fluo2"
                        save_dir = base_output_subdir_fluo2

                    for time_idx in range(num_timepoints):
                        frame_data = images.get_frame_2D(
                            v=field_idx, c=channel_idx, t=time_idx
                        )
                        channel_image = cls.process_image(frame_data)
                        tiff_filename = os.path.join(
                            save_dir, f"time_{time_idx + 1}.tif"
                        )

                        if time_idx == 0:
                            Image.fromarray(channel_image).save(tiff_filename)
                        else:
                            transform = drift_transforms.get(time_idx)
                            corrected = cls.apply_transform(channel_image, transform, drift_method)
                            Image.fromarray(corrected).save(tiff_filename)

    @classmethod
    def create_combined_gif(
        cls, field_folder: str, resize_factor: float = 0.5
    ) -> io.BytesIO:
        ph_folder = os.path.join(field_folder, "ph")
        fluo1_folder = os.path.join(field_folder, "fluo1")
        fluo2_folder = os.path.join(field_folder, "fluo2")

        def extract_time_index(path: str) -> int:
            match = re.search(r"time_(\d+)\.tif", path)
            return int(match.group(1)) if match else 0

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

        ph_images = [Image.open(img_file) for img_file in ph_image_files]
        fluo1_images = [Image.open(img_file) for img_file in fluo1_image_files]
        fluo2_images = [Image.open(img_file) for img_file in fluo2_image_files]

        if not ph_images or not fluo1_images or not fluo2_images:
            raise ValueError("Not all channel images found to create combined GIF.")

        combined_images = []
        for ph_img, fluo1_img, fluo2_img in zip(ph_images, fluo1_images, fluo2_images):
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

            combined_width = ph_img.width + fluo1_img.width + fluo2_img.width
            combined_height = max(ph_img.height, fluo1_img.height, fluo2_img.height)
            combined_img = Image.new("RGB", (combined_width, combined_height))

            offset_x = 0
            combined_img.paste(ph_img, (offset_x, 0))
            offset_x += ph_img.width
            combined_img.paste(fluo1_img, (offset_x, 0))
            offset_x += fluo1_img.width
            combined_img.paste(fluo2_img, (offset_x, 0))

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
            optimize=True,
        )
        gif_buffer.seek(0)
        return gif_buffer


class AsyncChores:
    def __init__(self):
        # use process pool for CPU intensive tasks
        self.executor = ProcessPoolExecutor(max_workers=os.cpu_count())

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

    async def correct_drift_phase_correlation(self, reference_image, target_image):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            partial(
                SyncChores.correct_drift_phase_correlation, reference_image, target_image
            ),
        )

    async def process_image(self, array):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, partial(SyncChores.process_image, array)
        )

    async def extract_timelapse_nd2(self, file_name: str, drift_method: str = "phase"):
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
    def __init__(self, nd2_path: str):
        self.nd2_path = nd2_path

    async def get_nd2_filenames(self) -> list[str]:
        return [i for i in os.listdir("uploaded_files") if i.endswith("_timelapse.nd2")]

    async def delete_nd2_file(self, file_path: str):
        filename = file_path.split("/")[-1]
        await asyncio.to_thread(os.remove, f"uploaded_files/{filename}")
        return True

    async def main(self, drift_method: str = "phase"):
        await AsyncChores().extract_timelapse_nd2(self.nd2_path, drift_method=drift_method)
        return JSONResponse(
            content={"message": f"Timelapse extracted with {drift_method}."}
        )

    async def create_combined_gif(self, field: str):
        return await AsyncChores().create_combined_gif("TimelapseParserTemp/" + field)

    async def get_fields_of_nd2(self) -> list[str]:
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
        distance_threshold: float = 80.0,  # ← 距離閾値を引数化 (デフォルト80)
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

        ph_folder = os.path.join("TimelapseParserTemp", field, "ph")
        fluo1_folder = os.path.join("TimelapseParserTemp", field, "fluo1")
        fluo2_folder = os.path.join("TimelapseParserTemp", field, "fluo2")

        def extract_time_index(filename: str) -> int:
            match = re.search(r"time_(\d+)\.tif", filename)
            return int(match.group(1)) if match else 0

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

        # cell_idx -> {'cx': x, 'cy': y, 'contour_shifted': contour, 'area': area, 'perimeter': perimeter}
        active_cells = {}
        next_cell_idx = 1
        base_ids = {}

        for i, ph_file in enumerate(ph_files):
            ph_time_idx = extract_time_index(ph_file)
            ph_path = os.path.join(ph_folder, ph_file)

            candidates_fluo1 = [
                f for f in fluo1_files if extract_time_index(f) == ph_time_idx
            ]
            fluo1_path = os.path.join(fluo1_folder, candidates_fluo1[0]) if candidates_fluo1 else None

            candidates_fluo2 = [
                f for f in fluo2_files if extract_time_index(f) == ph_time_idx
            ]
            fluo2_path = os.path.join(fluo2_folder, candidates_fluo2[0]) if candidates_fluo2 else None

            ph_img = cv2.imread(ph_path, cv2.IMREAD_COLOR)
            fluo1_img = cv2.imread(fluo1_path, cv2.IMREAD_COLOR) if fluo1_path else None
            fluo2_img = cv2.imread(fluo2_path, cv2.IMREAD_COLOR) if fluo2_path else None

            if ph_img is None:
                print(f"Skipping because ph image not found: {ph_path}")
                continue

            height, width = ph_img.shape[:2]

            ph_gray = cv2.cvtColor(ph_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(ph_gray, param1, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(thresh, 0, 130)
            contours, hierarchy = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            new_active_cells = {}

            async with async_session() as session:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    x_min = int(width * 0.1)
                    x_max = int(width * 0.9)
                    y_min = int(height * 0.1)
                    y_max = int(height * 0.9)

                    if not (x_min < cx < x_max and y_min < cy < y_max):
                        continue

                    assigned_cell_idx = None
                    min_dist = float("inf")

                    # --- ここで distance_threshold を使う ---
                    for prev_idx, info in active_cells.items():
                        px, py = info['cx'], info['cy']
                        dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                        if dist < distance_threshold and dist < min_dist:
                            min_dist = dist
                            assigned_cell_idx = prev_idx

                    if assigned_cell_idx is None:
                        assigned_cell_idx = next_cell_idx
                        next_cell_idx += 1

                    x1 = max(0, cx - output_size[0] // 2)
                    y1 = max(0, cy - output_size[1] // 2)
                    x2 = min(ph_img.shape[1], cx + output_size[0] // 2)
                    y2 = min(ph_img.shape[0], cy + output_size[1] // 2)

                    if (y2 - y1) != output_size[1] or (x2 - x1) != output_size[0]:
                        continue

                    cropped_ph = ph_img[y1:y2, x1:x2]
                    cropped_ph_gray = cv2.cvtColor(cropped_ph, cv2.COLOR_BGR2GRAY)
                    ph_gray_encode = cv2.imencode(".png", cropped_ph_gray)[1].tobytes()

                    fluo1_gray_encode = None
                    if fluo1_img is not None:
                        cropped_fluo1 = fluo1_img[y1:y2, x1:x2]
                        cropped_fluo1_gray = cv2.cvtColor(cropped_fluo1, cv2.COLOR_BGR2GRAY)
                        fluo1_gray_encode = cv2.imencode(".png", cropped_fluo1_gray)[1].tobytes()

                    fluo2_gray_encode = None
                    if fluo2_img is not None:
                        cropped_fluo2 = fluo2_img[y1:y2, x1:x2]
                        cropped_fluo2_gray = cv2.cvtColor(cropped_fluo2, cv2.COLOR_BGR2GRAY)
                        fluo2_gray_encode = cv2.imencode(".png", cropped_fluo2_gray)[1].tobytes()

                    contour_shifted = contour.copy()
                    contour_shifted[:, :, 0] -= x1
                    contour_shifted[:, :, 1] -= y1

                    new_active_cells[assigned_cell_idx] = {
                        "cx": cx,
                        "cy": cy,
                        "contour_shifted": contour_shifted,
                        "area": area,
                        "perimeter": perimeter,
                    }

                    new_ulid = get_ulid()
                    if assigned_cell_idx not in base_ids:
                        base_ids[assigned_cell_idx] = new_ulid

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

                    existing = await session.execute(
                        select(Cell).filter_by(cell_id=new_ulid, time=i + 1)
                    )
                    if existing.scalar() is None:
                        session.add(cell_obj)

                # Track cells that disappeared in this frame
                missing_cells = set(active_cells.keys()) - set(new_active_cells.keys())
                for m_idx in missing_cells:
                    info = active_cells[m_idx]
                    cx, cy = info["cx"], info["cy"]
                    contour_shifted = info["contour_shifted"]
                    area = info.get("area", 0)
                    perimeter = info.get("perimeter", 0)

                    x1 = max(0, cx - output_size[0] // 2)
                    y1 = max(0, cy - output_size[1] // 2)
                    x2 = min(ph_img.shape[1], cx + output_size[0] // 2)
                    y2 = min(ph_img.shape[0], cy + output_size[1] // 2)

                    if (y2 - y1) != output_size[1] or (x2 - x1) != output_size[0]:
                        continue

                    cropped_ph = ph_img[y1:y2, x1:x2]
                    cropped_ph_gray = cv2.cvtColor(cropped_ph, cv2.COLOR_BGR2GRAY)
                    ph_gray_encode = cv2.imencode(".png", cropped_ph_gray)[1].tobytes()

                    fluo1_gray_encode = None
                    if fluo1_img is not None:
                        cropped_fluo1 = fluo1_img[y1:y2, x1:x2]
                        cropped_fluo1_gray = cv2.cvtColor(cropped_fluo1, cv2.COLOR_BGR2GRAY)
                        fluo1_gray_encode = cv2.imencode(".png", cropped_fluo1_gray)[1].tobytes()

                    fluo2_gray_encode = None
                    if fluo2_img is not None:
                        cropped_fluo2 = fluo2_img[y1:y2, x1:x2]
                        cropped_fluo2_gray = cv2.cvtColor(cropped_fluo2, cv2.COLOR_BGR2GRAY)
                        fluo2_gray_encode = cv2.imencode(".png", cropped_fluo2_gray)[1].tobytes()

                    new_ulid = get_ulid()
                    if m_idx not in base_ids:
                        base_ids[m_idx] = new_ulid

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
                        cell=m_idx,
                        base_cell_id=base_ids[m_idx],
                        is_dead=0,
                        gif_ph=None,
                        gif_fluo1=None,
                        gif_fluo2=None,
                    )

                    existing = await session.execute(
                        select(Cell).filter_by(cell_id=new_ulid, time=i + 1)
                    )
                    if existing.scalar() is None:
                        session.add(cell_obj)

                    new_active_cells[m_idx] = info

                await session.commit()

            active_cells = new_active_cells

        # 以下は「最初のフレームに存在しないセルを削除する」処理
        async with async_session() as session:
            first_frame = 1
            subquery = select(Cell.cell).where(
                Cell.field == field,
                Cell.time == first_frame,
            )
            result = await session.execute(subquery)
            cells_with_first_frame = [row[0] for row in result]

            delete_stmt = (
                delete(Cell)
                .where(Cell.field == field)
                .where(~Cell.cell.in_(cells_with_first_frame))
            )
            await session.execute(delete_stmt)
            await session.commit()

        print("Cell extraction finished (with cropping).")
        print("Removed cells that did not appear in the first frame.")

        # 以下はGIFをDBに保存する処理（省略せず残していますが、メインの変更点ではありません）
        async with async_session() as session:
            base_cells = await session.execute(select(Cell).where(Cell.time == 1))
            base_cells = base_cells.scalars().all()
            base_cell_ids = [cell.cell_id for cell in base_cells]

            for base_id in base_cell_ids:
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
    
    async def extract_cells_with_hungarian(
        self,
        field: str,
        dbname: str,
        param1: int = 90,
        min_area: int = 300,
        crop_size: int = 200,
        max_dist_for_matching: float = 150.0,
    ):
        """
        ハンガリアン法を用いて前フレーム -> 今フレームの細胞対応付けを行う例。
        max_dist_for_matching: コスト行列で許容する最大距離（これを超える場合はマッチさせない＝大きいペナルティにする）。
        """
        db_path = f"timelapse_databases/{dbname}"
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}?timeout=30", echo=False
        )
        async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        ph_folder = os.path.join("TimelapseParserTemp", field, "ph")
        fluo1_folder = os.path.join("TimelapseParserTemp", field, "fluo1")
        fluo2_folder = os.path.join("TimelapseParserTemp", field, "fluo2")

        def extract_time_index(filename: str) -> int:
            match = re.search(r"time_(\d+)\.tif", filename)
            return int(match.group(1)) if match else 0

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

        output_size = (crop_size, crop_size)

        # 前フレーム（i-1）で登場した全セルの情報を持つリスト
        # ここでは list[dict] のように、 { 'cell_idx': int, 'cx': float, 'cy': float } を格納する想定
        previous_cells = []
        next_cell_idx = 1
        base_ids = {}

        for frame_i, ph_file in enumerate(ph_files):
            time_idx = extract_time_index(ph_file)
            ph_path = os.path.join(ph_folder, ph_file)

            candidates_fluo1 = [f for f in fluo1_files if extract_time_index(f) == time_idx]
            fluo1_path = os.path.join(fluo1_folder, candidates_fluo1[0]) if candidates_fluo1 else None
            candidates_fluo2 = [f for f in fluo2_files if extract_time_index(f) == time_idx]
            fluo2_path = os.path.join(fluo2_folder, candidates_fluo2[0]) if candidates_fluo2 else None

            ph_img = cv2.imread(ph_path, cv2.IMREAD_COLOR)
            fluo1_img = cv2.imread(fluo1_path, cv2.IMREAD_COLOR) if fluo1_path else None
            fluo2_img = cv2.imread(fluo2_path, cv2.IMREAD_COLOR) if fluo2_path else None

            if ph_img is None:
                print(f"Skipping because ph image not found: {ph_path}")
                continue

            height, width = ph_img.shape[:2]
            ph_gray = cv2.cvtColor(ph_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(ph_gray, param1, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(thresh, 0, 130)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

            # 今フレームの細胞候補
            current_cells = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                x_min = int(width * 0.1)
                x_max = int(width * 0.9)
                y_min = int(height * 0.1)
                y_max = int(height * 0.9)
                if not (x_min < cx < x_max and y_min < cy < y_max):
                    continue

                current_cells.append({
                    'contour': cnt,
                    'cx': cx,
                    'cy': cy,
                })

            # --- ハンガリアン法で previous_cells と current_cells を対応付け ---
            # コスト行列は shape=(len(previous_cells), len(current_cells))
            # コスト = 距離。一定距離を超える場合は非常に大きい値を入れて対応付けを避けるようにする
            if previous_cells and current_cells:
                cost_matrix = np.zeros((len(previous_cells), len(current_cells)), dtype=np.float32)
                for i_p, pcell in enumerate(previous_cells):
                    for i_c, ccell in enumerate(current_cells):
                        dist = math.dist((pcell['cx'], pcell['cy']), (ccell['cx'], ccell['cy']))
                        if dist > max_dist_for_matching:
                            cost_matrix[i_p, i_c] = 999999.0
                        else:
                            cost_matrix[i_p, i_c] = dist

                row_idx, col_idx = linear_sum_assignment(cost_matrix)
                # row_idx が previous, col_idx が current のインデックス対応
                matched_pairs = []
                unmatched_current = set(range(len(current_cells)))
                for r, c in zip(row_idx, col_idx):
                    if cost_matrix[r, c] < 999999.0:
                        matched_pairs.append((r, c))
                        if c in unmatched_current:
                            unmatched_current.remove(c)

                # previous_cells のうち対応が付かなかったものは消す
                # (すぐ消すか残すかは要件次第。ここでは消してしまう)
                new_previous_cells = []

                # 今フレームで見つかったセルに「previous_cell_idx」を割り当てる
                assigned_cell_map = {}

                for r, c in matched_pairs:
                    cell_idx = previous_cells[r]['cell_idx']
                    assigned_cell_map[c] = cell_idx
                    new_previous_cells.append({
                        'cell_idx': cell_idx,
                        'cx': current_cells[c]['cx'],
                        'cy': current_cells[c]['cy'],
                    })

                # unmatched (＝新たに出現した) current cell
                for uc in unmatched_current:
                    cell_idx = next_cell_idx
                    next_cell_idx += 1
                    assigned_cell_map[uc] = cell_idx
                    new_previous_cells.append({
                        'cell_idx': cell_idx,
                        'cx': current_cells[uc]['cx'],
                        'cy': current_cells[uc]['cy'],
                    })

                previous_cells = new_previous_cells

            else:
                # 前フレームが空、あるいは今回フレームが空の場合
                # 今フレームがあるのに前フレームが無い → 全セルが新規
                new_previous_cells = []
                for ccell in current_cells:
                    cell_idx = next_cell_idx
                    next_cell_idx += 1
                    new_previous_cells.append({
                        'cell_idx': cell_idx,
                        'cx': ccell['cx'],
                        'cy': ccell['cy'],
                    })
                previous_cells = new_previous_cells

            # DB 保存
            async with async_session() as session:
                for pcell in previous_cells:
                    # 既に割り当てられているものだけが対象
                    # ただし current_cells は「今回」検出分なので、pCell 内に contour が入ってない場合がある。
                    # matched_pairs 内のインデックスを覚えておいて contour 等を参照する場合は工夫が必要です。
                    # ここではシンプルに current_cells から再検索して保存します。
                    ccell = None
                    # pcell に対応する center をもつ current cell を探す
                    # （小数点ずれを考慮して誤差0pixelだと一致しないこともあるため注意）
                    for c in current_cells:
                        if c['cx'] == pcell['cx'] and c['cy'] == pcell['cy']:
                            ccell = c
                            break
                    if not ccell:
                        # このフレームでは見つからない(前フレームから継続のみ)
                        continue

                    cx, cy = ccell['cx'], ccell['cy']
                    cnt = ccell['contour']
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)

                    x1 = max(0, cx - output_size[0] // 2)
                    y1 = max(0, cy - output_size[1] // 2)
                    x2 = min(ph_img.shape[1], cx + output_size[0] // 2)
                    y2 = min(ph_img.shape[0], cy + output_size[1] // 2)

                    if (y2 - y1) != output_size[1] or (x2 - x1) != output_size[0]:
                        continue

                    cropped_ph = ph_img[y1:y2, x1:x2]
                    cropped_ph_gray = cv2.cvtColor(cropped_ph, cv2.COLOR_BGR2GRAY)
                    ph_gray_encode = cv2.imencode(".png", cropped_ph_gray)[1].tobytes()

                    fluo1_gray_encode = None
                    if fluo1_img is not None:
                        cropped_fluo1 = fluo1_img[y1:y2, x1:x2]
                        cropped_fluo1_gray = cv2.cvtColor(cropped_fluo1, cv2.COLOR_BGR2GRAY)
                        fluo1_gray_encode = cv2.imencode(".png", cropped_fluo1_gray)[1].tobytes()

                    fluo2_gray_encode = None
                    if fluo2_img is not None:
                        cropped_fluo2 = fluo2_img[y1:y2, x1:x2]
                        cropped_fluo2_gray = cv2.cvtColor(cropped_fluo2, cv2.COLOR_BGR2GRAY)
                        fluo2_gray_encode = cv2.imencode(".png", cropped_fluo2_gray)[1].tobytes()

                    contour_shifted = cnt.copy()
                    contour_shifted[:, :, 0] -= x1
                    contour_shifted[:, :, 1] -= y1

                    cell_idx = pcell['cell_idx']
                    # まだ base_ids に登録がなければULID割り当て
                    if cell_idx not in base_ids:
                        base_ids[cell_idx] = get_ulid()

                    new_ulid_value = get_ulid()
                    cell_obj = Cell(
                        cell_id=new_ulid_value,
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
                        time=frame_i + 1,  # time は1始まり
                        cell=cell_idx,
                        base_cell_id=base_ids[cell_idx],
                        is_dead=0,
                        gif_ph=None,
                        gif_fluo1=None,
                        gif_fluo2=None,
                    )

                    existing = await session.execute(
                        select(Cell).filter_by(cell_id=new_ulid_value, time=frame_i + 1)
                    )
                    if existing.scalar() is None:
                        session.add(cell_obj)

                await session.commit()

        # 最初のフレームに存在しないセルを削除する処理が必要ならここで行う（省略可）
        async with async_session() as session:
            first_frame = 1
            subquery = select(Cell.cell).where(Cell.field == field, Cell.time == first_frame)
            result = await session.execute(subquery)
            cells_with_first_frame = [row[0] for row in result]

            delete_stmt = (
                delete(Cell)
                .where(Cell.field == field)
                .where(~Cell.cell.in_(cells_with_first_frame))
            )
            await session.execute(delete_stmt)
            await session.commit()

        print("Hungarian-based tracking finished!")

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

        async with async_session() as session:
            result = await session.execute(
                select(Cell).filter_by(field=field, cell=cell_number, time=1)
            )
            cell = result.scalar()
            if cell is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for field={field}, cell={cell_number}",
                )

            if channel == "ph" and cell.gif_ph:
                return io.BytesIO(cell.gif_ph)
            elif channel == "fluo1" and cell.gif_fluo1:
                return io.BytesIO(cell.gif_fluo1)
            elif channel == "fluo2" and cell.gif_fluo2:
                return io.BytesIO(cell.gif_fluo2)

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
                else:
                    img_binary = row.img_fluo2

                if img_binary is None:
                    continue

                np_img_gray = cv2.imdecode(
                    np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if np_img_gray is None:
                    continue

                if row.contour is not None:
                    try:
                        contours = pickle.loads(row.contour)
                        if not isinstance(contours, list):
                            contours = [contours]
                        np_img_color = cv2.cvtColor(np_img_gray, cv2.COLOR_GRAY2BGR)
                        for c in contours:
                            c = np.array(c, dtype=np.float32)
                            if len(c.shape) == 2:
                                c = c[:, np.newaxis, :]
                            c = c.astype(np.int32)
                            cv2.drawContours(np_img_color, [c], -1, (0, 0, 255), 2)
                        np_img_rgb = cv2.cvtColor(np_img_color, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(np_img_rgb)
                    except Exception:
                        pil_img = Image.fromarray(np_img_gray)
                else:
                    pil_img = Image.fromarray(np_img_gray)

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

                np_img_color = cv2.cvtColor(np_img_gray, cv2.COLOR_GRAY2BGR)
                original_h, original_w = np_img_gray.shape[:2]

                if first_valid_img_size is None:
                    first_valid_img_size = (original_w, original_h)

                target_w, target_h = first_valid_img_size
                if (original_w, original_h) != (target_w, target_h):
                    scale_x = target_w / original_w
                    scale_y = target_h / original_h
                    np_img_color = cv2.resize(
                        np_img_color, (target_w, target_h), interpolation=cv2.INTER_AREA
                    )
                else:
                    scale_x, scale_y = 1.0, 1.0

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
                        pass

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
        draw_frame_number: bool = True,
    ) -> io.BytesIO:
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
            if channel == "ph":
                img_binary = row.img_ph
            elif channel == "fluo1":
                img_binary = row.img_fluo1
            else:
                img_binary = row.img_fluo2

            if img_binary is None:
                continue

            np_img = cv2.imdecode(
                np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if np_img is None:
                continue

            if draw_contour and row.contour is not None:
                try:
                    contour_data = pickle.loads(row.contour)
                    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(
                        bgr_img,
                        [np.array(contour_data, dtype=np.int32)],
                        -1,
                        (0, 255, 0),
                        1,
                    )
                    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                except Exception:
                    pil_img = Image.fromarray(np_img)
            else:
                pil_img = Image.fromarray(np_img)

            if draw_frame_number:
                draw = ImageDraw.Draw(pil_img)
                font = ImageFont.load_default()
                draw.text((5, 5), f"Frame: {i+1}", font=font, fill=(255, 255, 255))

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
                        contour_area = cv2.contourArea(
                            np.array(contour_data, dtype=np.int32)
                        )
                except Exception:
                    contour_area = 0.0
            areas.append(contour_area)

        return areas

    async def get_pixel_sd_by_cell_number(
        self,
        field: str,
        cell_number: int,
        channel: Literal["ph", "fluo1", "fluo2"] = "ph",
    ) -> list[float]:
        """Return standard deviation of pixel values inside the contour for each frame."""
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

        sds: list[float] = []
        for row in cells:
            if channel == "ph":
                img_blob = row.img_ph
            elif channel == "fluo1":
                img_blob = row.img_fluo1
            else:
                img_blob = row.img_fluo2

            if img_blob is None:
                sds.append(0.0)
                continue

            np_img = cv2.imdecode(np.frombuffer(img_blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if np_img is None:
                sds.append(0.0)
                continue

            if row.contour:
                try:
                    contours_data = pickle.loads(row.contour)
                    if not isinstance(contours_data, list):
                        contours_data = [contours_data]
                    mask = np.zeros(np_img.shape, dtype=np.uint8)
                    for c in contours_data:
                        cnt = np.array(c, dtype=np.int32)
                        cv2.drawContours(mask, [cnt], -1, 255, -1)
                    pixel_vals = np_img[mask == 255]
                except Exception:
                    pixel_vals = np_img.flatten()
            else:
                pixel_vals = np_img.flatten()

            sd = float(pixel_vals.std()) if pixel_vals.size else 0.0
            sds.append(sd)

        return sds

    async def get_pixel_cv_by_cell_number(
        self,
        field: str,
        cell_number: int,
        channel: Literal["ph", "fluo1", "fluo2"] = "ph",
    ) -> list[float]:
        """Return coefficient of variation of pixel values inside the contour for each frame."""
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

        cvs: list[float] = []
        for row in cells:
            if channel == "ph":
                img_blob = row.img_ph
            elif channel == "fluo1":
                img_blob = row.img_fluo1
            else:
                img_blob = row.img_fluo2

            if img_blob is None:
                cvs.append(0.0)
                continue

            np_img = cv2.imdecode(np.frombuffer(img_blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if np_img is None:
                cvs.append(0.0)
                continue

            if row.contour:
                try:
                    contours_data = pickle.loads(row.contour)
                    if not isinstance(contours_data, list):
                        contours_data = [contours_data]
                    mask = np.zeros(np_img.shape, dtype=np.uint8)
                    for c in contours_data:
                        cnt = np.array(c, dtype=np.int32)
                        cv2.drawContours(mask, [cnt], -1, 255, -1)
                    pixel_vals = np_img[mask == 255]
                except Exception:
                    pixel_vals = np_img.flatten()
            else:
                pixel_vals = np_img.flatten()

            mean_val = float(pixel_vals.mean()) if pixel_vals.size else 0.0
            sd_val = float(pixel_vals.std()) if pixel_vals.size else 0.0
            cv = sd_val / mean_val if mean_val > 0 else 0.0
            cvs.append(cv)

        return cvs

    async def get_contour_areas_csv_by_cell_number(
        self, field: str, cell_number: int
    ) -> StreamingResponse:
        areas = await self.get_contour_areas_by_cell_number(field, cell_number)
        df = pd.DataFrame({"frame": list(range(len(areas))), "area": areas})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def get_pixel_sd_csv_by_cell_number(
        self,
        field: str,
        cell_number: int,
        channel: Literal["ph", "fluo1", "fluo2"] = "ph",
    ) -> StreamingResponse:
        sds = await self.get_pixel_sd_by_cell_number(field, cell_number, channel)
        df = pd.DataFrame({"frame": list(range(len(sds))), "sd": sds})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def get_pixel_cv_csv_by_cell_number(
        self,
        field: str,
        cell_number: int,
        channel: Literal["ph", "fluo1", "fluo2"] = "ph",
    ) -> StreamingResponse:
        cvs = await self.get_pixel_cv_by_cell_number(field, cell_number, channel)
        df = pd.DataFrame({"frame": list(range(len(cvs))), "cv": cvs})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def get_area_vs_sd_csv_by_cell_number(
        self,
        field: str,
        cell_number: int,
        channel: Literal["ph", "fluo1", "fluo2"] = "ph",
    ) -> StreamingResponse:
        """Return CSV of contour area vs. pixel SD for each frame."""
        areas = await self.get_contour_areas_by_cell_number(field, cell_number)
        sds = await self.get_pixel_sd_by_cell_number(field, cell_number, channel)
        df = pd.DataFrame({"area": areas, "sd": sds})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def get_area_vs_cv_csv_by_cell_number(
        self,
        field: str,
        cell_number: int,
        channel: Literal["ph", "fluo1", "fluo2"] = "ph",
    ) -> StreamingResponse:
        """Return CSV of contour area vs. pixel CV for each frame."""
        areas = await self.get_contour_areas_by_cell_number(field, cell_number)
        cvs = await self.get_pixel_cv_by_cell_number(field, cell_number, channel)
        df = pd.DataFrame({"area": areas, "cv": cvs})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def replot_cell(
        self,
        field: str,
        cell_number: int,
        channel: str,
        degree: int,
        dark_mode: bool = False,
    ) -> io.BytesIO:
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

            # replot前にキャッシュクリア
            if hasattr(CellDBAsyncChores.replot, "cache_clear"):
                CellDBAsyncChores.replot.cache_clear()

            buf = await CellDBAsyncChores.replot(image_fluo_raw, cell.contour, degree, dark_mode)
            buf.seek(0)
            frames.append(Image.open(buf))

        if not frames:
            raise HTTPException(
                status_code=404,
                detail=f"No frames were generated for field={field}, cell={cell_number}",
            )

        gif_buf = io.BytesIO()
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

    async def get_cell_heatmap_gif(
        self,
        field: str,
        cell_number: int,
        channel: Literal["fluo1", "fluo2"] = "fluo1",
        degree: int = 3,
    ) -> io.BytesIO:
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
                detail=f"No data found for field={field}, cell={cell_number}",
            )

        frames: list[Image.Image] = []

        for cell in cells:
            if channel == "fluo1":
                img_blob = cell.img_fluo1
            else:
                img_blob = cell.img_fluo2

            if not img_blob or not cell.contour:
                continue

            path = await CellDBAsyncChores.find_path_return_list(
                img_blob, cell.contour, degree
            )
            buf = await CellDBAsyncChores.heatmap_path(path)
            buf.seek(0)
            frames.append(Image.open(buf))

        if not frames:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No heatmap frames generated for field={field}, "
                    f"cell={cell_number}, channel={channel}"
                ),
            )

        gif_buf = io.BytesIO()
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

    async def get_cell_heatmap_timecourse_as_single_image(
        self,
        field: str,
        cell_number: int,
        channel: Literal["fluo1", "fluo2"] = "fluo1",
        degree: int = 3,
    ) -> io.BytesIO:
        """Return heatmap timecourse as a single PNG lined horizontally."""

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
                detail=f"No data found for field={field}, cell={cell_number}",
            )

        frames: list[Image.Image] = []
        for cell in cells:
            if channel == "fluo1":
                img_blob = cell.img_fluo1
            else:
                img_blob = cell.img_fluo2

            if not img_blob or not cell.contour:
                continue

            path = await CellDBAsyncChores.find_path_return_list(
                img_blob, cell.contour, degree
            )
            buf = await CellDBAsyncChores.heatmap_path(path)
            buf.seek(0)
            img = Image.open(buf).resize((200, 200), Image.Resampling.LANCZOS)
            frames.append(img)

        if not frames:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No heatmap frames generated for field={field}, "
                    f"cell={cell_number}, channel={channel}"
                ),
            )

        total_width = 200 * len(frames)
        out_img = Image.new("RGB", (total_width, 200), (0, 0, 0))
        offset_x = 0
        for img in frames:
            if img.mode not in ["RGB", "RGBA"]:
                img = img.convert("RGB")
            out_img.paste(img, (offset_x, 0))
            offset_x += 200

        buf_out = io.BytesIO()
        out_img.save(buf_out, format="PNG")
        buf_out.seek(0)
        return buf_out

    async def get_cell_timecourse_as_single_image(
        self,
        field: str,
        cell_number: int,
        channel_mode: Literal[
            "ph",
            "ph_replot",
            "fluo1",
            "fluo1_replot",
            "fluo2",
            "fluo2_replot",
        ] = "ph",
        degree: int = 0,
        draw_contour: bool = True,
    ) -> io.BytesIO:
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
                detail=f"No data found for field={field}, cell={cell_number}",
            )

        frames: list[Image.Image] = []

        async def get_frame_blob(row: Cell) -> bytes | None:
            if "ph" in channel_mode:
                raw_blob = row.img_ph
            elif "fluo1" in channel_mode:
                raw_blob = row.img_fluo1
            else:
                raw_blob = row.img_fluo2

            if raw_blob is None:
                return None

            if "_replot" in channel_mode:
                if not row.contour:
                    return None
                if hasattr(CellDBAsyncChores.replot, "cache_clear"):
                    CellDBAsyncChores.replot.cache_clear()
                buf = await CellDBAsyncChores.replot(raw_blob, row.contour, degree, dark_mode)
                return buf.getvalue()
            else:
                return raw_blob

        for row in cells:
            blob = await get_frame_blob(row)
            if not blob:
                continue

            is_replot_mode = ("_replot" in channel_mode)
            if not is_replot_mode and draw_contour:
                np_img_gray = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if np_img_gray is None:
                    continue
                np_img_color = cv2.cvtColor(np_img_gray, cv2.COLOR_GRAY2BGR)

                if row.contour:
                    try:
                        contours_data = pickle.loads(row.contour)
                        if not isinstance(contours_data, list):
                            contours_data = [contours_data]
                        for c in contours_data:
                            cnt = np.array(c, dtype=np.int32)
                            cv2.drawContours(np_img_color, [cnt], -1, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"[WARN] Failed to parse contour: {e}")

                np_img_rgb = cv2.cvtColor(np_img_color, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(np_img_rgb)
            else:
                pil_img = Image.open(io.BytesIO(blob))

            pil_img = pil_img.resize((200, 200), Image.Resampling.LANCZOS)
            frames.append(pil_img)

        if not frames:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No valid frames found after decoding for "
                    f"field={field}, cell={cell_number}, channel_mode={channel_mode}"
                ),
            )

        total_width = 200 * len(frames)
        height = 200

        out_img = Image.new("RGB", (total_width, height), (0, 0, 0))

        offset_x = 0
        for img in frames:
            if img.mode not in ["RGB", "RGBA"]:
                img = img.convert("RGB")
            out_img.paste(img, (offset_x, 0))
            offset_x += 200

        png_buffer = io.BytesIO()
        out_img.save(png_buffer, format="PNG")
        png_buffer.seek(0)
        return png_buffer

    async def get_all_channels_timecourse_as_single_image(
        self,
        field: str,
        cell_number: int,
        degree: int = 0,
        draw_contour: bool = True,
    ) -> io.BytesIO:
        modes = [
            "ph",
            "fluo1",
            "fluo2",
        ]

        row_images = []
        for mode in modes:
            try:
                buf = await self.get_cell_timecourse_as_single_image(
                    field=field,
                    cell_number=cell_number,
                    channel_mode=mode,
                    degree=degree,
                    draw_contour=draw_contour,
                )
            except HTTPException as e:
                print(f"[WARN] Skipped mode={mode}: {e.detail}")
                continue

            row_img = Image.open(buf)
            row_images.append(row_img)

        try:
            hm_buf = await self.get_cell_heatmap_timecourse_as_single_image(
                field=field,
                cell_number=cell_number,
                channel="fluo1",
                degree=degree,
            )
            row_images.append(Image.open(hm_buf))
        except HTTPException as e:
            print(f"[WARN] Skipped heatmap: {e.detail}")

        if not row_images:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No valid images found for field={field}, "
                    f"cell={cell_number} in any channel modes."
                ),
            )

        max_width = max(im.width for im in row_images)
        total_height = sum(im.height for im in row_images)

        combined_img = Image.new("RGB", (max_width, total_height), (255, 255, 255))

        offset_y = 0
        for row_img in row_images:
            combined_img.paste(row_img, (0, offset_y))
            offset_y += row_img.height

        out_buf = io.BytesIO()
        combined_img.save(out_buf, format="PNG")
        out_buf.seek(0)
        return out_buf

    async def get_cell_heatmap(
        self,
        field: str,
        cell_number: int,
        channel: Literal["fluo1", "fluo2"] = "fluo1",
        degree: int = 3,
    ) -> io.BytesIO:
        """Return heatmap PNG for the specified cell."""

        async with get_session(self.dbname) as session:
            result = await session.execute(
                select(Cell)
                .filter_by(field=field, cell=cell_number)
                .order_by(Cell.time)
            )
            cell = result.scalars().first()

        if not cell:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for field={field}, cell={cell_number}",
            )

        if channel == "fluo1":
            img_blob = cell.img_fluo1
        else:
            img_blob = cell.img_fluo2

        if not img_blob:
            raise HTTPException(
                status_code=404,
                detail=f"No {channel} data found for field={field}, cell={cell_number}",
            )
        if not cell.contour:
            raise HTTPException(
                status_code=404,
                detail=f"No contour data found for field={field}, cell={cell_number}",
            )

        path = await CellDBAsyncChores.find_path_return_list(img_blob, cell.contour, degree)
        buf = await CellDBAsyncChores.heatmap_path(path)
        buf.seek(0)
        return buf
