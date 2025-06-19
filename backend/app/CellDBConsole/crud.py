from __future__ import annotations

import aiofiles
import aiofiles.os
import asyncio
import cv2
import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from matplotlib.figure import Figure
from numpy.linalg import eig, inv
from scipy.integrate import quad
from scipy.optimize import minimize
from sqlalchemy import update
from sqlalchemy.future import select
from typing import Literal
from CellDBConsole.schemas import CellId, CellMorhology, ListDBresponse
from Dropbox.crud import DropboxCrud
from database import get_session, Cell
from exceptions import CellNotFoundError

matplotlib.use("Agg")


@dataclass
class Point:
    def __init__(self, u1: float, G: float) -> None:
        self.u1 = u1
        self.G = G

    def __gt__(self, other) -> bool:
        return self.u1 > other.u1

    def __lt__(self, other) -> bool:
        return self.u1 < other.u1

    def __repr__(self) -> str:
        return f"({self.u1},{self.G})"


class SyncChores:
    @staticmethod
    def poly_fit(U: list[list[float]], degree: int = 1) -> list[float]:
        """
        与えられた (f, u1) データから多項式近似係数を計算する関数。
        """
        u1_values = np.array([i[1] for i in U])
        f_values = np.array([i[0] for i in U])
        W = np.vander(u1_values, degree + 1)
        return inv(W.T @ W) @ W.T @ f_values

    @staticmethod
    def calc_arc_length(theta: list[float], u_1_1: float, u_1_2: float) -> float:
        """
        与えられた多項式係数 theta の関数 f(x) の弧長を数値積分で計算する関数。
        """

        def fx(x):
            # f'(x) = sum(i * j * x^(i-1) for i, j in enumerate(theta[::-1][1:], start=1))
            # 1 + [f'(x)]^2 の平方根を返す
            return np.sqrt(
                1
                + sum(
                    i * j * x ** (i - 1) for i, j in enumerate(theta[::-1][1:], start=1)
                )
                ** 2
            )

        arc_length, _ = quad(fx, u_1_1, u_1_2, epsabs=1e-01)
        return arc_length

    @staticmethod
    def find_minimum_distance_and_point(coefficients, x_Q, y_Q):
        """
        与えられた多項式係数と点 Q (x_Q, y_Q) 間の最短距離を求め、その点を返す関数。
        """

        def f_x(x):
            return sum(
                coefficient * x**i for i, coefficient in enumerate(coefficients[::-1])
            )

        def distance(x):
            return np.sqrt((x - x_Q) ** 2 + (f_x(x) - y_Q) ** 2)

        # Nelder-Mead 法で最適化 (精度は下げて高速化)
        result = minimize(
            distance, 0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-2}
        )

        x_min = result.x[0]
        min_distance = distance(x_min)
        min_point = (x_min, f_x(x_min))
        return min_distance, min_point

    @staticmethod
    def basis_conversion(
        contour: list[list[int]],
        X: np.ndarray,
        center_x: float,
        center_y: float,
        coordinates_incide_cell: list[list[int]],
    ) -> list[list[float]]:
        """
        X = [ [x1, x2, ...], [y1, y2, ...] ] 形式の配列に対して
        共分散行列から主成分を計算し、その基底に変換する関数。
        """
        Sigma = np.cov(X)
        eigenvalues, eigenvectors = eig(Sigma)

        # 大きい固有値の成分を u1 ( = 主軸 ) としている
        if eigenvalues[1] < eigenvalues[0]:
            Q = np.array([eigenvectors[1], eigenvectors[0]])
            U = [Q.transpose() @ np.array([i, j]) for i, j in coordinates_incide_cell]
            U = [[j, i] for i, j in U]
            contour_U = [Q.transpose() @ np.array([j, i]) for i, j in contour]
            contour_U = [[j, i] for i, j in contour_U]
            center = [center_x, center_y]
            u1_c, u2_c = center @ Q
        else:
            Q = np.array([eigenvectors[0], eigenvectors[1]])
            U = [
                Q.transpose() @ np.array([j, i]).transpose()
                for i, j in coordinates_incide_cell
            ]
            contour_U = [Q.transpose() @ np.array([i, j]) for i, j in contour]
            center = [center_x, center_y]
            u2_c, u1_c = center @ Q

        u1 = [i[1] for i in U]
        u2 = [i[0] for i in U]
        u1_contour = [i[1] for i in contour_U]
        u2_contour = [i[0] for i in contour_U]
        min_u1, max_u1 = min(u1), max(u1)
        return u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U

    @staticmethod
    def calculate_volume_and_widths(
        u1_adj: list[float], u2_adj: list[float], split_num: int, deltaL: float
    ) -> tuple[float, list[float]]:
        """
        セルを複数区画に分割し、円柱断面積近似で体積を求めるサンプル実装。
        """
        volume = 0
        widths = []
        y_mean = 0
        for i in range(split_num):
            x_0 = min(u1_adj) + i * deltaL
            x_1 = min(u1_adj) + (i + 1) * deltaL
            points = [p for p in zip(u1_adj, u2_adj) if x_0 <= p[0] <= x_1]
            if points:
                y_mean = sum([i[1] for i in points]) / len(points)
            volume += y_mean**2 * np.pi * deltaL
            widths.append(y_mean)
        return volume, widths

    @staticmethod
    def box_plot(
        values: list[float],
        target_val: float,
        y_label: str,
        cell_id: str,
        label: str | None = None,
    ) -> io.BytesIO:
        """
        与えられた values に対して box プロットを行い、target_val を強調表示する。
        """
        fig = plt.figure(figsize=(8, 6))
        closest_point = min(values, key=lambda x: abs(x - target_val))
        if abs(closest_point - target_val) <= 0.001:
            close_points = [closest_point]
        else:
            close_points = []

        other_points = [val for val in values if val not in close_points]

        x_other = np.random.normal(1, 0.02, size=len(other_points))
        plt.plot(x_other, other_points, "o", alpha=0.5, label="Other cells")
        if close_points:
            x_close = np.random.normal(1, 0.04, size=len(close_points))
            plt.plot(
                x_close,
                close_points,
                "o",
                color="red",
                alpha=0.5,
                label=f"{cell_id}",
            )

        plt.boxplot(values, flierprops=dict(marker=""))
        plt.ylim(0, 1.05)
        plt.ylabel(y_label)
        if label:
            plt.text(1.3, 0.5, label, fontsize=12, ha="center", va="center")
        plt.legend()
        plt.gca().axes.get_xaxis().set_visible(False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)
        return buf

    @staticmethod
    def create_histogram(
        data: list[int],
        num_bins: int,
        title: str,
        xlabel: str,
        ylabel: str,
        bin_scale: Literal["default", "normalized"],
    ) -> io.BytesIO:
        """
        0-255の範囲で整数のリスト、または0-1で正規化されたリストからヒストグラムを作成し、バッファとして返す。

        Parameters
        ----------
        data : list[int]
            ヒストグラム化したいデータのリスト。
            bin_scale="default" の場合は 0-255 の整数値を想定。
            bin_scale="normalized" の場合は 0-1 に正規化された値を想定。
        num_bins : int
            ヒストグラムのビン数。
        title : str
            グラフのタイトル（今回は使用していないが、拡張の可能性あり）。
        xlabel : str
            x 軸のラベル（ここでは「cell id : {xlabel}」の形でタイトルに使用）。
        ylabel : str
            y 軸のラベル（例: "Count"など）。
        bin_scale : Literal["default", "normalized"]
            "default" の場合 0-255 でヒストグラムを作成。
            "normalized" の場合 0-1 でヒストグラムを作成。

        Returns
        -------
        io.BytesIO
            作成したヒストグラムの画像データを格納したバッファ。
        """

        # Figure を6x6インチで作成
        fig = plt.figure(figsize=(6, 6))

        # bin_scale によってヒストグラムのビンの取り方や表示範囲を変える
        if bin_scale == "normalized":
            # 0-1 の範囲で num_bins 個のビンを作成
            bins = np.linspace(0, 1, num_bins + 1)
            # ヒストグラムの描画(range=(0,1) で0-1に収める)
            plt.hist(data, bins=bins, range=(0, 1), edgecolor="black", color="skyblue")
            # x 軸の表示範囲も 0-1 に設定
            plt.xlim(0, 1)
        else:
            # 0-255 の範囲で num_bins 個のビンを作成
            bins = np.linspace(0, 255, num_bins + 1)
            # ヒストグラムの描画(range=(0,255) で0-255に収める)
            plt.hist(
                data, bins=bins, range=(0, 255), edgecolor="black", color="skyblue"
            )
            # x 軸の表示範囲を 0-255 に設定
            plt.xlim(0, 255)

        # タイトルの設定：cell id : {xlabel}
        plt.title(f"cell id : {xlabel}", fontsize=10)
        # x 軸ラベル（今回は「Fluo. intensity」）
        plt.xlabel(
            (
                "Fluo. intensity"
                if bin_scale == "default"
                else "Fluo. intensity(normalized)"
            ),
            fontsize=10,
        )
        # y 軸ラベル
        plt.ylabel("Count", fontsize=10)
        # グリッドを薄めに表示
        plt.grid(True, alpha=0.3)

        # バッファにプロットを書き出し
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        # バッファのポインタを先頭に戻す
        buf.seek(0)

        # plt.close() でメモリ解放
        plt.close(fig)

        return buf

    @staticmethod
    def plot_paths(paths: list[float]) -> io.BytesIO:
        """
        渡されたパスリストの強度変化を折れ線グラフで表示する。
        """
        fig = plt.figure(figsize=(8, 6))
        relative_positions = range(len(paths))
        plt.plot(relative_positions, paths, label="Path")
        plt.xlabel("Relative position")
        plt.ylabel("Normalized Fluorescence Intensity")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)
        return buf

    @staticmethod
    def heatmap_path(paths: list[list[float]]) -> io.BytesIO:
        """
        1次元データを縦に並べたヒートマップを描画する。
        """
        paths = [i[1] for i in paths]
        data = np.array(paths).reshape(-1, 1)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(data, cmap="inferno", interpolation="nearest", aspect="auto")
        fig.colorbar(cax)
        plt.ylabel("Relative position")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)
        return buf

    @staticmethod
    async def heatmap_all_abs(
        u1s: list[list[float]], Gs: list[list[float]], label: str = "1"
    ) -> io.BytesIO:
        """
        複数の細胞パスを積み上げたヒートマップのサンプル（再描画用）。
        """

        @dataclass
        class HeatMapVector:
            index: int
            u1: list[float]
            G: list[float]
            length: float

            def __gt__(self, other):
                return sum(self.G) > sum(other.G)

        # ダミーとして外部呼び出しになっている AsyncChores.find_path_return_list(...) は
        # 通常は別途実装や入力値を取得してくる想定
        u1, G, *_ = await AsyncChores.find_path_return_list(label)

        heatmap_vectors = sorted(
            [
                HeatMapVector(index=i, u1=u1, G=G, length=max(u1) - min(u1))
                for i, (u1, G) in enumerate(zip(u1s, Gs))
            ],
        )
        max_length = max(heatmap_vectors).length
        heatmap_vectors = [
            HeatMapVector(
                index=vec.index,
                u1=[d + (max_length - vec.length) / 2 - max_length / 2 for d in vec.u1],
                G=vec.G,
                length=vec.length,
            )
            for vec in heatmap_vectors
        ]
        u1_min = min(map(min, [vec.u1 for vec in heatmap_vectors]))
        u1_max = max(map(max, [vec.u1 for vec in heatmap_vectors]))
        cmap = plt.cm.inferno
        fig, ax = plt.subplots(figsize=(14, 9))
        for idx, vec in enumerate(heatmap_vectors):
            u1 = vec.u1
            G_normalized = (np.array(vec.G) - min(vec.G)) / (max(vec.G) - min(vec.G))
            colors = cmap(G_normalized)
            offset = len(heatmap_vectors) - idx - 1
            for i in range(len(u1) - 1):
                ax.plot([offset, offset], u1[i : i + 2], color=colors[i], lw=10)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Normalized G Value")
        ax.set_ylim([u1_min, u1_max])
        ax.set_xlim([-0.5, len(heatmap_vectors) - 0.5])
        ax.set_ylabel("cell length (px)")
        ax.set_xlabel("cell number")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        buf.seek(0)
        plt.close(fig)
        return buf


class AsyncChores:
    @staticmethod
    async def upload_file_chunked(data: UploadFile) -> None:
        """
        100MB 単位で分割アップロードするサンプル実装。
        """
        chunk_size = 1024 * 1024 * 100  # 100MB
        save_name = (
            f"databases/{data.filename.split('/')[-1].split('.')[0]}-uploaded.db"
        )
        async with aiofiles.open(f"{save_name}", "wb") as f:
            while True:
                content = await data.read(chunk_size)
                if not content:
                    break
                await f.write(content)
            await data.close()

    @staticmethod
    async def get_database_names(handle_id: str | None = None) -> ListDBresponse:
        """
        databases/ ディレクトリ下の .db ファイル一覧を取得する関数。
        handle_id が指定されている場合は、各データベース内の Cell テーブルの user_id カラムに
        handle_id が存在するデータベースのみを返す。
        """
        loop = asyncio.get_running_loop()
        names = await loop.run_in_executor(None, os.listdir, "databases/")
        db_files = [i for i in names if i.endswith(".db")]

        if handle_id is None:
            return ListDBresponse(databases=db_files)

        async def db_contains_user_id(db_file: str, user_id: str) -> bool:
            async for session in get_session(dbname=db_file):
                stmt = select(Cell).where(Cell.user_id == user_id)
                result = await session.execute(stmt)
                cell = result.scalars().first()
                await session.close()
                return cell is not None
            return False

        filtered_dbs = []
        for db_file in db_files:
            if await db_contains_user_id(db_file, handle_id):
                filtered_dbs.append(db_file)
        print(filtered_dbs)
        print("++++++++++++++++++++")
        print("++++++++++++++++++++")
        print("++++++++++++++++++++")
        print("++++++++++++++++++++")
        print("++++++++++++++++++++")
        print("++++++++++++++++++++")
        return ListDBresponse(databases=filtered_dbs)

    @staticmethod
    async def validate_database_name(db_name: str) -> None:
        """
        db_name が実際に存在するか確認。
        """
        res = await AsyncChores.get_database_names()
        databases = res.databases
        if db_name not in databases:
            raise ValueError("Database with given name does not exist")

    @staticmethod
    async def async_imdecode(data: bytes) -> np.ndarray:
        """
        画像バイト列を非同期で OpenCV 形式にデコード。
        """
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            img = await loop.run_in_executor(
                executor, cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
            )
        return img

    @staticmethod
    async def async_cv2_imencode(img) -> tuple[bool, np.ndarray]:
        """
        画像を PNG 形式に非同期でエンコード。
        """
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            success, buffer = await loop.run_in_executor(
                executor, lambda: cv2.imencode(".png", img)
            )
        return success, buffer

    #
    # ▼ 以下、重複処理を削除し get_points_inside_cell を使った実装に変更
    #
    @staticmethod
    async def calc_mean_normalized_fluo_intensity_incide_cell(
        image_fluo_raw: bytes, contour_raw: bytes
    ) -> float:
        """
        細胞領域の画素値を 0~255 で正規化したときの平均値を計算。
        """
        points = await AsyncChores.get_points_inside_cell(
            image_fluo_raw, contour_raw, normalize=False
        )
        # 0~255正規化
        return round(np.mean([p / 255 for p in points]), 2)

    @staticmethod
    async def calc_median_normalized_fluo_intensity_inside_cell(
        image_fluo_raw: bytes, contour_raw: bytes
    ) -> float:
        """
        細胞領域の画素値を最大値で正規化したときの中央値を計算。
        """
        points = await AsyncChores.get_points_inside_cell(
            image_fluo_raw, contour_raw, normalize=False
        )
        max_val = np.max(points) if len(points) else 1
        return round(np.median([p / max_val for p in points]), 2)

    @staticmethod
    async def calc_variance_normalized_fluo_intensity_inside_cell(
        image_fluo_raw: bytes, contour_raw: bytes
    ) -> float:
        """
        細胞領域の画素値を最大値で正規化したときの分散を計算。
        """
        points = await AsyncChores.get_points_inside_cell(
            image_fluo_raw, contour_raw, normalize=False
        )
        max_val = np.max(points) if len(points) else 1
        return round(np.var([p / max_val for p in points]), 2)

    @staticmethod
    async def get_points_inside_cell(
        image_fluo_raw: bytes, contour_raw: bytes, normalize: bool = False
    ) -> list[float]:
        """
        細胞内の画素値を返す。
        normalize=True で最大値(255ではなく)で正規化。
        """
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            image_fluo = await loop.run_in_executor(
                executor,
                cv2.imdecode,
                np.frombuffer(image_fluo_raw, np.uint8),
                cv2.IMREAD_COLOR,
            )
            image_fluo_gray = await loop.run_in_executor(
                executor, cv2.cvtColor, image_fluo, cv2.COLOR_BGR2GRAY
            )

            mask = np.zeros_like(image_fluo_gray)
            unpickled_contour = pickle.loads(contour_raw)
            await loop.run_in_executor(
                executor, cv2.fillPoly, mask, [unpickled_contour], 255
            )

            coords_inside_cell_1 = np.column_stack(np.where(mask))
            points_inside_cell_1 = image_fluo_gray[
                coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
            ]
            points_inside_cell_1 = points_inside_cell_1.flatten()

            if normalize and len(points_inside_cell_1) > 0:
                max_val = np.max(points_inside_cell_1)
                normalized_points = [i / max_val for i in points_inside_cell_1]
                return normalized_points

        return points_inside_cell_1

    @staticmethod
    async def draw_contour(
        image: np.ndarray, contour: bytes, thickness: int = 1
    ) -> np.ndarray:
        """
        輪郭を画像上に描画。
        """
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            contour = pickle.loads(contour)
            image = await loop.run_in_executor(
                executor,
                lambda: cv2.drawContours(image, contour, -1, (0, 255, 0), thickness),
            )
        return image

    @staticmethod
    async def draw_scale_bar_with_centered_text(image_ph) -> np.ndarray:
        """
        1 pixel = 0.0625 um 前提で、5um のスケールバーを右下に描画。
        """
        pixels_per_um = 1 / 0.0625
        scale_bar_um = 5
        scale_bar_length_px = int(scale_bar_um * pixels_per_um)
        scale_bar_thickness = 2
        scale_bar_color = (255, 255, 255)

        margin = 20
        x1 = image_ph.shape[1] - margin - scale_bar_length_px
        y1 = image_ph.shape[0] - margin
        x2 = x1 + scale_bar_length_px
        y2 = y1 + scale_bar_thickness

        cv2.rectangle(
            image_ph, (x1, y1), (x2, y2), scale_bar_color, thickness=cv2.FILLED
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{scale_bar_um} um"
        text_scale = 0.4
        text_thickness = 0
        text_color = (255, 255, 255)

        text_size = cv2.getTextSize(text, font, text_scale, text_thickness)[0]
        text_x = x1 + (scale_bar_length_px - text_size[0]) // 2
        text_y = y2 + text_size[1] + 5

        cv2.putText(
            image_ph,
            text,
            (text_x, text_y),
            font,
            text_scale,
            text_color,
            text_thickness,
        )
        return image_ph

    @staticmethod
    async def async_eig(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            eigenvalues, eigenvectors = await loop.run_in_executor(executor, eig, Sigma)
        return eigenvalues, eigenvectors

    @staticmethod
    async def async_pickle_loads(data: bytes) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            result = await loop.run_in_executor(executor, pickle.loads, data)
        return result

    @staticmethod
    async def calculate_volume_and_widths(
        u1_adj: list[float], u2_adj: list[float], split_num: int, deltaL: float
    ) -> tuple[float, list[float]]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            result = await loop.run_in_executor(
                executor,
                SyncChores.calculate_volume_and_widths,
                u1_adj,
                u2_adj,
                split_num,
                deltaL,
            )
        return result

    @staticmethod
    async def poly_fit(U: list[list[float]], degree: int = 1) -> list[float]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            result = await loop.run_in_executor(
                executor, SyncChores.poly_fit, U, degree
            )
        return result

    @staticmethod
    async def calc_arc_length(theta: list[float], u_1_1: float, u_1_2: float) -> float:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            result = await loop.run_in_executor(
                executor, SyncChores.calc_arc_length, theta, u_1_1, u_1_2
            )
        return result

    @staticmethod
    async def find_minimum_distance_and_point(
        coefficients, x_Q, y_Q
    ) -> tuple[float, tuple[float, float]]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            result = await loop.run_in_executor(
                executor,
                SyncChores.find_minimum_distance_and_point,
                coefficients,
                x_Q,
                y_Q,
            )
        return result

    @staticmethod
    async def get_contour(contour: bytes) -> np.ndarray:
        contour_unpickled = await AsyncChores.async_pickle_loads(contour)
        contour = np.array([[i, j] for i, j in [i[0] for i in contour_unpickled]])
        return {"raw": contour, "converted": contour}

    @staticmethod
    async def save_fig_async(fig: Figure, filename: str) -> None:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            await loop.run_in_executor(executor, fig.savefig, filename)
            plt.close(fig)

    @staticmethod
    async def morpho_analysis(
        image_ph: bytes, image_fluo_raw: bytes, contour_raw: bytes, degree: int
    ) -> np.ndarray:
        """
        位相差画像 (image_ph), 蛍光画像 (image_fluo_raw), 輪郭データ (contour_raw) を用いて
        細胞の面積, 体積, 幅, 長さ, 画素値の平均/中央値などを計算するサンプル。
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            image_fluo = await loop.run_in_executor(
                executor,
                cv2.imdecode,
                np.frombuffer(image_fluo_raw, np.uint8),
                cv2.IMREAD_COLOR,
            )
            image_fluo_gray = await loop.run_in_executor(
                executor, cv2.cvtColor, image_fluo, cv2.COLOR_BGR2GRAY
            )

            image_ph = await loop.run_in_executor(
                executor,
                cv2.imdecode,
                np.frombuffer(image_ph, np.uint8),
                cv2.IMREAD_COLOR,
            )
            image_ph_gray = await loop.run_in_executor(
                executor, cv2.cvtColor, image_ph, cv2.COLOR_BGR2GRAY
            )

            mask = np.zeros_like(image_fluo_gray)
            unpickled_contour = pickle.loads(contour_raw)
            await loop.run_in_executor(
                executor, cv2.fillPoly, mask, [unpickled_contour], 255
            )

        coords_inside_cell_1 = np.column_stack(np.where(mask))
        points_inside_cell_1 = image_fluo_gray[
            coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
        ]
        ph_points_inside_cell_1 = image_ph_gray[
            coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
        ]

        X = np.array(
            [
                [i[1] for i in coords_inside_cell_1],
                [i[0] for i in coords_inside_cell_1],
            ]
        )

        (
            u1,
            u2,
            u1_contour,
            u2_contour,
            min_u1,
            max_u1,
            u1_c,
            u2_c,
            U,
            contour_U,
        ) = SyncChores.basis_conversion(
            [list(i[0]) for i in unpickled_contour],
            X,
            image_fluo.shape[0] / 2,
            image_fluo.shape[1] / 2,
            coords_inside_cell_1,
        )

        # 中心を (0,0) に移動
        u1_adj = u1 - u1_c
        u2_adj = u2 - u2_c
        cell_length = max(u1_adj) - min(u1_adj)
        deltaL = cell_length / 20

        area = cv2.contourArea(np.array([i[0] for i in unpickled_contour]))

        if degree is None or degree == 1:
            volume, widths = await AsyncChores.calculate_volume_and_widths(
                u1_adj, [abs(i) for i in u2_adj], 20, deltaL
            )
            width = sum(sorted(widths, reverse=True)[:3]) * 2 / 3
        else:
            theta = await AsyncChores.poly_fit(np.array([u2, u1]).T, degree=degree)
            y = np.polyval(theta, np.linspace(min(u1_adj), max(u1_adj), 1000))
            # 弧長を計算する場合は任意の実装
            cell_length = await AsyncChores.calc_arc_length(theta, min(u1), max(u1))

            split_num = 20
            deltaL = cell_length / split_num
            raw_points = []
            for i, j in zip(u1, u2):
                min_distance, min_point = (
                    await AsyncChores.find_minimum_distance_and_point(theta, i, j)
                )
                arc_length = await AsyncChores.calc_arc_length(theta, min(u1), i)
                raw_points.append([arc_length, min_distance])
            raw_points.sort(key=lambda x: x[0])

            volume = 0
            widths = []
            y_mean = 0
            for i in range(split_num):
                x_0 = i * deltaL
                x_1 = (i + 1) * deltaL
                points = [p for p in raw_points if x_0 <= p[0] <= x_1]
                if points:
                    y_mean = sum([pt[1] for pt in points]) / len(points)
                volume += y_mean**2 * np.pi * deltaL
                widths.append(y_mean)
            width = sum(sorted(widths, reverse=True)[:3]) * 2 / 3

        return CellMorhology(
            area=round(area * 0.0625 * 0.0625, 2),
            volume=round(volume * 0.0625**3, 2),
            width=round(width * 0.0625, 2),
            length=round(cell_length * 0.0625, 2),
            mean_fluo_intensity=round(np.mean(points_inside_cell_1), 2),
            mean_ph_intensity=round(np.mean(ph_points_inside_cell_1), 2),
            mean_fluo_intensity_normalized=(
                round(np.mean(points_inside_cell_1) / np.max(points_inside_cell_1), 2)
                if np.max(points_inside_cell_1) != 0
                else 0
            ),
            mean_ph_intensity_normalized=(
                round(
                    np.mean(ph_points_inside_cell_1) / np.max(ph_points_inside_cell_1),
                    2,
                )
                if np.max(ph_points_inside_cell_1) != 0
                else 0
            ),
            median_fluo_intensity=round(np.median(points_inside_cell_1), 2),
            median_ph_intensity=round(np.median(ph_points_inside_cell_1), 2),
            median_fluo_intensity_normalized=(
                round(np.median(points_inside_cell_1) / np.max(points_inside_cell_1), 2)
                if np.max(points_inside_cell_1) != 0
                else 0
            ),
            median_ph_intensity_normalized=(
                round(
                    np.median(ph_points_inside_cell_1)
                    / np.max(ph_points_inside_cell_1),
                    2,
                )
                if np.max(ph_points_inside_cell_1) != 0
                else 0
            ),
        )

    @staticmethod
    async def find_path(
        image_fluo_raw: bytes, contour_raw: bytes, degree: int
    ) -> io.BytesIO:
        """
        細胞断面から得た (u1, u2, 輝度) データ全体を散布図化し、
        区画ごとに輝度の最大値を結んだパスを重ね描きするサンプル。
        """
        image_fluo = cv2.imdecode(
            np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR
        )
        image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(image_fluo_gray)
        unpickled_contour = pickle.loads(contour_raw)
        cv2.fillPoly(mask, [unpickled_contour], 255)

        coords_inside_cell_1 = np.column_stack(np.where(mask))
        points_inside_cell_1 = image_fluo_gray[
            coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
        ]

        X = np.array(
            [
                [i[1] for i in coords_inside_cell_1],
                [i[0] for i in coords_inside_cell_1],
            ]
        )

        (
            u1,
            u2,
            u1_contour,
            u2_contour,
            min_u1,
            max_u1,
            u1_c,
            u2_c,
            U,
            contour_U,
        ) = SyncChores.basis_conversion(
            [list(i[0]) for i in unpickled_contour],
            X,
            image_fluo.shape[0] / 2,
            image_fluo.shape[1] / 2,
            coords_inside_cell_1,
        )

        theta = await AsyncChores.poly_fit(U, degree=degree)
        raw_points: list[Point] = []
        for i, j, p in zip(u1, u2, points_inside_cell_1):
            min_distance, min_point = await AsyncChores.find_minimum_distance_and_point(
                theta, i, j
            )
            raw_points.append(Point(min_point[0], p))
        raw_points.sort()

        split_num: int = 35
        delta_L: float = (max(u1) - min(u1)) / split_num
        first_point: Point = raw_points[0]
        last_point: Point = raw_points[-1]
        path: list[Point] = [first_point]
        for i in range(1, int(split_num)):
            x_0 = min(u1) + i * delta_L
            x_1 = min(u1) + (i + 1) * delta_L
            points = [p for p in raw_points if x_0 <= p.u1 <= x_1]
            if len(points) == 0:
                continue
            point = max(points, key=lambda x: x.G)
            path.append(point)
        path.append(last_point)

        fig = plt.figure(figsize=(6, 6))
        plt.axis("equal")
        x = [i.u1 for i in raw_points]
        y = [i.G for i in raw_points]
        plt.scatter(x, y, s=10, cmap="jet", c=y)
        px, py = [i.u1 for i in path], [i.G for i in path]
        plt.scatter(px, py, s=50, color="magenta", zorder=100)
        plt.xlim(min(px) - 10, max(px) + 10)
        plt.plot(px, py, color="magenta")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)
        return buf

    @staticmethod
    async def replot(
        image_fluo_raw: bytes,
        contour_raw: bytes,
        degree: int,
    ) -> io.BytesIO:
        """
        (u1,u2) 平面に生データを散布し、多項式近似した曲線と輪郭を表示。
        重心 (u1_c, u2_c) を (0,0) に平行移動して描画する。
        """
        image_fluo = cv2.imdecode(
            np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR
        )
        image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(image_fluo_gray)
        unpickled_contour = pickle.loads(contour_raw)
        cv2.fillPoly(mask, [unpickled_contour], 255)

        coords_inside_cell_1 = np.column_stack(np.where(mask))
        points_inside_cell_1 = image_fluo_gray[
            coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
        ]

        X = np.array(
            [
                [i[1] for i in coords_inside_cell_1],  # x 座標
                [i[0] for i in coords_inside_cell_1],  # y 座標
            ]
        )

        (
            u1,
            u2,
            u1_contour,
            u2_contour,
            min_u1,
            max_u1,
            u1_c,
            u2_c,
            U,
            contour_U,
        ) = SyncChores.basis_conversion(
            [list(i[0]) for i in unpickled_contour],
            X,
            image_fluo.shape[0] / 2,
            image_fluo.shape[1] / 2,
            coords_inside_cell_1,
        )

        u1_shifted = u1 - u1_c
        u2_shifted = u2 - u2_c

        u1_contour_shifted = u1_contour - u1_c
        u2_contour_shifted = u2_contour - u2_c

        U_shifted = []
        for y_val, x_val in U:
            x_val_shifted = x_val - u1_c
            y_val_shifted = y_val - u2_c
            U_shifted.append([y_val_shifted, x_val_shifted])

        contour_U_shifted = []
        for y_val, x_val in contour_U:
            x_val_shifted = x_val - u1_c
            y_val_shifted = y_val - u2_c
            contour_U_shifted.append([y_val_shifted, x_val_shifted])

        min_u1_shifted = np.min(u1_shifted)
        max_u1_shifted = np.max(u1_shifted)
        min_u2_shifted = np.min(u2_shifted)
        max_u2_shifted = np.max(u2_shifted)

        fig = plt.figure(figsize=(6, 6))

        plt.scatter(u1_shifted, u2_shifted, s=5, label="Points in cell")

        plt.scatter([0], [0], color="red", s=100, label="Centroid (0,0)")

        plt.axis("equal")
        margin_width = 20
        margin_height = 20
        plt.scatter(
            [i[1] for i in U_shifted],  # x
            [i[0] for i in U_shifted],  # y
            c=points_inside_cell_1,
            cmap="jet",
            marker="o",
            s=20,
            label="Intensity",
        )

        plt.xlim([min_u1_shifted - margin_width, max_u1_shifted + margin_width])
        plt.ylim([min_u2_shifted - margin_height, max_u2_shifted + margin_height])

        max_val = np.max(points_inside_cell_1) if len(points_inside_cell_1) else 1
        normalized_points = [i / max_val for i in points_inside_cell_1]

        # テキストで統計量を表示
        plt.text(
            0.5,
            0.20,
            f"Median: {np.median(points_inside_cell_1):.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.text(
            0.5,
            0.15,
            f"Mean: {np.mean(points_inside_cell_1):.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.text(
            0.5,
            0.10,
            f"Normalized median: {np.median(normalized_points):.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.text(
            0.5,
            0.05,
            f"Normalized mean: {np.mean(normalized_points):.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )

        # 多項式近似のための x 軸生成: シフト後の範囲で
        x_for_fit = np.linspace(min_u1_shifted, max_u1_shifted, 1000)
        # シフト後の U を渡してフィッティング
        theta = await AsyncChores.poly_fit(U_shifted, degree=degree)
        y_for_fit = np.polyval(theta, x_for_fit)

        # 多項式近似曲線をプロット
        plt.plot(x_for_fit, y_for_fit, color="red", label="Poly fit")

        # 輪郭を lime 色で描画 (シフト済み)
        plt.scatter(
            u1_contour_shifted, u2_contour_shifted, color="lime", s=20, label="Contour"
        )

        plt.tick_params(direction="in")
        plt.grid(True)
        plt.legend()

        # 出力用バッファ
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        return buf

    @staticmethod
    async def find_path_return_list(
        image_fluo_raw: bytes, contour_raw: bytes, degree: int
    ) -> list[float]:
        """
        find_path() と同様のロジックでパスの (u1, G) リストを返す。
        """
        image_fluo = cv2.imdecode(
            np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR
        )
        image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(image_fluo_gray)
        unpickled_contour = pickle.loads(contour_raw)
        cv2.fillPoly(mask, [unpickled_contour], 255)

        coords_inside_cell_1 = np.column_stack(np.where(mask))
        points_inside_cell_1 = image_fluo_gray[
            coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
        ]

        X = np.array(
            [
                [i[1] for i in coords_inside_cell_1],
                [i[0] for i in coords_inside_cell_1],
            ]
        )

        (
            u1,
            u2,
            u1_contour,
            u2_contour,
            min_u1,
            max_u1,
            u1_c,
            u2_c,
            U,
            contour_U,
        ) = SyncChores.basis_conversion(
            [list(i[0]) for i in unpickled_contour],
            X,
            image_fluo.shape[0] / 2,
            image_fluo.shape[1] / 2,
            coords_inside_cell_1,
        )

        theta = await AsyncChores.poly_fit(U, degree=degree)
        raw_points: list[Point] = []
        for i, j, p in zip(u1, u2, points_inside_cell_1):
            min_distance, min_point = await AsyncChores.find_minimum_distance_and_point(
                theta, i, j
            )
            raw_points.append(Point(min_point[0], p))
        raw_points.sort()

        split_num: int = 35
        delta_L: float = (max(u1) - min(u1)) / split_num

        first_point: Point = raw_points[0]
        last_point: Point = raw_points[-1]
        path: list[Point] = [first_point]
        for i in range(1, int(split_num)):
            x_0 = min(u1) + i * delta_L
            x_1 = min(u1) + (i + 1) * delta_L
            points = [p for p in raw_points if x_0 <= p.u1 <= x_1]
            if len(points) == 0:
                continue
            point = max(points, key=lambda x: x.G)
            path.append(point)
        path.append(last_point)

        return [(p.u1, p.G) for p in path]

    @staticmethod
    async def box_plot(
        values: list[float],
        target_val: float,
        y_label: str,
        cell_id: str,
        label: str,
    ) -> io.BytesIO:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            buf = await loop.run_in_executor(
                pool, SyncChores.box_plot, values, target_val, y_label, cell_id, label
            )
        return buf

    @staticmethod
    async def histogram(
        values: list[float],
        y_label: str,
        cell_id: str,
        label: str,
        bin_scale: Literal["normalized", "default"] = "default",
    ) -> io.BytesIO:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            buf = await loop.run_in_executor(
                pool,
                SyncChores.create_histogram,
                values,
                256,
                y_label,
                cell_id,
                label,
                bin_scale,
            )
        return buf

    @staticmethod
    async def heatmap_path(path: list[float]) -> io.BytesIO:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            buf = await loop.run_in_executor(pool, SyncChores.heatmap_path, path)
        return buf

    @staticmethod
    async def plot_paths(paths: list[float]) -> io.BytesIO:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            buf = await loop.run_in_executor(pool, SyncChores.plot_paths, paths)
        return buf

    @staticmethod
    async def heatmap_all_abs(u1s: list[float], Gs: list[float]) -> io.BytesIO:
        buf = await SyncChores.heatmap_all_abs(u1s, Gs)
        return buf


class CellCrudBase:
    def __init__(self, db_name: str) -> None:
        self.db_name: str = db_name

    async def delete_database(self) -> None:
        await aiofiles.os.remove(f"databases/{self.db_name}")

    @staticmethod
    async def parse_image(
        data: bytes,
        contour: bytes | None = None,
        scale_bar: bool = False,
        brightness_factor: float = 1.0,
    ) -> StreamingResponse:
        img = await AsyncChores.async_imdecode(data)
        if contour:
            img = await AsyncChores.draw_contour(img, contour)
        if brightness_factor != 1.0:
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        if scale_bar:
            img = await AsyncChores.draw_scale_bar_with_centered_text(img)
        _, buffer = await AsyncChores.async_cv2_imencode(img)
        buffer_io = io.BytesIO(buffer)
        return StreamingResponse(buffer_io, media_type="image/png")

    async def parse_image_to_bytes(
        data: bytes,
        contour: bytes | None = None,
        scale_bar: bool = False,
        brightness_factor: float = 1.0,
        thickness: int = 2,
    ) -> bytes:
        img = await AsyncChores.async_imdecode(data)
        if contour:
            img = await AsyncChores.draw_contour(img, contour, thickness=thickness)
        if brightness_factor != 1.0:
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        if scale_bar:
            img = await AsyncChores.draw_scale_bar_with_centered_text(img)
        success, buffer = await AsyncChores.async_cv2_imencode(img)
        if success:
            return buffer.tobytes()
        else:
            raise ValueError("Failed to encode image")

    async def read_cell_ids(self, label: str | None = None) -> list[CellId]:
        """
        DB からセルID一覧を取得し、F{frame}C{cell} 形式でソートして返す。
        """

        def sort_key(cell_id: str) -> tuple[int, int]:
            match = re.match(r"F(\d+)C(\d+)", cell_id)
            if match:
                frame, cell = match.groups()
                return int(frame), int(cell)
            return float("inf"), float("inf")

        stmt = select(Cell)
        if label:
            stmt = stmt.where(Cell.manual_label == label)
        async for session in get_session(dbname=self.db_name):
            result = await session.execute(stmt)
            cells: list[Cell] = result.scalars().all()
        await session.close()
        sorted_cells = sorted(cells, key=lambda cell: sort_key(cell.cell_id))
        return [CellId(cell_id=cell.cell_id) for cell in sorted_cells]

    async def read_cell_label(self, cell_id: str) -> str:
        stmt = select(Cell).where(Cell.cell_id == cell_id)
        async for session in get_session(dbname=self.db_name):
            result = await session.execute(stmt)
            cell: Cell = result.scalars().first()
        await session.close()
        return cell.manual_label if not cell.manual_label == "N/A" else "1000"

    async def update_label(self, cell_id: str, label: str) -> None:
        stmt = update(Cell).where(Cell.cell_id == cell_id).values(manual_label=label)
        async for session in get_session(dbname=self.db_name):
            await session.execute(stmt)
            await session.commit()
        await session.close()

    async def read_cell_ids_count(self, label: str | None = None) -> int:
        return len(await self.read_cell_ids(label))

    async def read_cell(self, cell_id: str) -> Cell:
        stmt = select(Cell).where(Cell.cell_id == cell_id)
        async for session in get_session(dbname=self.db_name):
            result = await session.execute(stmt)
            cell: Cell = result.scalars().first()
        await session.close()
        if cell is None:
            raise CellNotFoundError(cell_id, "Cell with given ID does not exist")
        return cell

    async def update_all_cells_metadata(self, metadata: str) -> None:
        stmt = update(Cell).values(label_experiment=metadata)
        async for session in get_session(dbname=self.db_name):
            await session.execute(stmt)
            await session.commit()
        await session.close()

    async def get_metadata(self) -> str:
        stmt = select(Cell).limit(1)
        async for session in get_session(dbname=self.db_name):
            result = await session.execute(stmt)
            cell: Cell = result.scalars().first()
        await session.close()
        return cell.label_experiment

    async def get_cell_ph(
        self, cell_id: str, draw_contour: bool = False, draw_scale_bar: bool = False
    ) -> StreamingResponse:
        cell = await self.read_cell(cell_id)
        if draw_contour:
            return await self.parse_image(
                data=cell.img_ph, contour=cell.contour, scale_bar=draw_scale_bar
            )
        return await self.parse_image(data=cell.img_ph, scale_bar=draw_scale_bar)

    async def get_cell_fluo(
        self,
        cell_id: str,
        draw_contour: bool = False,
        draw_scale_bar: bool = False,
        brightness_factor: float = 1.0,
    ) -> StreamingResponse:
        cell = await self.read_cell(cell_id)
        if draw_contour:
            return await self.parse_image(
                data=cell.img_fluo1,
                contour=cell.contour,
                scale_bar=draw_scale_bar,
                brightness_factor=brightness_factor,
            )
        return await self.parse_image(
            data=cell.img_fluo1,
            scale_bar=draw_scale_bar,
            brightness_factor=brightness_factor,
        )

    async def get_cell_fluo2(
        self,
        cell_id: str,
        draw_contour: bool = False,
        draw_scale_bar: bool = False,
        brightness_factor: float = 1.0,
    ) -> StreamingResponse:
        cell = await self.read_cell(cell_id)
        if draw_contour:
            return await self.parse_image(
                data=cell.img_fluo2,
                contour=cell.contour,
                scale_bar=draw_scale_bar,
                brightness_factor=brightness_factor,
            )
        return await self.parse_image(
            data=cell.img_fluo2,
            scale_bar=draw_scale_bar,
            brightness_factor=brightness_factor,
        )

    async def get_cell_contour(self, cell_id: str) -> list[list[float]]:
        cell = await self.read_cell(cell_id)
        return await AsyncChores.async_pickle_loads(cell.contour)

    async def get_cell_contour_plot_data(self, cell_id: str) -> dict:
        cell = await self.read_cell(cell_id)
        return await AsyncChores.get_contour(cell.contour)

    async def morpho_analysis(self, cell_id: str, polyfit_degree: int) -> CellMorhology:
        cell = await self.read_cell(cell_id)
        return await AsyncChores.morpho_analysis(
            cell.img_ph, cell.img_fluo1, cell.contour, polyfit_degree
        )

    async def replot(
        self, cell_id: str, degree: int, channel: int = 1
    ) -> StreamingResponse:
        cell = await self.read_cell(cell_id)
        img = cell.img_fluo2 if channel == 2 else cell.img_fluo1
        return StreamingResponse(
            await AsyncChores.replot(img, cell.contour, degree),
            media_type="image/png",
        )

    async def find_path(
        self, cell_id: str, degree: int, channel: int = 1
    ) -> StreamingResponse:
        cell = await self.read_cell(cell_id)
        img = cell.img_fluo2 if channel == 2 else cell.img_fluo1
        return StreamingResponse(
            await AsyncChores.find_path(img, cell.contour, degree),
            media_type="image/png",
        )

    async def get_all_mean_normalized_fluo_intensities(
        self, cell_id: str, y_label: str, label: str | None = None
    ) -> StreamingResponse:
        cell_ids = await self.read_cell_ids(label)
        cells = await asyncio.gather(
            *(self.read_cell(cell.cell_id) for cell in cell_ids)
        )
        target_cell = await self.read_cell(cell_id=cell_id)
        target_val = await AsyncChores.calc_mean_normalized_fluo_intensity_incide_cell(
            target_cell.img_fluo1, target_cell.contour
        )
        mean_intensities = await asyncio.gather(
            *(
                AsyncChores.calc_mean_normalized_fluo_intensity_incide_cell(
                    cell.img_fluo1, cell.contour
                )
                for cell in cells
            )
        )
        ret = await AsyncChores.box_plot(
            mean_intensities,
            target_val=target_val,
            y_label=y_label,
            cell_id=cell_id,
            label=None,
        )
        return StreamingResponse(ret, media_type="image/png")

    async def get_all_median_normalized_fluo_intensities(
        self, cell_id: str, y_label: str, label: str | None = None
    ) -> StreamingResponse:
        cell_ids = await self.read_cell_ids(label)
        cells = await asyncio.gather(
            *(self.read_cell(cell.cell_id) for cell in cell_ids)
        )
        target_cell = await self.read_cell(cell_id=cell_id)
        target_val = (
            await AsyncChores.calc_median_normalized_fluo_intensity_inside_cell(
                target_cell.img_fluo1, target_cell.contour
            )
        )
        median_intensities = await asyncio.gather(
            *(
                AsyncChores.calc_median_normalized_fluo_intensity_inside_cell(
                    cell.img_fluo1, cell.contour
                )
                for cell in cells
            )
        )
        buf = await AsyncChores.box_plot(
            median_intensities,
            target_val=target_val,
            y_label=y_label,
            cell_id=cell_id,
            label=str(
                round(
                    len([i for i in median_intensities if i < 0.6])
                    / len(median_intensities),
                    4,
                )
            ),
        )
        return StreamingResponse(buf, media_type="image/png")

    async def get_all_variance_normalized_fluo_intensities(
        self, cell_id: str, y_label: str, label: str | None = None
    ) -> StreamingResponse:
        cell_ids = await self.read_cell_ids(label)
        cells = await asyncio.gather(
            *(self.read_cell(cell.cell_id) for cell in cell_ids)
        )
        target_cell = await self.read_cell(cell_id=cell_id)
        target_val = (
            await AsyncChores.calc_variance_normalized_fluo_intensity_inside_cell(
                target_cell.img_fluo1, target_cell.contour
            )
        )
        variance_intensities = await asyncio.gather(
            *(
                AsyncChores.calc_variance_normalized_fluo_intensity_inside_cell(
                    cell.img_fluo1, cell.contour
                )
                for cell in cells
            )
        )
        ret = await AsyncChores.box_plot(
            variance_intensities,
            target_val=target_val,
            y_label=y_label,
            cell_id=cell_id,
            label=None,
        )
        return StreamingResponse(ret, media_type="image/png")

    async def extract_intensity_and_create_histogram(
        self,
        cell_id: str,
        label: str,
        channel: int = 1,
    ) -> io.BytesIO:
        cell = await self.read_cell(cell_id)
        img = cell.img_fluo2 if channel == 2 else cell.img_fluo1
        intensity_values = await AsyncChores.get_points_inside_cell(
            img, cell.contour
        )
        ret = await AsyncChores.histogram(
            values=intensity_values,
            y_label="count",
            cell_id=cell_id,
            label=label,
            bin_scale="default",
        )
        return StreamingResponse(ret, media_type="image/png")

    async def extract_normalized_intensity_and_create_histogram(
        self,
        cell_id: str,
        label: str,
        channel: int = 1,
    ) -> io.BytesIO:
        cell = await self.read_cell(cell_id)
        img = cell.img_fluo2 if channel == 2 else cell.img_fluo1
        intensity_values = await AsyncChores.get_points_inside_cell(
            img, cell.contour
        )
        max_val = np.max(intensity_values) if len(intensity_values) else 1
        normalized_values = [i / max_val for i in intensity_values]
        ret = await AsyncChores.histogram(
            values=normalized_values,
            y_label="count",
            cell_id=cell_id,
            label=label,
            bin_scale="normalized",
        )
        return StreamingResponse(ret, media_type="image/png")

    async def extract_normalized_intensities_raw(
        self, cell_id: str, channel: int = 1
    ) -> list[float]:
        cell = await self.read_cell(cell_id)
        img = cell.img_fluo2 if channel == 2 else cell.img_fluo1
        intensity_values = await AsyncChores.get_points_inside_cell(
            img, cell.contour
        )
        max_val = np.max(intensity_values) if len(intensity_values) else 1
        return [i / max_val for i in intensity_values]

    async def get_all_mean_normalized_fluo_intensities_csv(
        self, label: str | None = None
    ) -> StreamingResponse:
        cell_ids = await self.read_cell_ids(label)
        cells = await asyncio.gather(
            *(self.read_cell(cell.cell_id) for cell in cell_ids)
        )
        mean_intensities = await asyncio.gather(
            *(
                AsyncChores.calc_mean_normalized_fluo_intensity_incide_cell(
                    cell.img_fluo1, cell.contour
                )
                for cell in cells
            )
        )
        df = pd.DataFrame(
            mean_intensities,
            columns=[
                f"Mean normalized fluorescence intensity {self.db_name} cells with label {label}"
            ],
        )
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def get_all_median_normalized_fluo_intensities_csv(
        self, label: str | None = None
    ) -> StreamingResponse:
        cell_ids = await self.read_cell_ids(label)
        cells = await asyncio.gather(
            *(self.read_cell(cell.cell_id) for cell in cell_ids)
        )
        median_intensities = await asyncio.gather(
            *(
                AsyncChores.calc_median_normalized_fluo_intensity_inside_cell(
                    cell.img_fluo1, cell.contour
                )
                for cell in cells
            )
        )
        df = pd.DataFrame(
            median_intensities,
            columns=[
                f"Median normalized fluorescence intensity {self.db_name} cells with label {label}"
            ],
        )
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def get_all_variance_normalized_fluo_intensities_csv(
        self, label: str | None = None
    ) -> StreamingResponse:
        cell_ids = await self.read_cell_ids(label)
        cells = await asyncio.gather(
            *(self.read_cell(cell.cell_id) for cell in cell_ids)
        )
        variance_intensities = await asyncio.gather(
            *(
                AsyncChores.calc_variance_normalized_fluo_intensity_inside_cell(
                    cell.img_fluo1, cell.contour
                )
                for cell in cells
            )
        )
        df = pd.DataFrame(
            variance_intensities,
            columns=[
                f"Variance normalized fluorescence intensity {self.db_name} cells with label {label}"
            ],
        )
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def heatmap_all_abs(self, label: str | None = None) -> StreamingResponse:
        cell_ids = await self.read_cell_ids(label)
        cells = await asyncio.gather(
            *(self.read_cell(cell.cell_id) for cell in cell_ids)
        )
        u1s = []
        Gs = []
        for cell in cells:
            u1, G = await AsyncChores.find_path_return_list(
                cell.img_fluo1, cell.contour, 4
            )
            u1s.extend(u1)
            Gs.extend(G)
        buf = await AsyncChores.heatmap_all_abs(u1s, Gs)
        return StreamingResponse(buf, media_type="image/png")

    async def heatmap_path(self, cell_id: str, degree: int) -> StreamingResponse:
        cell = await self.read_cell(cell_id)
        path = await AsyncChores.find_path_return_list(
            cell.img_fluo1, cell.contour, degree
        )
        buf = await AsyncChores.heatmap_path(path)
        return StreamingResponse(buf, media_type="image/png")

    async def get_peak_path_csv(
        self, cell_id: str, degree: int = 3
    ) -> StreamingResponse:
        cell = await self.read_cell(cell_id)
        path: list[float] = await AsyncChores.find_path_return_list(
            cell.img_fluo1, cell.contour, degree
        )
        df = pd.DataFrame(path, columns=["u1", "G"])
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def get_peak_paths_csv(
        self, degree: int = 4, label: str = "1", channel: int = 1
    ) -> StreamingResponse:
        # 1. 指定ラベルの cell_ids を取得
        cell_ids = await self.read_cell_ids(label=label)
        print(f"cell_ids: {cell_ids}")
        # 2. DB から対応する Cell レコードを並列取得
        cells = await asyncio.gather(
            *(self.read_cell(cell.cell_id) for cell in cell_ids)
        )

        valid_pairs = []
        for cell in cells:
            img_blob = cell.img_fluo2 if channel == 2 else cell.img_fluo1
            if img_blob is not None:
                valid_pairs.append((img_blob, cell.contour))

        if not valid_pairs:
            return StreamingResponse(io.BytesIO(), media_type="text/csv")

        # 3. find_path_return_list も並列で実行
        #    （各セルの img_fluo1, contour をもとにピークパスを計算）
        paths_list = await asyncio.gather(
            *(
                AsyncChores.find_path_return_list(img, contour, degree)
                for img, contour in valid_pairs
            )
        )

        # 4. 複数セル分の (u1, G) を一つのリストへ
        combined_paths = []
        for path in paths_list:
            u1_values = [p[0] for p in path]
            G_values = [p[1] for p in path]
            combined_paths.append(u1_values)
            combined_paths.append(G_values)

        # 5. CSV に書き出し
        df = pd.DataFrame(combined_paths)
        buf = io.BytesIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)

        csv_content = buf.getvalue().decode("utf-8")

        # もしローカルにファイル保存したい場合は aiofiles で非同期書き込み
        async with aiofiles.open(f"results/peak_paths_{self.db_name}.csv", "w") as f:
            await f.write(csv_content)

        buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv")

    async def plot_peak_paths(
        self, degree: int = 3, label: str = 1
    ) -> StreamingResponse:
        cell_ids = await self.read_cell_ids(label=label)
        cells = [await self.read_cell(cell.cell_id) for cell in cell_ids]
        paths = await asyncio.gather(
            *(
                AsyncChores.find_path_return_list(cell.img_fluo1, cell.contour, degree)
                for cell in cells
            )
        )
        paths_G = [list(map(lambda x: x[1], path)) for path in paths]
        normalized_paths_G = [
            [round(i / max(path), 3) for i in path]
            for path in paths_G
            if max(path) != 0
        ]
        buf = await AsyncChores.plot_paths(normalized_paths_G)
        return StreamingResponse(buf, media_type="image/png")

    async def rename_database_to_completed(self):
        if "-uploaded" not in self.db_name:
            return False
        dbname_cleaned = self.db_name.split("/")[-1]
        dbname_cleaned = "".join(dbname_cleaned.split(".")[:-1]) + ".db"
        os.rename(
            f"databases/{dbname_cleaned}",
            f"databases/{dbname_cleaned.replace('-uploaded.db','')}-completed.db",
        )
        date = datetime.now().strftime("%Y-%m-%d")
        try:
            await DropboxCrud().upload_file(
                file_path=f"databases/{dbname_cleaned.replace('-uploaded.db','')}-completed.db",
                file_name=f"databases/{dbname_cleaned.replace('-uploaded.db','')}-completed-{date}.db",
            )
        except Exception as e:
            print(f"Failed to upload to Dropbox: {e}")
        return True

    async def check_if_database_updated(self):
        length_all = len(await self.read_cell_ids())
        length_na = len(await self.read_cell_ids("N/A"))
        return length_all != length_na

    async def has_fluo2(self) -> bool:
        async for session in get_session(dbname=self.db_name):
            stmt = select(Cell).where(Cell.img_fluo2.is_not(None))
            result = await session.execute(stmt)
            row = result.scalars().first()
            await session.close()
            return row is not None

    async def get_cell_images_combined(
        self,
        label: str = "1",
        image_size: int = 200,
        mode: Literal[
            "fluo",
            "ph",
            "ph_contour",
            "fluo_contour",
            "fluo2",
            "fluo2_contour",
            "replot",
        ] = "fluo",
    ):
        async def combine_images_from_folder(
            folder_path, total_rows, total_cols, image_size
        ):
            image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
            num_images = len(image_files)
            result_image = np.zeros(
                (total_rows * image_size, total_cols * image_size, 3), dtype=np.uint8
            )

            for i in range(total_rows):
                for j in range(total_cols):
                    image_index = i * total_cols + j
                    if image_index < num_images:
                        image_path = os.path.join(folder_path, image_files[image_index])
                        img = cv2.imread(image_path)
                        img = cv2.resize(img, (image_size, image_size))
                        result_image[
                            i * image_size : (i + 1) * image_size,
                            j * image_size : (j + 1) * image_size,
                        ] = img

            _, buffer = cv2.imencode(".png", result_image)
            buf = io.BytesIO(buffer)
            return buf

        tmp_folder = "TempImages"
        try:
            os.mkdir(tmp_folder)
        except FileExistsError:
            pass

        n = await self.read_cell_ids_count(label)
        total_rows: int = int(np.sqrt(n)) + 1
        total_cols: int = n // total_rows + 1

        try:
            cell_ids = await self.read_cell_ids(label)
            cells = [await self.read_cell(cell.cell_id) for cell in cell_ids]

            for cell in cells:
                async with aiofiles.open(f"{tmp_folder}/{cell.cell_id}.png", "wb") as f:
                    if mode == "fluo":
                        await f.write(cell.img_fluo1)
                    elif mode == "ph":
                        await f.write(cell.img_ph)
                    elif mode == "ph_contour":
                        await f.write(
                            await CellCrudBase.parse_image_to_bytes(
                                cell.img_ph, cell.contour, scale_bar=False, thickness=3
                            )
                        )
                    elif mode == "fluo_contour":
                        await f.write(
                            await CellCrudBase.parse_image_to_bytes(
                                cell.img_fluo1,
                                cell.contour,
                                scale_bar=False,
                                thickness=3,
                            )
                        )
                    elif mode == "fluo2":
                        if cell.img_fluo2 is not None:
                            await f.write(cell.img_fluo2)
                        else:
                            continue
                    elif mode == "fluo2_contour":
                        if cell.img_fluo2 is not None:
                            await f.write(
                                await CellCrudBase.parse_image_to_bytes(
                                    cell.img_fluo2,
                                    cell.contour,
                                    scale_bar=False,
                                    thickness=3,
                                )
                            )
                        else:
                            continue
                    elif mode == "replot":
                        buf = await AsyncChores.replot(
                            cell.img_fluo1,
                            cell.contour,
                            3,
                        )
                        await f.write(buf.getvalue())
                    
            return StreamingResponse(
                await combine_images_from_folder(
                    tmp_folder, total_rows, total_cols, image_size
                ),
                media_type="image/png",
            )

        finally:
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)

    async def get_cloud_points(
        self,
        cell_id: str,
        angle: float = 270,
        mode: Literal["fluo", "ph"] = "fluo",
        channel: int = 1,
    ) -> io.BytesIO:
        cell = await self.read_cell(cell_id)
        if mode == "fluo":
            img_blob = cell.img_fluo2 if channel == 2 else cell.img_fluo1
            image = np.frombuffer(img_blob, dtype=np.uint8)
        else:
            image = np.frombuffer(cell.img_ph, dtype=np.uint8)

        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape

        point_cloud = []
        for y in range(height):
            for x in range(width):
                z = image[y, x]
                if z > 0:
                    point_cloud.append([x, y, z])

        point_cloud = np.array(point_cloud)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            cmap="jet",
            c=point_cloud[:, 2],
            marker="o",
            s=2,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("G")
        ax.set_ylim(height, 0)
        ax.set_xlim(0, width)
        ax.view_init(elev=30, azim=angle)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    async def get_contour_canny_draw(
        self, cell_id: str, canny_thresh2: int = 100
    ) -> list[list[float]]:
        """
        Read an image from the database by cell_id, apply Canny edge detection using the given
        thresholds, and return a single contour (as a list of [x, y] coordinates).
        The chosen contour is the one whose centroid is closest to the image center.

        Parameters:
        - cell_id: str
            ID of the cell to retrieve from the database.
        - canny_thresh1: int
            First threshold for the hysteresis procedure in the Canny edge detector.
        - canny_thresh2: int
            Second threshold for the hysteresis procedure in the Canny edge detector.

        Returns:
        - best_contour: list[list[float]]
            The coordinates of the contour with the centroid closest to the image center,
            formatted as [[x0, y0], [x1, y1], ...].
        """

        # セル情報（画像バイナリなど）を取得
        cell = await CellCrudBase(self.db_name).read_cell(cell_id)

        # 画像バイナリから OpenCV 画像オブジェクトを作成
        img = await AsyncChores.async_imdecode(cell.img_ph)

        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Cannyエッジ検出
        # ２値化を行う
        ret, thresh = cv2.threshold(gray, canny_thresh2, 255, cv2.THRESH_BINARY)
        img_canny = cv2.Canny(thresh, 0, 130)
        contours, hierarchy = cv2.findContours(
            img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            raise ValueError("No contours found")

        # 画像の中心座標
        image_center = (img.shape[1] // 2, img.shape[0] // 2)

        min_distance = float("inf")
        best_contour = None

        # すべての輪郭について重心を求め、画像の中心に最も近いものを選ぶ
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distance = np.sqrt(
                    (cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    best_contour = contour

        if best_contour is None:
            raise ValueError("No valid contours found")

        return best_contour.squeeze().tolist()

    async def elastic_contour(self, cell_id: str, delta: int = 0) -> list[list[float]]:
        """Expand or shrink contour by |delta| pixels."""
        cell = await self.read_cell(cell_id)
        contour_np = np.array(await AsyncChores.async_pickle_loads(cell.contour), dtype=np.int32)
        contour_np = contour_np.reshape(-1, 1, 2)
        img = await AsyncChores.async_imdecode(cell.img_ph)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour_np], -1, 255, -1)
        if delta > 0:
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=delta)
        elif delta < 0:
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=-delta)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return contour_np.squeeze().tolist()
        M = cv2.moments(contour_np)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        min_distance = float("inf")
        best = contours[0]
        for cnt in contours:
            M2 = cv2.moments(cnt)
            if M2["m00"] != 0:
                cx2 = int(M2["m10"] / M2["m00"])
                cy2 = int(M2["m01"] / M2["m00"])
                dist = np.sqrt((cx - cx2) ** 2 + (cy - cy2) ** 2)
                if dist < min_distance:
                    min_distance = dist
                    best = cnt
        return best.squeeze().tolist()

    async def update_contour(
        self, cell_id: str, new_contour: list[list[float]]
    ) -> None:
        """
        予測した輪郭(new_contour)を pickle 化して DB を更新するメソッド。
        従来の get_contour() が想定する形式 (N,1,2) をそのまま pickle 化して保存する。
        """
        # まず Nx1x2 の配列に変換
        contour_array = np.array(new_contour, dtype=np.int32).reshape(-1, 1, 2)

        # ここで “リストで包む” をやめる
        pickled_contour = pickle.dumps(contour_array)

        stmt = (
            update(Cell).where(Cell.cell_id == cell_id).values(contour=pickled_contour)
        )

        async for session in get_session(dbname=self.db_name):
            await session.execute(stmt)
            await session.commit()
        await session.close()
