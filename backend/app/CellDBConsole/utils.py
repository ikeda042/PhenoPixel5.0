from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import minimize
from __future__ import annotations
import numpy as np
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
import cv2
import pickle

# from database_parser import database_parser, Cell
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.integrate import quad
from dataclasses import dataclass
from tqdm import tqdm
import os


class SyncChores:
    @classmethod
    def flip_image_if_needed(cls: Map64, image: np.ndarray) -> np.ndarray:
        # 画像がカラーの場合、グレースケールに変換
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape
        left_half = image[:, : w // 2]
        right_half = image[:, w // 2 :]

        left_brightness = np.mean(left_half)
        right_brightness = np.mean(right_half)

        if right_brightness > left_brightness:
            # 右の輝度が高い場合は左右反転
            image = cv2.flip(image, 1)
        return image

    @classmethod
    def find_minimum_distance_and_point(
        cls, coefficients: float, x_Q: float, y_Q: float
    ) -> tuple[float, tuple[float, float]]:
        # 関数の定義
        def f_x(x):
            return sum(
                coefficient * x**i for i, coefficient in enumerate(coefficients[::-1])
            )

        # 点Qから関数上の点までの距離 D の定義
        def distance(x):
            return np.sqrt((x - x_Q) ** 2 + (f_x(x) - y_Q) ** 2)

        # scipyのminimize関数を使用して最短距離を見つける
        # 初期値は0とし、精度は低く設定して計算速度を向上させる
        result = minimize(
            distance, 0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-2}
        )

        # 最短距離とその時の関数上の点
        x_min = result.x[0]
        min_distance = distance(x_min)
        min_point = (x_min, f_x(x_min))

        return min_distance, min_point

    @classmethod
    def poly_fit(cls: Map64, U: list[list[float]], degree: int = 1) -> list[float]:
        u1_values = np.array([i[1] for i in U])
        f_values = np.array([i[0] for i in U])
        W = np.vander(u1_values, degree + 1)
        return inv(W.T @ W) @ W.T @ f_values

    @classmethod
    def basis_conversion(
        cls: Map64,
        contour: list[list[int]],
        X: np.ndarray,
        center_x: float,
        center_y: float,
        coordinates_incide_cell: list[list[int]],
    ) -> list[list[float]]:
        Sigma = np.cov(X)
        eigenvalues, eigenvectors = eig(Sigma)
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

    @classmethod
    def replot(
        cls: Map64,
        image_fluo_raw: bytes,
        contour_raw: bytes,
        degree: int,
    ) -> None:

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
        ) = cls.basis_conversion(
            [list(i[0]) for i in unpickled_contour],
            X,
            image_fluo.shape[0] / 2,
            image_fluo.shape[1] / 2,
            coords_inside_cell_1,
        )

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(u1, u2, s=5)
        plt.scatter(u1_c, u2_c, color="red", s=100)
        plt.axis("equal")
        margin_width = 50
        margin_height = 50
        plt.scatter(
            [i[1] for i in U],
            [i[0] for i in U],
            points_inside_cell_1,
            c=points_inside_cell_1,
            cmap="inferno",
            marker="o",
        )
        plt.xlim([min_u1 - margin_width, max_u1 + margin_width])
        plt.ylim([min(u2) - margin_height, max(u2) + margin_height])

        x = np.linspace(min_u1, max_u1, 1000)
        theta = cls.poly_fit(U, degree=degree)
        y = np.polyval(theta, x)
        plt.plot(x, y, color="red")
        plt.scatter(u1_contour, u2_contour, color="lime", s=3)
        plt.tick_params(direction="in")
        plt.grid(True)
        plt.savefig("experimental/DotPatternMap/images/contour.png")
        plt.close(fig)

    @classmethod
    def extract_map(
        cls,
        image_fluo_raw: bytes,
        contour_raw: bytes,
        degree: int,
        cell_id: str = "default_cell_id",
    ):
        def calculate_arc_length(theta, x1, x_target):
            # 多項式の導関数を取得
            poly_derivative = np.polyder(theta)

            # 弧長積分のための関数
            def arc_length_function(x):
                return np.sqrt(1 + (np.polyval(poly_derivative, x)) ** 2)

            # x1からx_targetまでの弧長を計算
            arc_length, _ = quad(arc_length_function, x1, x_target)

            return arc_length

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

        # write out the image fluo with contour
        cv2.polylines(
            image_fluo,
            [unpickled_contour],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.imwrite(
            f"experimental/DotPatternMap/images/fluo_raw/{cell_id}.png",
            image_fluo,
        )

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
        ) = cls.basis_conversion(
            [list(i[0]) for i in unpickled_contour],
            X,
            image_fluo.shape[0] / 2,
            image_fluo.shape[1] / 2,
            coords_inside_cell_1,
        )

        theta = cls.poly_fit(U, degree=degree)
        raw_points: list[cls.Point] = []
        for i, j, p in zip(u1, u2, points_inside_cell_1):
            min_distance, min_point = cls.find_minimum_distance_and_point(theta, i, j)
            sign = 1 if j > min_point[1] else -1
            raw_points.append(
                cls.Point(
                    calculate_arc_length(theta, min(u1), min_point[0]),
                    min_point[1],
                    i,
                    j,
                    min_distance,
                    p,
                    sign,
                )
            )
        raw_points.sort()

        fig = plt.figure(figsize=(6, 6))
        plt.axis("equal")

        ps = np.array([i.p for i in raw_points])
        dists = np.array([i.dist * i.sign for i in raw_points])
        gs = np.array([i.G for i in raw_points])
        gs_norm = (
            (gs - np.min(gs)) / (np.max(gs) - np.min(gs)) * 255
            if np.max(gs) > np.min(gs)
            else np.zeros(len(gs))
        )

        min_p, max_p = np.min(ps), np.max(ps)
        min_dist, max_dist = np.min(dists), np.max(dists)

        plt.scatter(ps, dists, s=80, c=gs_norm, cmap="jet")
        plt.xlabel(r"$L(u_{1_i}^\star)$ (px)")
        plt.ylabel(r"$\text{min\_dist}$ (px)")
        # 外接矩形の描画
        plt.plot([min_p, max_p], [min_dist, min_dist], color="red")
        plt.plot([min_p, max_p], [max_dist, max_dist], color="red")
        plt.plot([min_p, min_p], [min_dist, max_dist], color="red")
        plt.plot([max_p, max_p], [min_dist, max_dist], color="red")

        fig.savefig(
            f"experimental/DotPatternMap/images/points_box/{cell_id}.png", dpi=300
        )
        fig.savefig(f"experimental/DotPatternMap/images/points_box.png", dpi=300)
        plt.close(fig)
        plt.clf()

        # psとdistsを曲座標変換
        r = [np.sqrt(p**2 + dist**2) for p, dist in zip(ps, dists)]
        theta = [np.arctan2(dist, p) for p, dist in zip(ps, dists)]

        # rとthetaを正規化(0-1)
        r = (r - min(r)) / (max(r) - min(r))
        theta = (theta - min(theta)) / (max(theta) - min(theta))

        # プロット
        fig = plt.figure(figsize=(6, 6))
        plt.axis("equal")
        plt.scatter(theta, gs, s=80, c=gs_norm, cmap="jet")
        plt.ylabel(r"$r_i$ (px)")
        plt.xlabel(r"$\theta_i$ (rad)")
        fig.savefig(f"experimental/DotPatternMap/images/polar/{cell_id}.png", dpi=300)
        plt.close(fig)
        plt.clf()

        # 画像サイズを元の範囲に厳密に設定
        scale_factor = 1
        scaled_width = int((max_p - min_p) * scale_factor)
        scaled_height = int((max_dist - min_dist) * scale_factor)
        high_res_image = np.zeros((scaled_height, scaled_width), dtype=np.uint8)

        # 点群を描画
        for p, dist, G in zip(ps, dists, gs_norm):
            p_scaled = int((p - min_p) * scale_factor)
            dist_scaled = int((dist - min_dist) * scale_factor)
            cv2.circle(high_res_image, (p_scaled, dist_scaled), 1, int(G), -1)
        # リサイズ前の画像を保存
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_raw/{cell_id}.png",
            high_res_image,
        )
        # resize image to 64x64
        high_res_image = cv2.resize(
            high_res_image, (64, 64), interpolation=cv2.INTER_NEAREST
        )
        # 画像反転関数を適用
        high_res_image = cls.flip_image_if_needed(high_res_image)
        # 画像を保存
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64/{cell_id}.png",
            high_res_image,
        )
        high_res_image_colormap = cv2.applyColorMap(high_res_image, cv2.COLORMAP_JET)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_jet/{cell_id}.png",
            high_res_image_colormap,
        )
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_jet.png",
            high_res_image_colormap,
        )
        return cls.perform_pca_on_3d_point_cloud_and_save(
            high_res_image,
            f"experimental/DotPatternMap/images/pca_2d/{cell_id}.png",
            f"experimental/DotPatternMap/images/pca_1d/{cell_id}.png",
        )


class Map64:
    @dataclass
    class Point:
        def __init__(
            self,
            p: float,
            q: float,
            u1: float,
            u2: float,
            dist: float,
            G: float,
            sign: int,
        ) -> None:
            self.p: float = p
            self.q: float = q
            self.u1: float = u1
            self.u2: float = u2
            self.dist: float = dist
            self.G: float = G
            self.sign: int = sign

        def __gt__(self, other) -> bool:
            return self.u1 > other.u1

        def __lt__(self, other) -> bool:
            return self.u1 < other.u1

        def __repr__(self) -> str:
            return f"({self.u1},{self.G})"
