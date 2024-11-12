import numpy as np
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
import cv2
import pickle
from database_parser import database_parser, Cell
from scipy.optimize import minimize
from dataclasses import dataclass
from tqdm import tqdm
import os


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
            self.p = p
            self.q = q
            self.u1 = u1
            self.u2 = u2
            self.dist = dist
            self.G = G
            self.sign = sign

        def __gt__(self, other) -> bool:
            return self.u1 > other.u1

        def __lt__(self, other) -> bool:
            return self.u1 < other.u1

        def __repr__(self) -> str:
            return f"({self.u1},{self.G})"

    @classmethod
    def find_minimum_distance_and_point(cls, coefficients, x_Q, y_Q):
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
    def poly_fit(cls, U: list[list[float]], degree: int = 1) -> list[float]:
        u1_values = np.array([i[1] for i in U])
        f_values = np.array([i[0] for i in U])
        W = np.vander(u1_values, degree + 1)
        return inv(W.T @ W) @ W.T @ f_values

    @classmethod
    def basis_conversion(
        cls,
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
        cls,
        image_fluo_raw: bytes,
        contour_raw: bytes,
        degree: int,
    ):

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

    @classmethod
    def extract_map(
        cls,
        image_fluo_raw: bytes,
        contour_raw: bytes,
        degree: int,
        cell_id: str = "default_cell_id",
    ):
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

        theta = cls.poly_fit(U, degree=degree)
        raw_points: list[cls.Point] = []
        for i, j, p in zip(u1, u2, points_inside_cell_1):
            min_distance, min_point = cls.find_minimum_distance_and_point(theta, i, j)
            sign = 1 if j > min_point[1] else -1
            raw_points.append(
                cls.Point(min_point[0], min_point[1], i, j, min_distance, p, sign)
            )
        raw_points.sort()

        fig = plt.figure(figsize=(6, 6))
        plt.axis("equal")

        ps = [i.p for i in raw_points]
        qs = [i.q for i in raw_points]
        dists = [i.dist * i.sign for i in raw_points]
        gs = [i.G for i in raw_points]
        min_p, max_p = min(ps), max(ps)
        min_dist, max_dist = min(dists), max(dists)
        print(min_dist, max_dist)
        # gsを正規化する（最大を255に、最小を0にする）
        gs_norm = (
            (gs - min(gs)) / (max(gs) - min(gs)) * 255
            if max(gs) > min(gs)
            else [0] * len(gs)
        )

        plt.scatter(ps, dists, s=100, c=gs_norm, cmap="inferno")
        plt.xlabel("p")
        plt.ylabel("dist")
        # 外接矩形の描画
        plt.plot([min_p, max_p], [min_dist, min_dist], color="red")
        plt.plot([min_p, max_p], [max_dist, max_dist], color="red")
        plt.plot([min_p, min_p], [min_dist, max_dist], color="red")
        plt.plot([max_p, max_p], [min_dist, max_dist], color="red")

        fig.savefig(f"experimental/DotPatternMap/images/points_box/{cell_id}.png")
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
        # resize image to 28x28
        high_res_image = cv2.resize(
            high_res_image, (64, 64), interpolation=cv2.INTER_NEAREST
        )
        # 画像を保存
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64/{cell_id}.png",
            high_res_image,
        )

    @classmethod
    def combine_images(cls):
        map64_dir = "experimental/DotPatternMap/images/map64"
        points_box_dir = "experimental/DotPatternMap/images/points_box"

        map64_images = [
            cv2.imread(os.path.join(map64_dir, filename), cv2.IMREAD_GRAYSCALE)
            for filename in os.listdir(map64_dir)
            if cv2.imread(os.path.join(map64_dir, filename), cv2.IMREAD_GRAYSCALE)
            is not None
        ]
        points_box_images = [
            cv2.imread(os.path.join(points_box_dir, filename), cv2.IMREAD_COLOR)
            for filename in os.listdir(points_box_dir)
            if cv2.imread(os.path.join(points_box_dir, filename), cv2.IMREAD_COLOR)
            is not None
        ]

        brightness = [np.sum(img) for img in map64_images]

        sorted_indices = np.argsort(brightness)[::-1]
        map64_images = [map64_images[i] for i in sorted_indices]
        points_box_images = [points_box_images[i] for i in sorted_indices]

        def calculate_grid_size(n):
            row = int(np.sqrt(n))
            col = (n + row - 1) // row
            return row, col

        def combine_images_grid(images, image_size, channels=1):
            n = len(images)
            row, col = calculate_grid_size(n)
            combined_image = np.zeros(
                (image_size * row, image_size * col, channels), dtype=np.uint8
            )
            for i, img in enumerate(images):
                x = (i % col) * image_size
                y = (i // col) * image_size
                img = cv2.resize(img, (image_size, image_size))
                if img.ndim == 2:
                    img = img[:, :, np.newaxis]
                combined_image[y : y + image_size, x : x + image_size] = img
            return combined_image

        combined_map64_image = combine_images_grid(map64_images, 64, 1)
        cv2.imwrite(
            "experimental/DotPatternMap/images/combined_image.png", combined_map64_image
        )

        combined_points_box_image = combine_images_grid(points_box_images, 64, 3)
        cv2.imwrite(
            "experimental/DotPatternMap/images/combined_image_box.png",
            combined_points_box_image,
        )


cells: list[Cell] = database_parser("sk326Gen90min.db")
map64 = Map64()

# for cell in tqdm(cells):
#     image_fluo_raw = cell.img_fluo1
#     contour_raw = cell.contour
#     cell_id = cell.cell_id
# map64.extract_map(image_fluo_raw, contour_raw, 4, cell_id)
map64.combine_images()
