from __future__ import annotations
import numpy as np
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
import cv2
import pickle
from database_parser import database_parser, Cell
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.integrate import quad
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
    def perform_pca_on_3d_point_cloud_and_save(
        cls, high_res_image: np.ndarray, output_path_2d: str, output_path_1d: str
    ):
        # 画像のサイズを取得
        height, width = high_res_image.shape

        # 3次元空間の点群を作成 (x, y, intensity)
        points = []
        for y in range(height):
            for x in range(width):
                intensity = high_res_image[y, x]
                points.append([x, y, intensity])

        points = np.array(points)

        # データの中心化
        points_centered = points - np.mean(points, axis=0)

        # PCAを適用し、2次元に次元削減
        pca_2d = PCA(n_components=2)
        transformed_points_2d = pca_2d.fit_transform(points_centered)

        # 2次元PCA結果を可視化して保存
        fig_2d = plt.figure(figsize=(6, 6))
        plt.scatter(
            transformed_points_2d[:, 0],
            transformed_points_2d[:, 1],
            c=points[:, 2],
            cmap="inferno",
            marker="o",
        )
        plt.title("PCA - 2D representation of the 3D point cloud")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Original Intensity")
        fig_2d.savefig(output_path_2d, bbox_inches="tight")
        plt.close(fig_2d)

        # PCAを適用し、1次元に次元削減
        pca_1d = PCA(n_components=1)
        transformed_points_1d = pca_1d.fit_transform(points_centered)

        # 1次元PCA結果をボックスプロットで可視化して保存
        fig_1d = plt.figure(figsize=(6, 6))
        plt.boxplot(transformed_points_1d, vert=False)
        plt.title("PCA - 1D Boxplot of the 3D point cloud")
        plt.xlabel("Principal Component 1")
        fig_1d.savefig(output_path_1d, bbox_inches="tight")
        plt.close(fig_1d)

        return transformed_points_2d, transformed_points_1d

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

    @classmethod
    def combine_images(cls: Map64, out_name: str = "combined_image.png") -> None:
        map64_dir = "experimental/DotPatternMap/images/map64"
        points_box_dir = "experimental/DotPatternMap/images/points_box"
        pca_2d_dir = "experimental/DotPatternMap/images/pca_2d"
        pca_1d_dir = "experimental/DotPatternMap/images/pca_1d"
        map64_jet_dir = "experimental/DotPatternMap/images/map64_jet"

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
        pca_2d_images = [
            cv2.imread(os.path.join(pca_2d_dir, filename), cv2.IMREAD_COLOR)
            for filename in os.listdir(pca_2d_dir)
            if cv2.imread(os.path.join(pca_2d_dir, filename), cv2.IMREAD_COLOR)
            is not None
        ]

        pca_1d_images = [
            cv2.imread(os.path.join(pca_1d_dir, filename), cv2.IMREAD_COLOR)
            for filename in os.listdir(pca_1d_dir)
            if cv2.imread(os.path.join(pca_1d_dir, filename), cv2.IMREAD_COLOR)
            is not None
        ]

        map64_jet_images = [
            cv2.imread(os.path.join(map64_jet_dir, filename), cv2.IMREAD_COLOR)
            for filename in os.listdir(map64_jet_dir)
            if cv2.imread(os.path.join(map64_jet_dir, filename), cv2.IMREAD_COLOR)
            is not None
        ]

        brightness = [np.sum(img) for img in map64_images]

        sorted_indices = np.argsort(brightness)[::-1]
        map64_images = [map64_images[i] for i in sorted_indices]
        points_box_images = [points_box_images[i] for i in sorted_indices]
        pca_2d_images = [pca_2d_images[i] for i in sorted_indices]
        pca_1d_images = [pca_1d_images[i] for i in sorted_indices]
        map64_jet_images = [map64_jet_images[i] for i in sorted_indices]

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
                img = cv2.resize(
                    img, (image_size, image_size), interpolation=cv2.INTER_LINEAR
                )
                if img.ndim == 2:
                    img = img[:, :, np.newaxis]
                combined_image[y : y + image_size, x : x + image_size] = img
            return combined_image

        combined_map64_image = combine_images_grid(map64_images, 64, 1)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/{out_name}",
            combined_map64_image,
        )

        combined_points_box_image = combine_images_grid(points_box_images, 256, 3)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/{out_name}_combined_image_box.png",
            combined_points_box_image,
        )

        combined_pca_2d_image = combine_images_grid(pca_2d_images, 64, 3)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/{out_name}_combined_image_pca_2d.png",
            combined_pca_2d_image,
        )

        combined_pca_1d_image = combine_images_grid(pca_1d_images, 64, 3)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/{out_name}_combined_image_pca_1d.png",
            combined_pca_1d_image,
        )

        combined_map64_jet_image = combine_images_grid(map64_jet_images, 64, 3)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/{out_name}_combined_image_jet.png",
            combined_map64_jet_image,
        )

    @classmethod
    def extract_probability_map(cls, out_name: str) -> np.ndarray:
        # 64x64の画像を左右反転、上下反転、回転させた画像を作成
        def augment_image(image: np.ndarray) -> list[np.ndarray]:
            augmented_images = []
            augmented_images.append(image)
            augmented_images.append(cv2.flip(image, 0))
            augmented_images.append(cv2.flip(image, 1))
            augmented_images.append(cv2.flip(image, -1))
            for i in range(1, 4):
                augmented_images.append(cv2.rotate(image, i))
            return augmented_images

        # /map64の画像を読み込む
        map64_dir = "experimental/DotPatternMap/images/map64"
        map64_images = [
            cv2.imread(os.path.join(map64_dir, filename), cv2.IMREAD_GRAYSCALE)
            for filename in os.listdir(map64_dir)
            if cv2.imread(os.path.join(map64_dir, filename), cv2.IMREAD_GRAYSCALE)
            is not None
        ]

        # 画像を左右反転、上下反転、回転させた画像を作成
        augmented_images = []
        for image in map64_images:
            augmented_images.extend(augment_image(image))
        # 画像を左右反転、上下反転、回転させた画像を全て重ねる(輝度を平均)
        probability_map = np.mean(augmented_images, axis=0)
        # 画像を保存
        cv2.imwrite(
            f"experimental/DotPatternMap/images/probability_map_{out_name}.png",
            probability_map,
        )
        return probability_map


def delete_pngs(dir: str) -> None:
    for filename in [
        i
        for i in os.listdir(f"experimental/DotPatternMap/images/{dir}")
        if i.endswith(".png")
    ]:
        os.remove(os.path.join(f"experimental/DotPatternMap/images/{dir}", filename))


def main(db: str):
    for i in [
        "map64",
        "points_box",
        "pca_2d",
        "pca_1d",
        "fluo_raw",
        "map64_jet",
        "map64_raw",
    ]:
        delete_pngs(i)
    cells: list[Cell] = database_parser(db)
    map64: Map64 = Map64()
    vectors = []
    for cell in tqdm(cells):
        vectors.append(map64.extract_map(cell.img_fluo1, cell.contour, 4, cell.cell_id))
    map64.combine_images(out_name=db.replace(".db", ".png"))
    map64.extract_probability_map(db.replace(".db", ""))
    return vectors


if __name__ == "__main__":
    for i in os.listdir("experimental/DotPatternMap"):
        if i.endswith(".db"):
            with open(f"experimental/DotPatternMap/images/{i}_vectors.txt", "w") as f:
                for vector in main(f"{i}"):
                    f.write(f"{','.join(map(str, vector))}\n")
