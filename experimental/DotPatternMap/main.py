from __future__ import annotations
import numpy as np
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
import cv2
import pickle
from database_parser import database_parser, Cell

# from sklearn.decomposition import PCA  # PCA処理はコメントアウト
from scipy.optimize import minimize
from scipy.integrate import quad
from dataclasses import dataclass
from tqdm import tqdm
import os


def subtract_background(gray_img: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    """
    背景引き算を行う関数

    数式:
        B(x,y) = morph_open(I(x,y))
        I_sub(x,y) = I(x,y) - B(x,y)

    Latex生コード:
    \[
    B(x,y) = \mathrm{morph\_open}(I(x,y))
    \]
    \[
    I\_sub(x,y) = I(x,y) - B(x,y)
    \]
    """
    kernel: np.ndarray = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    background: np.ndarray = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    subtracted: np.ndarray = cv2.subtract(gray_img, background)
    return subtracted


def ensure_dirs():
    """
    必要なディレクトリが存在しなければ作成する関数。
    実験用のDotPatternMap配下のimagesディレクトリ、およびサブディレクトリを生成。
    """
    # ベースとなるディレクトリ
    base_dir = "experimental/DotPatternMap/images"

    # サブディレクトリの一覧
    subdirs = [
        "map64",
        "points_box",
        # "pca_2d",  # PCA-2D 関連はコメントアウト
        # "pca_1d",  # PCA-1D 関連はコメントアウト
        "fluo_raw",
        "map64_jet",
        "map64_raw",
        # "polar",   # polar 処理はコメントアウト
    ]

    # ベースディレクトリが存在しない場合は作成
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # サブディレクトリが存在しない場合は作成
    for subdir in subdirs:
        target_path = os.path.join(base_dir, subdir)
        if not os.path.exists(target_path):
            os.makedirs(target_path)


def delete_pngs(dir: str) -> None:
    """
    指定されたディレクトリ配下のPNGファイルを削除する関数。
    """
    for filename in [
        i
        for i in os.listdir(f"experimental/DotPatternMap/images/{dir}")
        if i.endswith(".png")
    ]:
        os.remove(os.path.join(f"experimental/DotPatternMap/images/{dir}", filename))


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
        def f_x(x):
            return sum(
                coefficient * x**i for i, coefficient in enumerate(coefficients[::-1])
            )

        def distance(x):
            return np.sqrt((x - x_Q) ** 2 + (f_x(x) - y_Q) ** 2)

        result = minimize(
            distance, 0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-2}
        )

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
        # PCA処理はコメントアウトして高速化（ダミーの戻り値を返す）
        # 元のコードでは3D点群からPCAを行い可視化していたが、今回はスキップします。
        return None

    @classmethod
    def extract_map(
        cls,
        image_fluo_raw: bytes,
        contour_raw: bytes,
        degree: int,
        cell_id: str = "default_cell_id",
    ):
        def calculate_arc_length(theta, x1, x_target):
            poly_derivative = np.polyder(theta)

            def arc_length_function(x):
                return np.sqrt(1 + (np.polyval(poly_derivative, x)) ** 2)

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

        # 画像に輪郭を描画して保存
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

        # vmin, vmaxを固定（0～255）に設定
        plt.scatter(ps, dists, s=80, c=gs_norm, cmap="jet", vmin=0, vmax=255)
        plt.xlabel(r"$L(u_{1_i}^\star)$ (px)")
        plt.ylabel(r"$\text{min\_dist}$ (px)")
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

        # 以下、polarに関する処理はコメントアウトして高速化
        """
        # psとdistsを曲座標変換
        r = [np.sqrt(p**2 + dist**2) for p, dist in zip(ps, dists)]
        theta_vals = [np.arctan2(dist, p) for p, dist in zip(ps, dists)]

        # rとthetaを正規化(0-1)
        r = (r - min(r)) / (max(r) - min(r))
        theta_vals = (theta_vals - min(theta_vals)) / (max(theta_vals) - min(theta_vals))

        # プロット
        fig = plt.figure(figsize=(6, 6))
        plt.axis("equal")
        plt.scatter(theta_vals, gs, s=80, c=gs_norm, cmap="jet")
        plt.ylabel(r"$r_i$ (px)")
        plt.xlabel(r"$\theta_i$ (rad)")
        fig.savefig(f"experimental/DotPatternMap/images/polar/{cell_id}.png", dpi=300)
        plt.close(fig)
        plt.clf()
        """

        # 画像サイズを元の範囲に厳密に設定
        scale_factor = 1
        scaled_width = int((max_p - min_p) * scale_factor)
        scaled_height = int((max_dist - min_dist) * scale_factor)
        high_res_image = np.zeros((scaled_height, scaled_width), dtype=np.uint8)

        for p, dist, G in zip(ps, dists, gs_norm):
            p_scaled = int((p - min_p) * scale_factor)
            dist_scaled = int((dist - min_dist) * scale_factor)
            cv2.circle(high_res_image, (p_scaled, dist_scaled), 1, int(G), -1)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_raw/{cell_id}.png",
            high_res_image,
        )
        high_res_image = cv2.resize(
            high_res_image, (64, 64), interpolation=cv2.INTER_NEAREST
        )
        high_res_image = cls.flip_image_if_needed(high_res_image)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64/{cell_id}.png",
            high_res_image,
        )
        # 固定の最大値255に合わせるため、最大輝度が255未満の場合は線形補正を行う
        max_val = np.max(high_res_image)
        if max_val < 255 and max_val > 0:
            high_res_image = (
                high_res_image.astype(np.float32) * (255.0 / max_val)
            ).astype(np.uint8)
        high_res_image_colormap = cv2.applyColorMap(high_res_image, cv2.COLORMAP_JET)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_jet/{cell_id}.png",
            high_res_image_colormap,
        )
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_jet.png",
            high_res_image_colormap,
        )
        # PCA処理 (pca_2d, pca_1d)はコメントアウトして高速化
        """
        return cls.perform_pca_on_3d_point_cloud_and_save(
            high_res_image,
            f"experimental/DotPatternMap/images/pca_2d/{cell_id}.png",
            f"experimental/DotPatternMap/images/pca_1d/{cell_id}.png",
        )
        """
        # 代わりに、処理済み画像を返す
        return high_res_image

    @classmethod
    def combine_images(cls: Map64, out_name: str = "combined_image.png") -> None:
        map64_dir = "experimental/DotPatternMap/images/map64"
        points_box_dir = "experimental/DotPatternMap/images/points_box"
        # pca_2d_dir = "experimental/DotPatternMap/images/pca_2d"  # コメントアウト
        # pca_1d_dir = "experimental/DotPatternMap/images/pca_1d"  # コメントアウト
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
        # 以下のPCA関連画像は処理対象外とする
        """
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
        """
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
        # pca_2d_images = [pca_2d_images[i] for i in sorted_indices]  # コメントアウト
        # pca_1d_images = [pca_1d_images[i] for i in sorted_indices]  # コメントアウト
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

        # PCA関連の画像は結合処理から除外
        combined_map64_jet_image = combine_images_grid(map64_jet_images, 64, 3)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/{out_name}_combined_image_jet.png",
            combined_map64_jet_image,
        )


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
        # "pca_2d",  # コメントアウト
        # "pca_1d",  # コメントアウト
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


def extract_probability_map(cls, out_name: str) -> np.ndarray:
    # 64x64の画像を左右反転、上下反転、回転させた画像を作成する処理
    def augment_image(image: np.ndarray) -> list[np.ndarray]:
        augmented_images = []
        augmented_images.append(image)
        augmented_images.append(cv2.flip(image, 0))
        augmented_images.append(cv2.flip(image, 1))
        augmented_images.append(cv2.flip(image, -1))
        for i in range(1, 4):
            augmented_images.append(cv2.rotate(image, i))
        return augmented_images

    map64_dir = "experimental/DotPatternMap/images/map64"
    map64_images = [
        cv2.imread(os.path.join(map64_dir, filename), cv2.IMREAD_GRAYSCALE)
        for filename in os.listdir(map64_dir)
        if cv2.imread(os.path.join(map64_dir, filename), cv2.IMREAD_GRAYSCALE)
        is not None
    ]

    augmented_images = []
    for image in map64_images:
        augmented_images.extend(augment_image(image))
    probability_map = np.mean(augmented_images, axis=0).astype(np.uint8)
    cv2.imwrite(
        f"experimental/DotPatternMap/images/probability_map_{out_name}.png",
        probability_map,
    )
    probability_map_jet = cv2.applyColorMap(probability_map, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(
        f"experimental/DotPatternMap/images/probability_map_{out_name}_jet.png",
        probability_map_jet,
    )
    return probability_map


if __name__ == "__main__":
    ensure_dirs()
    for i in os.listdir("experimental/DotPatternMap"):
        if i.endswith(".db"):
            with open(f"experimental/DotPatternMap/images/{i}_vectors.txt", "w") as f:
                for vector in main(f"{i}"):
                    # vectorが ndarray の場合、カンマ区切りに変換して出力
                    if isinstance(vector, np.ndarray):
                        f.write(f"{','.join(map(str, vector.flatten()))}\n")
                    else:
                        f.write(f"{vector}\n")
