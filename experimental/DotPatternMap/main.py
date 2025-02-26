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
    base_dir = "experimental/DotPatternMap/images"
    subdirs = [
        "map64",
        "points_box",
        # "pca_2d",  # PCA-2D 関連はコメントアウト
        # "pca_1d",  # PCA-1D 関連はコメントアウト
        "fluo_raw",
        "map64_jet",
        "map64_raw",
        # "polar",   # polar 処理はコメントアウト
        "dot_loc",  # dot位置結果の出力先
    ]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
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
        """
        画像の左右輝度を見て、右側が明るければ水平反転
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        left_half = image[:, : w // 2]
        right_half = image[:, w // 2 :]
        left_brightness = np.mean(left_half)
        right_brightness = np.mean(right_half)
        if right_brightness > left_brightness:
            image = cv2.flip(image, 1)
        return image

    @classmethod
    def find_minimum_distance_and_point(
        cls, coefficients: float, x_Q: float, y_Q: float
    ) -> tuple[float, tuple[float, float]]:
        """
        多項式曲線 y=f(x) と任意点 (x_Q, y_Q) の最短距離を求める
        """

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
        """
        U = [[u2, u1], [u2, u1], ...] の2次元点から、
        u1 を入力（x）、u2 を出力（y）として 多項式近似を行い係数を返す
        """
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
        """
        共分散行列の固有ベクトルを用いた基底変換
        """
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
        cls: Map64, image_fluo_raw: bytes, contour_raw: bytes, degree: int
    ) -> None:
        """
        既存の画像バッファと輪郭データから再度plotする例
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
            [[i[1] for i in coords_inside_cell_1], [i[0] for i in coords_inside_cell_1]]
        )
        (u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U) = (
            cls.basis_conversion(
                [list(i[0]) for i in unpickled_contour],
                X,
                image_fluo.shape[0] / 2,
                image_fluo.shape[1] / 2,
                coords_inside_cell_1,
            )
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
        """
        PCA処理は重いのでコメントアウトし、ダミーを返す
        """
        return None

    @classmethod
    def extract_map(
        cls,
        image_fluo_raw: bytes,
        contour_raw: bytes,
        degree: int,
        cell_id: str = "default_cell_id",
    ):
        """
        与えられた蛍光画像バイナリと輪郭を用いて、
        基底変換＋曲線近似＋細胞内部座標の可視化を行い、
        背景差分後の map64_raw を出力する。
        """

        def calculate_arc_length(theta, x1, x_target):
            poly_derivative = np.polyder(theta)

            def arc_length_function(x):
                return np.sqrt(1 + (np.polyval(poly_derivative, x)) ** 2)

            arc_length, _ = quad(arc_length_function, x1, x_target)
            return arc_length

        # 画像デコード
        image_fluo = cv2.imdecode(
            np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR
        )
        # グレースケール化
        image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

        # --- ここで必ず背景差分を行う ---
        image_fluo_gray = subtract_background(image_fluo_gray)

        # マスク生成
        mask = np.zeros_like(image_fluo_gray)
        unpickled_contour = pickle.loads(contour_raw)
        cv2.fillPoly(mask, [unpickled_contour], 255)

        coords_inside_cell_1 = np.column_stack(np.where(mask))
        points_inside_cell_1 = image_fluo_gray[
            coords_inside_cell_1[:, 0], coords_inside_cell_1[:, 1]
        ]

        # 輪郭を重ね描画して fluo_raw に保存
        cv2.polylines(
            image_fluo,
            [unpickled_contour],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.imwrite(
            f"experimental/DotPatternMap/images/fluo_raw/{cell_id}.png", image_fluo
        )

        # 基底変換
        X = np.array(
            [[i[1] for i in coords_inside_cell_1], [i[0] for i in coords_inside_cell_1]]
        )
        (u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U) = (
            cls.basis_conversion(
                [list(i[0]) for i in unpickled_contour],
                X,
                image_fluo.shape[0] / 2,
                image_fluo.shape[1] / 2,
                coords_inside_cell_1,
            )
        )
        theta = cls.poly_fit(U, degree=degree)

        # 距離計算と座標格納
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
                    p,  # 背景差分後の輝度をそのまま格納
                    sign,
                )
            )
        raw_points.sort()

        # 箱ひげ図用plot (points_box)
        fig = plt.figure(figsize=(6, 6))
        plt.axis("equal")
        ps = np.array([i.p for i in raw_points])
        dists = np.array([i.dist * i.sign for i in raw_points])
        gs = np.array([i.G for i in raw_points])  # 背景差分後の輝度

        gs_used = gs  # そのまま使用（正規化しない）

        min_p, max_p = np.min(ps), np.max(ps)
        min_dist, max_dist = np.min(dists), np.max(dists)
        plt.scatter(ps, dists, s=80, c=gs_used, cmap="jet", vmin=0, vmax=255)
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

        # map64_rawの作成
        scale_factor = 1
        scaled_width = int((max_p - min_p) * scale_factor)
        scaled_height = int((max_dist - min_dist) * scale_factor)
        median_intensity = int(np.median(points_inside_cell_1))
        high_res_image = np.full(
            (scaled_height, scaled_width), median_intensity, dtype=np.uint8
        )
        for p, dist, G in zip(ps, dists, gs_used):
            p_scaled = int((p - min_p) * scale_factor)
            dist_scaled = int((dist - min_dist) * scale_factor)
            cv2.circle(high_res_image, (p_scaled, dist_scaled), 1, int(G), -1)

        # 背景差分後の画像を map64_raw に保存
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_raw/{cell_id}.png", high_res_image
        )

        # 64x64に縮小 & 右側が明るければフリップ
        high_res_image = cv2.resize(
            high_res_image, (64, 64), interpolation=cv2.INTER_NEAREST
        )
        high_res_image = cls.flip_image_if_needed(high_res_image)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64/{cell_id}.png", high_res_image
        )

        # そのままJetカラーマップで可視化
        high_res_image_colormap = cv2.applyColorMap(high_res_image, cv2.COLORMAP_JET)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_jet/{cell_id}.png",
            high_res_image_colormap,
        )
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_jet.png", high_res_image_colormap
        )

        # PCA処理はコメントアウト（高速化）
        """
        return cls.perform_pca_on_3d_point_cloud_and_save(
            high_res_image,
            f"experimental/DotPatternMap/images/pca_2d/{cell_id}.png",
            f"experimental/DotPatternMap/images/pca_1d/{cell_id}.png",
        )
        """
        return high_res_image

    @classmethod
    def combine_images(cls: Map64, out_name: str = "combined_image.png") -> None:
        """
        map64, points_box, map64_jet の画像をそれぞれまとめて並べた画像を作成
        """
        map64_dir = "experimental/DotPatternMap/images/map64"
        points_box_dir = "experimental/DotPatternMap/images/points_box"
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
        map64_jet_images = [
            cv2.imread(os.path.join(map64_jet_dir, filename), cv2.IMREAD_COLOR)
            for filename in os.listdir(map64_jet_dir)
            if cv2.imread(os.path.join(map64_jet_dir, filename), cv2.IMREAD_COLOR)
            is not None
        ]

        # 明るい順にソート（単純に和の大きいものを先頭に）
        brightness = [np.sum(img) for img in map64_images]
        sorted_indices = np.argsort(brightness)[::-1]
        map64_images = [map64_images[i] for i in sorted_indices]
        points_box_images = [points_box_images[i] for i in sorted_indices]
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
            f"experimental/DotPatternMap/images/{out_name}", combined_map64_image
        )

        combined_points_box_image = combine_images_grid(points_box_images, 256, 3)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/{out_name}_combined_image_box.png",
            combined_points_box_image,
        )

        combined_map64_jet_image = combine_images_grid(map64_jet_images, 64, 3)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/{out_name}_combined_image_jet.png",
            combined_map64_jet_image,
        )


def detect_dot(image_path: str) -> list[tuple[int, int, float]]:
    """
    map64_raw の画像を読み込み、輝度の高いドットを検出し、
    各ドットの(x, y, ドット領域の平均輝度)を返す。
    また、2値化画像（binary）と正規化画像（norm）、
    ドット検出結果（detected）の各画像を dot_loc フォルダに保存する。

    ※ 各画像中の輪郭の合計面積が 30 を超えた場合は、ドットがないものと判断します。

    さらに、2値化画像（thresh）における255のピクセルのx軸, y軸位置の変動係数
    $$\mathrm{CV} = \frac{\sigma}{\mu}$$
    （LaTeXコード: \mathrm{CV} = \frac{\sigma}{\mu}）
    が所定の閾値より大きい場合は、ドット検出を無効として全て0塗りします。
    """
    print("=========================================")
    print(f"Detecting dots in {image_path}")
    image = cv2.imread(image_path)
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    median_val = np.median(norm_gray)
    top97_val = np.percentile(norm_gray, 97)
    diff = top97_val - median_val
    print(f"diff: {diff}")
    print(f"median: {median_val}")
    dot_diff_threshold = 70

    coordinates: list[tuple[int, int, float]] = []

    # 保存先ディレクトリ(dot_loc)とファイル名のベースを設定
    dot_loc_dir = "experimental/DotPatternMap/images/dot_loc"
    if not os.path.exists(dot_loc_dir):
        os.makedirs(dot_loc_dir)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if diff > dot_diff_threshold:
        # ドットがある場合：しきい値180で2値化
        ret, thresh = cv2.threshold(norm_gray, 140, 255, cv2.THRESH_BINARY)

        # thresh画像における255ピクセルのx軸, y軸位置の変動係数を計算する
        white_pixels = np.where(thresh == 255)
        discard_based_on_cv = False
        if white_pixels[0].size > 0:
            # np.whereは (y座標, x座標) の順で返すため
            x_positions = white_pixels[1]
            y_positions = white_pixels[0]
            mean_x = np.mean(x_positions) if np.mean(x_positions) != 0 else 1
            mean_y = np.mean(y_positions) if np.mean(y_positions) != 0 else 1
            cv_x = np.std(x_positions) / mean_x
            cv_y = np.std(y_positions) / mean_y
            print(f"cv_x: {cv_x}, cv_y: {cv_y}")
            # 変動係数の閾値 (例として0.5を使用、必要に応じて調整)
            cv_threshold = 50
            # if cv_x > cv_threshold or cv_y > cv_threshold:
            #     print(
            #         "Coefficient of variation threshold exceeded. Discarding dot detection."
            #     )
            #     discard_based_on_cv = True
            #     thresh[:] = 0  # thresh画像を全て0にする

        # thresh画像中の255ピクセルの総数が多い場合も、ドットがないと判断
        if discard_based_on_cv or np.sum(thresh == 255) > 300:
            coordinates = []
            detected_img = np.zeros_like(image)
        else:
            # 輪郭検出（外側の輪郭のみ取得）
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # 各輪郭の面積の合計を計算
            total_area = sum(cv2.contourArea(cnt) for cnt in contours)
            print(f"Total contour area: {total_area}")

            if total_area > 30:
                # 合計面積が30を超える場合はドットがないと判断
                coordinates = []
                detected_img = np.zeros_like(image)
            else:
                for cnt in contours:
                    # モーメントを計算し、重心を求める
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # ドット領域の平均輝度を算出 (もとの norm_gray で計算)
                        mask = np.zeros_like(norm_gray)
                        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
                        avg_brightness = cv2.mean(norm_gray, mask=mask)[0]
                        coordinates.append((cX, cY, avg_brightness))

                # 検出結果の可視化用画像作成
                detected_img = np.zeros_like(image)
                cv2.drawContours(detected_img, contours, -1, (255, 255, 255), 2)
    else:
        # ドットがない場合
        thresh = np.zeros_like(norm_gray)
        detected_img = np.zeros_like(image)

    # 出力画像を dot_loc フォルダに保存
    cv2.imwrite(os.path.join(dot_loc_dir, f"{base_name}_detected.png"), detected_img)
    cv2.imwrite(os.path.join(dot_loc_dir, f"{base_name}_binary.png"), thresh)
    cv2.imwrite(os.path.join(dot_loc_dir, f"{base_name}_norm.png"), norm_gray)

    return coordinates


import matplotlib.colors


def compute_avg_brightness(image: np.ndarray, x: float, y: float, radius=2) -> float:
    """
    指定座標 (x, y) 周辺の (2*radius+1) x (2*radius+1) の領域の平均輝度を算出する。
    画像の境界を考慮してパッチを抽出する。
    """
    x_int = int(round(x))
    y_int = int(round(y))
    h, w = image.shape
    x0 = max(0, x_int - radius)
    x1 = min(w, x_int + radius + 1)
    y0 = max(0, y_int - radius)
    y1 = min(h, y_int + radius + 1)
    patch = image[y0:y1, x0:x1]
    return patch.mean()


def process_dot_locations():
    """
    experimental/DotPatternMap/images/map64_raw 内の各画像に対し、
    detect_dot() を用いてドットの中心座標を取得し、
    その座標をもとに map64_raw/image.png（元の画像）から小領域の平均輝度を算出します。
    各画像ごとに個別の散布図を保存し、全画像のドット位置と輝度をまとめたヒートマップを
    scatterプロットで表示します。
    """
    map64_raw_dir = "experimental/DotPatternMap/images/map64_raw"
    dot_loc_dir = "experimental/DotPatternMap/images/dot_loc"
    if not os.path.exists(dot_loc_dir):
        os.makedirs(dot_loc_dir)

    all_normalized_dots: list[tuple[float, float, float]] = (
        []
    )  # (normalized x, normalized y, avg_brightness)

    for filename in os.listdir(map64_raw_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(map64_raw_dir, filename)
            print(f"Processing {filename}")
            # detect_dot() によりドットの中心座標を取得（輝度は再計算するので無視）
            dots = detect_dot(image_path)

            # map64_raw/image.png を元に、グレースケール画像として再読み込み
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            h, w = image.shape
            center_x, center_y = w / 2, h / 2

            normalized_dots = []
            for dot in dots:
                # dot は (x, y) または (x, y, _) の形式になっているが、
                # 輝度は map64_raw/image.png から算出する
                x, y = dot[0], dot[1]
                brightness = compute_avg_brightness(image, x, y, radius=2)
                norm_x = (x - center_x) / (w / 2)
                norm_y = (y - center_y) / (h / 2)
                normalized_dots.append((norm_x, norm_y, brightness))

            all_normalized_dots.extend(normalized_dots)

            # 個別の散布図作成
            fig, ax = plt.subplots(figsize=(4, 4))
            if normalized_dots:
                xs = [p[0] for p in normalized_dots]
                ys = [p[1] for p in normalized_dots]
                brightness_vals = [p[2] for p in normalized_dots]
                sc = ax.scatter(
                    xs,
                    ys,
                    c=brightness_vals,
                    cmap="Blues",
                    norm=matplotlib.colors.NoNorm(),  # 輝度は元の値をそのまま使用
                    s=100,
                    label="Dot",
                )
                plt.colorbar(sc, ax=ax, label="Avg Brightness")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No dot detected",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.axhline(0, color="gray", linestyle="--")
            ax.axvline(0, color="gray", linestyle="--")
            ax.set_title(f"Dot Locations for {filename}")
            ax.set_xlabel("Relative X (normalized)")
            ax.set_ylabel("Relative Y (normalized)")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.grid(True)
            ax.legend()

            # 個別プロット画像の保存
            plot_save_path = os.path.join(dot_loc_dir, filename)
            fig.savefig(plot_save_path, dpi=300)
            plt.close(fig)

            print(f"Processed {filename}: {normalized_dots}")

    # 全画像のドット位置と輝度をまとめたヒートマップ作成
    fig, ax = plt.subplots(figsize=(6, 6))
    if all_normalized_dots:
        xs = [p[0] for p in all_normalized_dots]
        ys = [p[1] for p in all_normalized_dots]
        brightness_vals = [p[2] for p in all_normalized_dots]
        sc = ax.scatter(
            xs,
            ys,
            c=brightness_vals,
            cmap="Blues",
            norm=matplotlib.colors.NoNorm(),
            s=50,
            label="All Dots",
        )
        plt.colorbar(sc, ax=ax, label="Avg Brightness")
    else:
        ax.text(
            0.5,
            0.5,
            "No dots detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title("Combined Dot Locations with Brightness")
    ax.set_xlabel("Relative X (normalized)")
    ax.set_ylabel("Relative Y (normalized)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True)
    ax.legend()

    combined_save_path = os.path.join(dot_loc_dir, "combined_dot_locations.png")
    fig.savefig(combined_save_path, dpi=300)
    plt.close(fig)
    print(f"Combined dot locations saved to {combined_save_path}")


def combine_dot_loc_combined_images():
    """
    dot_loc 内の _binary, _detected, _norm の各種画像をグリッド状に結合し、
    experimental/DotPatternMap/images に combined_binary.png, combined_detected.png, combined_norm.png を生成する。
    同じ細胞の画像が各combined画像で同じ位置に来るよう、ファイル名（細胞ID）でソートしています。
    """
    dot_loc_dir = "experimental/DotPatternMap/images/dot_loc"
    images_dir = "experimental/DotPatternMap/images"
    suffixes = ["_binary", "_detected", "_norm"]

    def calculate_grid_size(n: int) -> tuple[int, int]:
        row = int(np.sqrt(n))
        col = (n + row - 1) // row
        return row, col

    def combine_images_grid(images, image_size, channels=1):
        n = len(images)
        row, col = calculate_grid_size(n)
        if channels == 1:
            combined_image = np.zeros(
                (image_size * row, image_size * col), dtype=np.uint8
            )
        else:
            combined_image = np.zeros(
                (image_size * row, image_size * col, channels), dtype=np.uint8
            )
        for idx, img in enumerate(images):
            resized = cv2.resize(
                img, (image_size, image_size), interpolation=cv2.INTER_LINEAR
            )
            if channels == 3 and resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            x = (idx % col) * image_size
            y = (idx // col) * image_size
            combined_image[y : y + image_size, x : x + image_size] = resized
        return combined_image

    for suffix in suffixes:
        # ファイル名をソートすることで、同じ細胞の画像が同じ順序・位置に並ぶようにする
        files = sorted(
            [f for f in os.listdir(dot_loc_dir) if f.endswith(suffix + ".png")]
        )
        if not files:
            print(f"No files found for suffix {suffix}")
            continue
        images = []
        for filename in files:
            img = cv2.imread(os.path.join(dot_loc_dir, filename), cv2.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img)
        if images:
            image_size = 256  # 結合後の各画像サイズ（ピクセル）
            channels = (
                3 if any((img.ndim == 3 and img.shape[2] == 3) for img in images) else 1
            )
            combined_image = combine_images_grid(images, image_size, channels)
            combined_filename = os.path.join(images_dir, f"combined{suffix}.png")
            cv2.imwrite(combined_filename, combined_image)
            print(f"Saved combined image for {suffix} as {combined_filename}")


def main(db: str):
    for i in ["map64", "points_box", "fluo_raw", "map64_jet", "map64_raw"]:
        delete_pngs(i)
    cells: list[Cell] = database_parser(db)
    map64: Map64 = Map64()
    vectors = []
    for cell in tqdm(cells[:]):
        vectors.append(map64.extract_map(cell.img_fluo1, cell.contour, 4, cell.cell_id))
    map64.combine_images(out_name=db.replace(".db", ".png"))
    # 修正: Map64インスタンスではなく、モジュールレベルの関数として呼び出す
    extract_probability_map(db.replace(".db", ""))
    return vectors


def extract_probability_map(out_name: str) -> np.ndarray:
    """
    64x64の map64 をAugmentationして、単純平均した確率マップを作る例
    """

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


# ------------------------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    ensure_dirs()
    for i in os.listdir("experimental/DotPatternMap"):
        if i.endswith(".db"):
            with open(f"experimental/DotPatternMap/images/{i}_vectors.txt", "w") as f:
                for vector in main(f"{i}"):
                    if isinstance(vector, np.ndarray):
                        f.write(f"{','.join(map(str, vector.flatten()))}\n")
                    else:
                        f.write(f"{vector}\n")
    process_dot_locations()
    combine_dot_loc_combined_images()
