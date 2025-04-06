#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本スクリプトは、データベースに格納された細胞レコード（Cell）ごとに
1. map64_raw (背景引き算済みの座標配置画像)
2. map64_jet (上記を Jet カラーマップで可視化)

をそれぞれ experimental/DotPatternMap/images/ 内のサブディレクトリに保存します。

依存ライブラリ:
    pip install numpy opencv-python matplotlib tqdm

Pythonバージョン: 3.12を想定
"""

from __future__ import annotations
import os
import sys
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.optimize import minimize
from scipy.integrate import quad
from dataclasses import dataclass
from numpy.linalg import eig, inv
from tqdm import tqdm

# ここで任意のデータベース読込用のモジュールをインポート
# database_parser, Cellクラスの実装は既にあることを想定
# 必要に応じてパスや実装を合わせてください
from database_parser import database_parser, Cell


# ============================ ユーティリティ関数 ============================
def ensure_dirs() -> None:
    """結果を保存するディレクトリが存在しない場合は作成する。"""
    base_dir = "experimental/DotPatternMap/images"
    subdirs = [
        "map64",
        "map64_raw",
        "map64_jet",
        "fluo_raw",
    ]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for subdir in subdirs:
        target_path = os.path.join(base_dir, subdir)
        if not os.path.exists(target_path):
            os.makedirs(target_path)


def subtract_background(gray_img: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    """
    グレースケール画像に対してモルフォロジー演算で背景を推定し、
    元画像 - 背景 を返す（背景差分）。
    """
    kernel: np.ndarray = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    background: np.ndarray = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    subtracted: np.ndarray = cv2.subtract(gray_img, background)
    return subtracted


# ============================ メイン処理クラス ============================
class Map64:
    """
    細胞ごとに以下の画像を出力する:
      - map64_raw: 背景差分後の輝度を 2次元座標にマッピングした画像
      - map64 (64x64): 上記を縮小して最終 64x64 画像
      - map64_jet: 64x64 を Jet カラーマップで可視化した画像

    ※ このクラス内には大枠の処理が詰まっているため、最小限実装だけ抜粋しています。
    """

    @dataclass
    class Point:
        """
        細胞内部座標を扱うための簡易データクラス
        p, q, u1, u2, dist, G, sign など各種メンバーを持つが
        ここでは主に map64_raw 生成時に利用
        """
        p: float
        q: float
        u1: float
        u2: float
        dist: float
        G: float
        sign: int

    @classmethod
    def flip_image_if_needed(cls, image: np.ndarray) -> np.ndarray:
        """
        64x64 の最終画像において、右半分が左半分よりも明るい場合に水平反転する。
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        left_half = image[:, : w // 2]
        right_half = image[:, w // 2 :]
        if np.mean(right_half) > np.mean(left_half):
            image = cv2.flip(image, 1)
        return image

    @classmethod
    def poly_fit(cls, U: list[list[float]], degree: int = 1) -> list[float]:
        """
        基底変換後の (u1, u2) 座標列 U を用いて、多項式近似の係数を求める。
        """
        u1_values = np.array([i[1] for i in U])
        f_values = np.array([i[0] for i in U])
        W = np.vander(u1_values, degree + 1)
        return inv(W.T @ W) @ W.T @ f_values

    @classmethod
    def find_minimum_distance_and_point(
        cls, coefficients: float, x_Q: float, y_Q: float
    ) -> tuple[float, tuple[float, float]]:
        """
        多項式曲線 y=f(x) と任意点 (x_Q, y_Q) の最短距離を求め、その最短距離と最短点を返す。
        """
        def f_x(x):
            return sum(
                c * x**i for i, c in enumerate(coefficients[::-1])
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
    def basis_conversion(
        cls,
        contour: list[list[int]],
        X: np.ndarray,
        center_x: float,
        center_y: float,
        coordinates_incide_cell: list[list[int]],
    ) -> tuple:
        """
        主成分（共分散行列の固有ベクトル）に基づいて座標を変換し、u1, u2 座標列を返す。
        """
        Sigma = np.cov(X)
        eigenvalues, eigenvectors = eig(Sigma)

        # 固有値の大小で並び替え
        if eigenvalues[1] < eigenvalues[0]:
            Q = np.array([eigenvectors[1], eigenvectors[0]])
        else:
            Q = np.array([eigenvectors[0], eigenvectors[1]])

        # 細胞内点を基底変換
        U = []
        for (row, col) in coordinates_incide_cell:
            # (row, col) は (y, x) なので順序を合わせる
            uv = Q.transpose() @ np.array([col, row])
            # u2 -> uv[0], u1 -> uv[1] として保存
            U.append([uv[0], uv[1]])

        # 輪郭点も基底変換
        contour_U = []
        for (col, row) in contour:
            uv = Q.transpose() @ np.array([col, row])
            contour_U.append([uv[0], uv[1]])

        u1 = [i[1] for i in U]
        u2 = [i[0] for i in U]
        u1_contour = [i[1] for i in contour_U]
        u2_contour = [i[0] for i in contour_U]
        min_u1, max_u1 = min(u1), max(u1)

        center = np.array([center_x, center_y])
        # 中心点も同じ基底で変換しておく（必要に応じて）
        u_center = center @ Q
        u1_c, u2_c = u_center[1], u_center[0]

        return u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U

    @classmethod
    def extract_map(
        cls,
        image_fluo_raw: bytes,
        contour_raw: bytes,
        degree: int,
        cell_id: str = "default_cell_id",
        db_prefix: str = "",
    ):
        """
        与えられた Cell レコードの蛍光画像と輪郭情報から、
        以下のファイルを experimental/DotPatternMap/images/ に出力する:
          - fluo_raw/{db_prefix}_{cell_id}.png
          - map64_raw/{db_prefix}_{cell_id}.png
          - map64/{db_prefix}_{cell_id}.png
          - map64_jet/{db_prefix}_{cell_id}.png
        """
        def calculate_arc_length(theta, x1, x_target):
            """多項式近似曲線の長さを数値積分で求める。"""
            poly_derivative = np.polyder(theta)
            def arc_length_function(x):
                return np.sqrt(1 + (np.polyval(poly_derivative, x)) ** 2)
            arc_length, _ = quad(arc_length_function, x1, x_target)
            return arc_length

        # 1) 画像デコード
        image_fluo = cv2.imdecode(np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR)
        image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

        # 2) 背景差分
        image_fluo_gray = subtract_background(image_fluo_gray)

        # 3) マスク生成
        mask = np.zeros_like(image_fluo_gray)
        contour_points = pickle.loads(contour_raw)  # 輪郭データ
        cv2.fillPoly(mask, [contour_points], 255)

        coords_inside_cell = np.column_stack(np.where(mask))
        points_inside_cell = image_fluo_gray[coords_inside_cell[:, 0], coords_inside_cell[:, 1]]

        # 輪郭を描画して fluo_raw 出力
        cv2.polylines(image_fluo, [contour_points], True, (0, 255, 0), 2)
        cv2.imwrite(f"experimental/DotPatternMap/images/fluo_raw/{db_prefix}_{cell_id}.png", image_fluo)

        # 4) 基底変換
        # X は (x座標群, y座標群) 形状の2×N行列が多いが、ここでは転置に注意
        X = np.array([
            coords_inside_cell[:, 1],  # x座標
            coords_inside_cell[:, 0],  # y座標
        ])
        (u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U) = cls.basis_conversion(
            [list(i[0]) for i in contour_points],
            X,
            image_fluo.shape[1] / 2,  # center_x = 画像幅/2
            image_fluo.shape[0] / 2,  # center_y = 画像高さ/2
            coords_inside_cell,
        )

        # 5) 多項式近似
        theta = cls.poly_fit(U, degree=degree)

        # 6) 曲線からの距離などを計算し、map64_raw用データを構築
        raw_points: list[cls.Point] = []
        for (i, j), brightness in zip(coords_inside_cell, points_inside_cell):
            # i, j = y, x なので、U配列と合わせるなら (u1, u2) は (j, i) 的に扱われている
            # ただしすでにUには格納済みなので、ループではなく u1, u2 を直接使ってもOK
            # ここでは比較的分かりやすく (i,j)->(u1, u2) 変換後で扱う例
            # ------------------------------------------
            #  u1, u2 は該当ピクセルが何番目かを調べて取得してもよいが、
            #  本コードでは抽出済みのUを同じ順序で使うために zip() の工夫が必要になる。
            # ------------------------------------------
            # 手短にするため "find_minimum_distance_and_point" を使わず単に近似線との距離を計算してもよい。

            # U配列と同じ要素順序で取得: indexを明示
            # ※ 大量のピクセルがあるため若干非効率ですが、わかりやすさ優先
            # ------------------------------------------
            # 実際には下記 raw_points 生成で同様のループを回す形です。
            pass

        # ↓ 実際の実装: U, points_inside_cell を同時にループ
        tmp_points = []
        for (uu2, uu1), brightness in zip(U, points_inside_cell):
            # uu1 -> x方向, uu2 -> y方向 として多項式との距離を計算
            min_distance, min_point = cls.find_minimum_distance_and_point(theta, uu1, uu2)
            sign = 1 if uu2 > min_point[1] else -1

            # アーク長
            arc_len = calculate_arc_length(theta, min(u1), min_point[0])
            p_val = arc_len   # p として曲線方向の長さ
            q_val = min_point[1]  # 近似曲線の u2 方向位置

            tmp_points.append(cls.Point(
                p = p_val,
                q = q_val,
                u1 = uu1,
                u2 = uu2,
                dist = min_distance,
                G = brightness,
                sign = sign,
            ))

        raw_points = sorted(tmp_points, key=lambda x: x.p)

        # 7) map64_raw 画像の作成
        #    p(曲線長) を x軸、dist(曲線からの直交距離) を y軸にプロット
        ps = np.array([pt.p for pt in raw_points])
        dists_signed = np.array([pt.dist * pt.sign for pt in raw_points])
        gs = np.array([pt.G for pt in raw_points])  # 輝度(0~255)

        min_p, max_p = ps.min(), ps.max()
        min_dist, max_dist = dists_signed.min(), dists_signed.max()

        # 背景用のグレー値：内部ピクセルの輝度の中央値
        median_intensity = int(np.median(gs))

        scale_factor = 1
        scaled_width = max(1, int((max_p - min_p) * scale_factor))
        scaled_height = max(1, int((max_dist - min_dist) * scale_factor))

        high_res_image = np.full(
            (scaled_height, scaled_width), median_intensity, dtype=np.uint8
        )
        for p_val, dist_val, g_val in zip(ps, dists_signed, gs):
            x_pt = int((p_val - min_p) * scale_factor)
            y_pt = int((dist_val - min_dist) * scale_factor)
            # 円を描く（あまり重ならないなら1pxでもOK）
            cv2.circle(high_res_image, (x_pt, y_pt), 1, int(g_val), -1)

        # 背景引き算済みの「生」画像として保存
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_raw/{db_prefix}_{cell_id}.png",
            high_res_image
        )

        # 8) 64x64にリサイズ
        resized_64 = cv2.resize(high_res_image, (64, 64), interpolation=cv2.INTER_NEAREST)
        # 右側が明るければ反転
        resized_64 = cls.flip_image_if_needed(resized_64)

        # map64 保存
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64/{db_prefix}_{cell_id}.png",
            resized_64
        )

        # map64_jet の可視化
        jet_img = cv2.applyColorMap(resized_64, cv2.COLORMAP_JET)
        cv2.imwrite(
            f"experimental/DotPatternMap/images/map64_jet/{db_prefix}_{cell_id}.png",
            jet_img
        )

        # ここでは戻り値なし
        return


# ============================ メイン関数 ============================
def main(db_path: str) -> None:
    """
    指定された .db ファイルを解析し、
    細胞ごとに map64_raw, map64, map64_jet を PNG で出力する。
    """

    # DB名プレフィックス(拡張子なし)
    db_prefix = os.path.splitext(os.path.basename(db_path))[0]

    # 出力先ディレクトリ準備
    ensure_dirs()

    # データベースをパースして cells を取得
    # cells: list[Cell] であることを想定
    cells = database_parser(db_path)

    # 細胞ごとに処理
    for cell in tqdm(cells, desc=f"Processing {db_prefix}"):
        # map64処理
        Map64.extract_map(
            image_fluo_raw=cell.img_fluo1,
            contour_raw=cell.contour,
            degree=4,                  # 多項式近似の次数
            cell_id=cell.cell_id,      # 細胞ID
            db_prefix=db_prefix        # DB名プレフィックス
        )

    print("処理が完了しました。実行ディレクトリ以下にある experimental/DotPatternMap/images/ に結果が保存されています。")


# ============================ エントリポイント ============================
if __name__ == "__main__":
    """
    使い方:
      python script_name.py your_database.db

    実行すると experimental/DotPatternMap/images/ 配下に
    - fluo_raw/{DB_PREFIX}_{CELL_ID}.png
    - map64_raw/{DB_PREFIX}_{CELL_ID}.png
    - map64/{DB_PREFIX}_{CELL_ID}.png
    - map64_jet/{DB_PREFIX}_{CELL_ID}.png

    が生成されます。
    """
    if len(sys.argv) < 2:
        print("Usage: python script_name.py /path/to/database.db")
        sys.exit(1)

    db_file = sys.argv[1]
    if not os.path.isfile(db_file):
        print(f"Error: File not found: {db_file}")
        sys.exit(1)

    main(db_file)
