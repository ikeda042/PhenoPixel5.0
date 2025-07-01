#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
images/ に保存された各細胞画像（サイズ 1024x256 のPNG）に対して、
Moran's I を計算し、結果を表示するスクリプトの例。

Python 3.12 での実行を想定。
事前に以下のパッケージをインストールしてください:
    pip install scikit-image libpysal esda
"""

import os
import glob
import numpy as np
import libpysal
from esda.moran import Moran
from skimage.io import imread

def compute_moran_i_for_image(img: np.ndarray, rook: bool = True) -> float:
    """
    画像配列を入力とし、Moran's I を計算して返す関数。
    rook=True で4近傍、Falseで8近傍(queen adjacency)を設定。
    """

    # img.shape -> (height, width)
    height, width = img.shape

    # 画像を一次元配列にフラット化
    intensities = img.ravel().astype(float)

    # lat2W(row, col, ...) により画素ごとの隣接リストを作成
    # rook=True なら上下左右、False なら対角含む8近傍
    w = libpysal.weights.lat2W(nrows=height, ncols=width, rook=rook)
    w.transform = 'r'  # 行標準化 (row-standardized)

    # Moran の計算
    moran = Moran(intensities, w)
    return moran.I, moran.p_sim

def main():
    # 解析対象のPNG画像を一括取得
    base = "experimental/MoranI/images/"
    image_paths = os.listdir(base)
    image_paths = [os.path.join(base, f) for f in image_paths if f.endswith('.png')]
    if not image_paths:
        print("images/ フォルダに PNG 画像が見つかりません。")
        return

    results = []
    for path in image_paths:
        # 画像読み込み (グレースケール想定)
        img = imread(path, as_gray=True)  # shape: (1024,256) を想定
        
        # 画像サイズが想定と異なる場合はスキップ or 警告
        if img.shape != (1024, 256):
            print(f"警告: {path} の画像サイズが (1024,256) ではありません -> {img.shape}")
            # 必要に応じて continue や resize を検討
            # ここでは強行的に計算する例
            pass
        
        moran_i, p_value = compute_moran_i_for_image(img, rook=True)
        results.append((path, moran_i, p_value))

    # 結果を表示
    print("=== Moran's I 結果一覧 (Rook隣接, 4近傍) ===")
    for (path, mi, pv) in results:
        print(f"{os.path.basename(path):30s} | Moran's I = {mi:.4f} | p-value = {pv:.5f}")

if __name__ == "__main__":
    main()
