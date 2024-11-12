import os
import numpy as np
import cv2  # OpenCVをインポート
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# 画像パスの設定
image_dir = "experimental/DotPatternMap/images/map64"
image_paths = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.endswith(".png")
]


# LBP特徴量を抽出する関数
def extract_lbp_features(image_path, num_points=24, radius=3):
    image = cv2.imread(
        image_path, cv2.IMREAD_GRAYSCALE
    )  # グレースケールで画像を読み込む
    image = cv2.resize(image, (64, 64))  # 画像を64x64ピクセルにリサイズ

    # LBP特徴量を計算
    lbp = local_binary_pattern(image, num_points, radius, method="uniform")

    # ヒストグラムを計算
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # ヒストグラムを正規化
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6  # ゼロ除算を防ぐ

    return hist


# 特徴量抽出
features = []
for path in image_paths:
    feature = extract_lbp_features(path)
    features.append(feature)

# 特徴量を行列に変換
X = np.vstack(features)

# PCAの適用 (n=3)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X)

# 3Dプロット
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
for i in range(X_pca_3d.shape[0]):
    ax.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], alpha=0.7)
    ax.text(
        X_pca_3d[i, 0],
        X_pca_3d[i, 1],
        X_pca_3d[i, 2],
        os.path.basename(image_paths[i]).replace(".png", ""),
        fontsize=8,
    )
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA 3D Projection with Image Names (LBP Features)")
plt.savefig("experimental/DotPatternMap/images/PCA_3D_LBP_map64.png")

# PCAの適用 (n=2)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)

# 2Dプロット
plt.figure(figsize=(12, 8))
for i in range(X_pca_2d.shape[0]):
    plt.scatter(X_pca_2d[i, 0], X_pca_2d[i, 1], alpha=0.7)
    plt.text(
        X_pca_2d[i, 0],
        X_pca_2d[i, 1],
        os.path.basename(image_paths[i]).replace(".png", ""),
        fontsize=8,
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection with Image Names (LBP Features)")
plt.savefig("experimental/DotPatternMap/images/PCA_2D_LBP_map64.png")

plt.show()
