import os
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.feature import hog
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 画像パスの設定
image_dir = "experimental/DotPatternMap/images/map64"
image_paths = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.endswith(".png")
]


# 特徴量抽出関数 (HOG特徴量)
def extract_hog_features(image_path):
    image = io.imread(image_path)
    if image.ndim == 3:  # RGBの場合はグレースケールに変換
        image = color.rgb2gray(image)
    image = img_as_ubyte(image)  # HOG特徴量はuint8に対応
    hog_features = hog(
        image,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,  # visualizeは必要ない場合Falseに
    )
    return hog_features


# 特徴量抽出
features = [extract_hog_features(path) for path in image_paths]

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
ax.set_title("PCA 3D Projection with Image Names (HOG Features)")
plt.savefig("experimental/DotPatternMap/images/PCA_3D_HOG_map64.png")

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
plt.title("PCA 2D Projection with Image Names (HOG Features)")
plt.savefig("experimental/DotPatternMap/images/PCA_2D_HOG_map64.png")
