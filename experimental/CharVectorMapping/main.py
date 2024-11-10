import os
import numpy as np
from skimage import io, color, feature
from skimage.feature import local_binary_pattern
from mahotas.features import zernike_moments
from skimage import img_as_ubyte
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 画像パスの設定
image_contorols = "experimental/CharVectorMapping/images/dataset/ctrls"
image_positives = "experimental/CharVectorMapping/images/dataset/positives"

image_ctrls_paths = sorted([
    os.path.join(image_contorols, file) for file in os.listdir(image_contorols) if file.endswith(".png")
])
image_positives_paths = sorted([
    os.path.join(image_positives, file) for file in os.listdir(image_positives) if file.endswith(".png")
])

# それぞれの群の画像を0.png, 1.png, 2.png, ... としてリネーム
for i, path in enumerate(image_ctrls_paths):
    os.rename(path, os.path.join(image_contorols, f"{i}.png"))
for i, path in enumerate(image_positives_paths):
    os.rename(path, os.path.join(image_positives, f"{i}.png"))

image_ctrls_paths = [
    os.path.join(image_contorols, file) for file in os.listdir(image_contorols) if file.endswith(".png")
]
image_positives_paths = [
    os.path.join(image_positives, file) for file in os.listdir(image_positives) if file.endswith(".png")
]



# 特徴量抽出関数 (LBP)
def extract_lbp_features(image_path):
    image = io.imread(image_path)
    if image.ndim == 3:  # RGBの場合はグレースケールに変換
        image = color.rgb2gray(image)
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)
    return lbp_hist

# 特徴量抽出関数 (Zernike Moments)
def extract_zernike_features(image_path, radius=100):
    image = io.imread(image_path)
    if image.ndim == 3:  # RGBの場合はグレースケールに変換
        image = color.rgb2gray(image)
    image = img_as_ubyte(image)  # Zernike Moments は uint8 に対応
    zernike_moments_features = zernike_moments(image, radius)
    return zernike_moments_features

# コントロール群とpositive群の特徴量抽出
features_ctrls = [extract_lbp_features(path) for path in image_ctrls_paths]
features_positives = [extract_lbp_features(path) for path in image_positives_paths]

# Zernike Moments の使用例（必要に応じて切り替えて試す）
features_ctrls = [extract_zernike_features(path) for path in image_ctrls_paths]
features_positives = [extract_zernike_features(path) for path in image_positives_paths]

# 特徴量を結合し、ラベルとファイル名を設定
X = np.vstack((features_ctrls, features_positives))
y = np.array([0] * len(features_ctrls) + [1] * len(features_positives))
image_paths = image_ctrls_paths + image_positives_paths

# PCAの適用
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 2Dプロット
plt.figure(figsize=(12, 8))
for i in range(X_pca.shape[0]):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], c='blue' if y[i] == 0 else 'red', alpha=0.7)
    plt.text(X_pca[i, 0], X_pca[i, 1], os.path.basename(image_paths[i]).replace(".png",""), fontsize=8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 2D Projection with Image Names')
plt.show()

# 3Dプロット
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(X_pca.shape[0]):
    ax.scatter(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], c='blue' if y[i] == 0 else 'red', alpha=0.7)
    ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], os.path.basename(image_paths[i]).replace(".png",""), fontsize=8)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3D Projection with Image Names')
plt.show()
