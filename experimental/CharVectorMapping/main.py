import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

# LBP特徴量の抽出
def extract_lbp_features(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # 正規化
    return hist

# HOG特徴量の抽出
def extract_hog_features(image):
    hog_features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return hog_features

# fluo_maskedフォルダにある画像パスを取得
image_folder = "experimental/CharVectorMapping/images/fluo_masked"
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.endswith(".png")
]

# 画像の読み込みとLBP & HOG特徴抽出
features = []
file_names = []
for img_path in image_paths:
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # グレースケールで読み込み
    image = cv2.resize(image, (200, 200))  # サイズを統一
    lbp_feature = extract_lbp_features(image)
    hog_feature = extract_hog_features(image)
    combined_feature = np.concatenate((lbp_feature, hog_feature))  # LBPとHOGの結合
    features.append(combined_feature)
    file_names.append(img_path)

# 特徴量を一つの配列に結合
features = np.array(features)

# 次元削減 (PCA)
pca = PCA(n_components=3)  # 2次元または3次元に変更可能
reduced_features = pca.fit_transform(features)

# 次元削減結果の可視化 (2Dプロット)
plt.figure(figsize=(10, 7))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
plt.title("PCA Visualization of Combined LBP and HOG Features (2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# 次元削減結果の可視化 (3Dプロット)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], alpha=0.7
)
ax.set_title("PCA Visualization of Combined LBP and HOG Features (3D)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.show()
