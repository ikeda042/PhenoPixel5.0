import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shutil
from mahotas.features import zernike_moments
from skimage.feature import hog


# 画像パスの設定
image_dir = "experimental/DotPatternMap/images/map64"
image_paths = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.endswith(".png")
]


# 特徴抽出関数 (Zernikeモーメント)
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (64, 64))  # 必要に応じてサイズ変更
    _, binary_image = cv2.threshold(image_resized, 128, 255, cv2.THRESH_BINARY)
    radius = 32  # 画像サイズの半分
    zernike_features = zernike_moments(binary_image, radius, degree=8)
    return zernike_features


# すべての画像から特徴を抽出
features = []
cell_ids = []
for path in image_paths:
    cell_id = os.path.basename(path).split(".")[0]
    cell_ids.append(cell_id)
    features.append(extract_features(path))

features = np.array(features)

# 次元削減 (PCA)
pca = PCA(n_components=min(features.shape[0], features.shape[1], 10))
features_pca = pca.fit_transform(features)

# クラスタリング
kmeans = KMeans(n_clusters=5)  # クラスタ数は調整可能
kmeans.fit(features_pca)
labels = kmeans.labels_

# クラスタごとのcellidを表示
clusters = {}
for cell_id, label in zip(cell_ids, labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(cell_id)

# クラスタの結果を表示
for cluster_id, cell_list in clusters.items():
    print(f"Cluster {cluster_id}: {', '.join(cell_list)}")

# クラスタごとに画像を保存
output_dir = "experimental/DotPatternMap/images/clustered_images"
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
os.makedirs(output_dir, exist_ok=True)

for path, label in zip(image_paths, labels):
    image = cv2.imread(path)
    cluster_dir = os.path.join(output_dir, f"cluster_{label}")
    os.makedirs(cluster_dir, exist_ok=True)
    output_path = os.path.join(cluster_dir, os.path.basename(path))
    cv2.imwrite(output_path, image)

# クラスタごとの画像を一枚にまとめる
for cluster_id, cell_list in clusters.items():
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
    images = []
    for cell_id in cell_list:
        img_path = os.path.join(cluster_dir, f"{cell_id}.png")
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)

    if images:  # Check if images list is not empty
        # 画像の数に応じてグリッドサイズを決定
        grid_size = int(np.ceil(np.sqrt(len(images))))
        image_height, image_width, _ = images[0].shape
        combined_image = np.zeros(
            (grid_size * image_height, grid_size * image_width, 3), dtype=np.uint8
        )

        for idx, image in enumerate(images):
            row = idx // grid_size
            col = idx % grid_size
            combined_image[
                row * image_height : (row + 1) * image_height,
                col * image_width : (col + 1) * image_width,
            ] = image

        combined_image_path = os.path.join(
            output_dir, f"cluster_{cluster_id}_combined.png"
        )
        cv2.imwrite(combined_image_path, combined_image)
