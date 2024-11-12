import os
import numpy as np
import cv2  # OpenCVをインポート
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 画像パスの設定
image_dir = "experimental/DotPatternMap/images/map64"
image_paths = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.endswith(".png")
]


# 特徴抽出関数（画像のピクセル値をそのまま使用）
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (64, 64))  # サイズ調整（必要に応じて）
    return image_resized.flatten()  # 平坦化して1次元配列にする


# すべての画像から特徴を抽出
features = []
cell_ids = []
for path in image_paths:
    cell_id = os.path.basename(path).split(".")[0]
    cell_ids.append(cell_id)
    features.append(extract_features(path))

features = np.array(features)

# 次元削減 (PCA)
pca = PCA(n_components=10)
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
    output_dir = "experimental/DotPatternMap/clustered_images"
    os.makedirs(output_dir, exist_ok=True)

    for path, label in zip(image_paths, labels):
        image = cv2.imread(path)
        output_path = os.path.join(
            output_dir, f"cluster{label}_{os.path.basename(path)}"
        )
        cv2.imwrite(output_path, image)
