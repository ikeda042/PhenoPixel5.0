import cv2
import numpy as np
import os
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

# VGG16モデルを使用して特徴量を抽出
def extract_deep_features(image):
    # 画像をVGG16に合う形に変換 (224x224x3)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)  # バッチ次元の追加
    image = preprocess_input(image)  # 前処理
    
    features = model.predict(image)
    return features.flatten()

# VGG16モデルの読み込み（全結合層を除いたもの）
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

# fluo_maskedフォルダにある画像パスを取得
image_folder = "experimental/CharVectorMapping/images/fluo_masked"
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.endswith(".png")
]

# 画像の読み込みと特徴抽出
features = []
file_names = []
for img_path in image_paths:
    image = cv2.imread(img_path)  # カラーモードで読み込み
    if image is None:
        continue  # 画像が読み込めない場合はスキップ
    deep_feature = extract_deep_features(image)
    features.append(deep_feature)
    file_names.append(img_path)

# 特徴量を一つの配列に結合
features = np.array(features)

# 次元削減 (PCA)
pca = PCA(n_components=3)  # 2次元または3次元に変更可能
reduced_features = pca.fit_transform(features)

# 次元削減結果の可視化 (2Dプロット)
plt.figure(figsize=(10, 7))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
plt.title("PCA Visualization of Deep Features (2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# 次元削減結果の可視化 (3Dプロット)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], alpha=0.7
)
ax.set_title("PCA Visualization of Deep Features (3D)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.show()
