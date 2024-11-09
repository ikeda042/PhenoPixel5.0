import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, FLOAT, BLOB

Base = declarative_base()


class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)


def parse_image(cell: Cell) -> tuple:
    img_fluo = cv2.imdecode(np.frombuffer(cell.img_fluo1, np.uint8), cv2.IMREAD_COLOR)
    contour = pickle.loads(cell.contour)
    mask = np.zeros_like(img_fluo)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img_fluo, mask)
    masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(masked_gray)
    if max_val > min_val:
        normalized = cv2.normalize(
            masked, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
    else:
        normalized = masked
    return img_fluo, normalized


dbpath = "sqlite:///experimental/CharVectorMapping/sk326Gen120min.db"
engine = create_engine(dbpath)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == 1).all()

for cell in cells_with_label_1:
    img_fluo, masked = parse_image(cell)
    cv2.imwrite(
        f"experimental/CharVectorMapping/images/fluo/{cell.cell_id}.png", img_fluo
    )
    cv2.imwrite(
        f"experimental/CharVectorMapping/images/fluo_masked/{cell.cell_id}.png",
        masked,
    )

import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

# fluo_maskedフォルダにある画像パスを取得
image_folder = "experimental/CharVectorMapping/images/fluo_masked"
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.endswith(".png")
]

# 画像読み込みと変換用
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 必要に応じて正規化の値を調整
    ]
)


# カスタムデータセットの作成
class CustomFluoDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # グレースケールで読み込み
        image = cv2.resize(image, (200, 200))  # 必要に応じてサイズを調整
        if self.transform:
            image = self.transform(image)
        return image, img_path


dataset = CustomFluoDataset(image_paths, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# 事前学習済みモデルを使って特徴抽出
model = models.resnet18(pretrained=True)
model.fc = nn.Identity()  # 最終全結合層を無効化して中間層の出力を取得
model.eval()

features = []
file_names = []

# 特徴抽出
with torch.no_grad():
    for images, paths in dataloader:
        images = images.expand(-1, 3, -1, -1)  # グレースケールをRGBに拡張
        output = model(images)
        features.append(output.numpy())
        file_names.extend(paths)

# 特徴を一つの配列に結合
features = np.concatenate(features, axis=0)

# 次元削減 (PCA)
pca = PCA(n_components=3)  # 2次元または3次元に変更可能
reduced_features = pca.fit_transform(features)

# 次元削減結果の可視化 (2Dプロット)
plt.figure(figsize=(10, 7))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
plt.title("PCA Visualization of Extracted Features (2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# 次元削減結果の可視化 (3Dプロット)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], alpha=0.7
)
ax.set_title("PCA Visualization of Extracted Features (3D)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.show()
