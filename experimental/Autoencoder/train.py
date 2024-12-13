import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, BLOB
import os

os.makedirs("experimental/Autoencoder/images/fluo/", exist_ok=True)

Base = declarative_base()


class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(Float)
    area = Column(Float)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(Float)
    center_y = Column(Float)


dbpath = "sqlite:///experimental/Autoencoder/sk25_pro_FITC_c.db"
engine = create_engine(dbpath)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def parse_image(cell: Cell) -> tuple:
    img_ph = cv2.imdecode(np.frombuffer(cell.img_ph, np.uint8), cv2.IMREAD_COLOR)
    contour = pickle.loads(cell.contour)
    img_fluo1 = cv2.imdecode(np.frombuffer(cell.img_fluo1, np.uint8), cv2.IMREAD_COLOR)
    mask = np.zeros_like(img_ph)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img_fluo1, mask)
    # maskedエリアを白(255)で塗りつぶす
    return img_ph, masked


def combine_images_grid(images, grid_size):
    img_height, img_width, _ = images[0].shape
    combined_image = np.zeros(
        (img_height * grid_size, img_width * grid_size, 3), dtype=np.uint8
    )

    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        combined_image[
            row * img_height : (row + 1) * img_height,
            col * img_width : (col + 1) * img_width,
        ] = img

    return combined_image


cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == 1).all()
print(len(cells_with_label_1))
for i, cell in enumerate(cells_with_label_1):
    _, masked = parse_image(cell)
    cv2.imwrite(f"experimental/Autoencoder/images/fluo/{cell.cell_id}.png", masked)


images = [
    cv2.imread(f"experimental/Autoencoder/images/fluo/{cell.cell_id}.png")
    for cell in cells_with_label_1
]
combined_image = combine_images_grid(images, 8)

cv2.imwrite("experimental/Autoencoder/images/combined.png", combined_image)


# --- Datasetの定義 ---
class MaskedCellDataset(Dataset):
    def __init__(self, cells):
        self.cells = cells
        self.data = []
        for cell in self.cells:
            _, masked = parse_image(cell)
            # モデルに渡すための前処理
            # 例：RGBを[0,1]に正規化してCHWにtranspose
            masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
            masked = masked.astype(np.float32) / 255.0
            masked = np.transpose(masked, (2, 0, 1))  # CxHxW
            self.data.append(masked)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # 入力と出力は同じ(自己再構築)


dataset = MaskedCellDataset(cells_with_label_1)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# --- Autoencoderモデル定義例 ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # ここでは簡易的なConv Autoencoder
        # 入力：3xHxW（カラー画像）
        # 適宜画像サイズに応じてConv層を定義
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # 出力を0~1に抑える
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- 学習ループ ---
epochs = 15  # 適宜変更
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")


torch.save(model.state_dict(), "experimental/Autoencoder/AE.pth")
