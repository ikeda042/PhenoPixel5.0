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
import shutil

# ディレクトリを空にする
dir_path = "experimental/Autoencoder/images/train_data/"
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)

os.makedirs(dir_path, exist_ok=True)
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
    user_id = Column(String, nullable=True)


dbpath = "sqlite:///experimental/Autoencoder/sk25_pro_FITC_c.db"
engine = create_engine(dbpath)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def parse_image(cell: Cell) -> tuple:
    img_ph = cv2.imdecode(np.frombuffer(cell.img_ph, np.uint8), cv2.IMREAD_GRAYSCALE)
    contour = pickle.loads(cell.contour)
    img_fluo1 = cv2.imdecode(
        np.frombuffer(cell.img_fluo1, np.uint8), cv2.IMREAD_GRAYSCALE
    )
    mask = np.zeros_like(img_ph)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    masked = cv2.bitwise_and(img_fluo1, mask)
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
    cv2.imwrite(
        f"experimental/Autoencoder/images/train_data/{cell.cell_id}.png", masked
    )

cell_ids = sorted([cell.cell_id for cell in cells_with_label_1])

images = [
    cv2.imread(f"experimental/Autoencoder/images/train_data/{cell_id}.png")
    for cell_id in cell_ids
]
combined_image = combine_images_grid(images, 8)
cv2.imwrite(
    "experimental/Autoencoder/images/train_dataset_combined.png", combined_image
)


class MaskedCellDataset(Dataset):
    def __init__(self, cells):
        self.cells = cells
        self.data = []
        for cell in self.cells:
            _, masked = parse_image(cell)
            masked = masked.astype(np.float32) / 255.0  # 正規化
            masked = np.expand_dims(masked, axis=0)  # チャンネル次元を追加
            self.data.append(masked)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x


dataset = MaskedCellDataset(cells_with_label_1)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
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
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
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


torch.save(model.state_dict(), "experimental/Autoencoder/AE2.pth")
