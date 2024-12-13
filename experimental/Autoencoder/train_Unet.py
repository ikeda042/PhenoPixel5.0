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


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32)

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(32, 16)

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool(x1)

        x3 = self.enc2(x2)
        x4 = self.pool(x3)

        x5 = self.enc3(x4)
        x6 = self.pool(x5)

        # Bottleneck
        x_bottleneck = self.bottleneck(x6)

        # Decoder
        x_up1 = self.up1(x_bottleneck)
        x_cat1 = torch.cat([x_up1, x5], dim=1)
        x_dec1 = self.dec1(x_cat1)

        x_up2 = self.up2(x_dec1)
        x_cat2 = torch.cat([x_up2, x3], dim=1)
        x_dec2 = self.dec2(x_cat2)

        x_up3 = self.up3(x_dec2)
        x_cat3 = torch.cat([x_up3, x1], dim=1)
        x_dec3 = self.dec3(x_cat3)

        out = self.out_conv(x_dec3)
        out = self.sigmoid(out)

        return out


if __name__ == "__main__":
    cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == 1).all()
    dataset = MaskedCellDataset(cells_with_label_1)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # M1 mac gpt
    device = torch.device("mps")
    model = UNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- 学習ループ ---
    epochs = 15  # 適宜変更
    model.train()
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
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

    torch.save(model.state_dict(), "experimental/Autoencoder/UNet.pth")
