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
    mask = np.zeros_like(img_ph)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img_ph, mask)
    # maskedエリアを白(255)で塗りつぶす
    masked[mask > 0] = 255
    return img_ph, masked


cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == 1).all()


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
epochs = 10  # 適宜変更
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

# 学習が完了したら、モデルを保存
torch.save(model.state_dict(), "autoencoder.pth")


# --- 異常判定フェーズ ---
# 新規画像をモデルに入力し、そのMSEを計算
def infer_anomaly(img_path, model, threshold):
    # 画像読み込みと前処理
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img).unsqueeze(0).to(device)  # [1,C,H,W]

    model.eval()
    with torch.no_grad():
        recon = model(img_tensor)
    mse = criterion(recon, img_tensor).item()

    if mse > threshold:
        print("Anomaly detected. MSE:", mse)
    else:
        print("Normal. MSE:", mse)


# 上記thresholdは学習データのMSE分布を見て適宜決める。
# 例：学習データ中の再構築誤差（MSE）を集計し、その平均+2σなどを閾値として用いる。

# 閾値の決め方例（学習データ中で実行）
model.eval()
with torch.no_grad():
    mse_list = []
    for i in range(len(dataset)):
        img, _ = dataset[i]
        img_tensor = torch.tensor(img).unsqueeze(0).to(device)
        recon = model(img_tensor)
        mse_val = criterion(recon, img_tensor).item()
        mse_list.append(mse_val)
threshold = np.mean(mse_list) + 2 * np.std(mse_list)
print("Threshold:", threshold)

# # テスト用画像で異常判定を行う
# test_image_path = "path/to/your/test_image.png"
# infer_anomaly(test_image_path, model, threshold)
