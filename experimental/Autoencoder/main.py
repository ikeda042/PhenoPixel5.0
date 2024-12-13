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
    masked[mask > 0] = 255
    return img_ph, masked


cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == 1).all()


class CellDataset(Dataset):
    def __init__(self, cells):
        self.cells = cells

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        cell = self.cells[idx]
        img_ph, masked = parse_image(cell)
        img_ph = cv2.resize(img_ph, (256, 256)) / 255.0
        masked = cv2.resize(masked, (256, 256)) / 255.0
        img_ph = torch.tensor(img_ph.transpose(2, 0, 1), dtype=torch.float32)
        masked = torch.tensor(masked[:, :, 0], dtype=torch.float32).unsqueeze(0)
        return img_ph, masked


print(cells_with_label_1)

import matplotlib.pyplot as plt


def create_histogram(
    data: list[int],
    num_bins: int = 256,
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
):
    """
    0-255の範囲で整数のリストからヒストグラムを作成する関数

    Parameters:
        data (list[int]): ヒストグラム化するデータ（0-255の整数）
        num_bins (int): ビンの数（デフォルト: 256）
        title (str): グラフのタイトル（デフォルト: "Histogram"）
        xlabel (str): x軸のラベル（デフォルト: "Value"）
        ylabel (str): y軸のラベル（デフォルト: "Frequency"）
    """
    plt.figure(figsize=(12, 6))

    # ビンの範囲を0-255に固定
    bins = np.linspace(0, 255, num_bins + 1)

    # ヒストグラムを作成
    plt.hist(data, bins=bins, range=(0, 255), edgecolor="black", color="skyblue")

    # グラフの装飾
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(True, alpha=0.3)

    # x軸の範囲を0-255に設定
    plt.xlim(0, 255)

    # グラフを表示
    plt.show()


data = np.random.randint(0, 100, 100)
create_histogram(data)
