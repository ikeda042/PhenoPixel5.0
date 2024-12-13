import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Union, Tuple
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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set()

# ディレクトリを空にする
dir_path = "experimental/Autoencoder/images/infer_data/"
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)

reconstructed_save_path = "experimental/Autoencoder/images/infer_data_reconstructed/"
if os.path.exists(reconstructed_save_path):
    shutil.rmtree(reconstructed_save_path)

os.makedirs(reconstructed_save_path, exist_ok=True)
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


dbname = "sk394_bta_19.db"
dbpath = "sqlite:///experimental/Autoencoder/" + dbname
engine = create_engine(dbpath)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def box_plot_function(
    data: list[np.ndarray] | list[float | int],
    labels: list[str],
    xlabel: str,
    ylabel: str,
    save_name: str,
) -> None:
    fig = plt.figure(figsize=[10, 7])
    plt.boxplot(data, sym="")
    for i, d in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(d))
        plt.plot(x, d, "o", alpha=0.5)
    plt.xticks([i + 1 for i in range(len(data))], [f"{i}" for i in labels])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    fig.savefig(f"{save_name}.png", dpi=500)


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


cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == "1").all()
print(len(cells_with_label_1))
for i, cell in enumerate(cells_with_label_1):
    _, masked = parse_image(cell)
    cv2.imwrite(
        f"experimental/Autoencoder/images/infer_data/{cell.cell_id}.png", masked
    )

cell_ids = sorted([cell.cell_id for cell in cells_with_label_1])
images = [
    cv2.imread(f"experimental/Autoencoder/images/infer_data/{cell_id}.png")
    for cell_id in cell_ids
]
combined_image = combine_images_grid(images, 10)
cv2.imwrite(
    "experimental/Autoencoder/images/infer_dataset_combined.png", combined_image
)


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


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """画像の前処理を行う関数

    Args:
        image (np.ndarray): 入力画像 (BGR形式)

    Returns:
        torch.Tensor: 前処理済みの画像テンソル (1, C, H, W)
    """
    # グレースケールに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # [H, W] -> [H, W, 1]に変換
    image = np.expand_dims(image, axis=-1)
    # [0,1]に正規化
    image = image.astype(np.float32) / 255.0
    # CHWに変換
    image = np.transpose(image, (2, 0, 1))
    # バッチ次元を追加
    image = np.expand_dims(image, axis=0)
    # Tensorに変換
    return torch.from_numpy(image)


def calculate_reconstruction_score(
    model: nn.Module, image: Union[str, np.ndarray], device: torch.device = None
) -> Tuple[float, np.ndarray]:
    """画像の再構築スコア(MSE)を計算する

    Args:
        model (nn.Module): 学習済みAutoencoder
        image (Union[str, np.ndarray]): 入力画像のパスまたはnumpy配列
        device (torch.device, optional): 使用デバイス. Defaults to None.

    Returns:
        Tuple[float, np.ndarray]: (MSEスコア, 再構築画像)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 画像の読み込み
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError("Failed to load image")

    # 前処理
    x = preprocess_image(image)
    x = x.to(device)

    # モデルを評価モードに設定
    model.eval()

    # 推論
    with torch.no_grad():
        output = model(x)

    # MSEの計算
    mse = nn.MSELoss()(output, x).item()

    # 再構築画像の変換 (表示用)
    reconstructed = output.cpu().numpy()[0]  # (C,H,W)
    reconstructed = np.transpose(reconstructed, (1, 2, 0))  # (H,W,C)
    reconstructed = (reconstructed * 255).astype(np.uint8)
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR)

    return mse, reconstructed


def process_image_with_autoencoder(
    image_path: str, model_path: str, device: str = "mps"
) -> Tuple[float, str]:
    """画像を読み込み、Autoencoderで処理して結果を保存する関数

    Args:
        image_path (str): 入力画像のパス
        model_path (str): モデルの重みファイルのパス
        device (str, optional): 使用するデバイス. Defaults to "mps".

    Returns:
        Tuple[float, str]: (MSEスコア, 保存された再構築画像のパス)
    """
    try:
        # デバイスの設定
        device = torch.device(device)

        # モデルの初期化と重みの読み込み
        model = Autoencoder().to(device)
        model.load_state_dict(torch.load(model_path))

        # スコアの計算
        mse_score, reconstructed_image = calculate_reconstruction_score(
            model, image_path, device
        )

        # 保存するファイル名の生成（元の画像名の末尾に_AEを追加）
        base_name = image_path.rsplit(".", 1)[0]
        print(base_name)
        image_name = base_name.split("/")[-1]
        save_path = (
            f"experimental/Autoencoder/images/infer_data_reconstructed/{image_name}.png"
        )

        # 再構築画像の保存
        cv2.imwrite(save_path, reconstructed_image)

        return mse_score, save_path

    except Exception as e:
        print(f"Error occurred while processing {image_path}: {str(e)}")
        return None, None


if __name__ == "__main__":
    model_path = "experimental/Autoencoder/AE.pth"
    image_names = sorted(os.listdir("experimental/Autoencoder/images/infer_data/"))
    scores = []
    for image_name in image_names:
        image_path = f"experimental/Autoencoder/images/infer_data/{image_name}"
        mse_score, save_path = process_image_with_autoencoder(image_path, model_path)
        print(f"{image_name}: MSE={mse_score:.4f}, saved as {save_path}")
        scores.append(mse_score)
        print("")
    resonstructed_images = [
        cv2.imread(
            f"experimental/Autoencoder/images/infer_data_reconstructed/{image_name}"
        )
        for image_name in image_names
    ]
    combined_image = combine_images_grid(resonstructed_images, 10)
    cv2.imwrite(
        "experimental/Autoencoder/images/infer_reconstructed_combined.png",
        combined_image,
    )
    print("Reconstructed images are saved as infer_reconstructed_combined.png")

    # トレーニングデータの再構築スコアを取得
    train_data_path = "experimental/Autoencoder/images/train_data/"
    train_data_names = sorted(os.listdir(train_data_path))
    train_data_scores = []
    for train_data_name in train_data_names:
        train_data_path = (
            f"experimental/Autoencoder/images/train_data/{train_data_name}"
        )
        mse_score, _ = process_image_with_autoencoder(train_data_path, model_path)
        train_data_scores.append(mse_score)
        print(f"{train_data_name}: MSE={mse_score:.4f}")
        print("")
    print("Mean MSE score of training data:", np.mean(train_data_scores))
    print("Mean MSE score of inference data:", np.mean(scores))
    print("Inference data MSE scores:", scores)
    print("Training data MSE scores:", train_data_scores)
    print("Done")

    data = [train_data_scores, scores]
    labels = ["Training data", "Inference data"]
    xlabel = "Data type"
    ylabel = "MSE"
    save_name = "result"
    box_plot_function(data, labels, xlabel, ylabel, save_name)
