import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Union, Tuple


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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
            nn.Sigmoid(),
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
    # BGRからRGBに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        save_path = f"{base_name}_AE.png"

        # 再構築画像の保存
        cv2.imwrite(save_path, reconstructed_image)

        return mse_score, save_path

    except Exception as e:
        print(f"Error occurred while processing {image_path}: {str(e)}")
        return None, None


if __name__ == "__main__":
    model_path = "experimental/Autoencoder/AE.pth"
    image_path = "sample_image.jpg"

    mse_score, saved_path = process_image_with_autoencoder(
        image_path=image_path, model_path=model_path
    )

    if mse_score is not None:
        print(f"Reconstruction MSE Score: {mse_score:.6f}")
        print(f"Reconstructed image saved to: {saved_path}")