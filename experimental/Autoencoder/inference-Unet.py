import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Union, Tuple
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, String, Float, BLOB
from sqlalchemy.orm import declarative_base, sessionmaker
import pickle
from train_Unet import UNet
import seaborn as sns

sns.set()


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
    fig.savefig(f"experimental/Autoencoder/{save_name}.png", dpi=500)


sns.set()

# Clear and recreate directories
dir_path = "experimental/Autoencoder/images/infer_data/"
reconstructed_save_path = "experimental/Autoencoder/images/infer_data_reconstructed/"
os.makedirs(dir_path, exist_ok=True)
os.makedirs(reconstructed_save_path, exist_ok=True)

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


def parse_image(cell: Cell) -> Tuple[np.ndarray, np.ndarray]:
    img_ph = cv2.imdecode(np.frombuffer(cell.img_ph, np.uint8), cv2.IMREAD_GRAYSCALE)
    contour = pickle.loads(cell.contour)
    img_fluo1 = cv2.imdecode(
        np.frombuffer(cell.img_fluo1, np.uint8), cv2.IMREAD_GRAYSCALE
    )
    mask = np.zeros_like(img_ph)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    masked = cv2.bitwise_and(img_fluo1, mask)
    return img_ph, masked


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = image.astype(np.float32) / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return torch.from_numpy(image)


def calculate_reconstruction_score(
    model: nn.Module, image: Union[str, np.ndarray], device: torch.device = None
) -> Tuple[float, np.ndarray]:
    if device is None:
        device = torch.device("mps")

    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError("Failed to load image")

    x = preprocess_image(image).to(device)
    model.eval()
    with torch.no_grad():
        output = model(x)

    mse = nn.MSELoss()(output, x).item()
    reconstructed = output.cpu().numpy()[0].transpose(1, 2, 0)
    reconstructed = (reconstructed * 255).astype(np.uint8)
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR)
    return mse, reconstructed


def process_image_with_unet(
    image_path: str, model_path: str, device: str = "mps"
) -> Tuple[float, str]:
    device = torch.device(device)
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path))

    mse_score, reconstructed_image = calculate_reconstruction_score(
        model, image_path, device
    )
    base_name = os.path.splitext(image_path)[0]
    save_path = f"experimental/Autoencoder/images/infer_data_reconstructed/{os.path.basename(base_name)}_reconstructed.png"
    cv2.imwrite(save_path, reconstructed_image)
    return mse_score, save_path


if __name__ == "__main__":
    model_path = "experimental/Autoencoder/UNet.pth"
    image_names = sorted(os.listdir("experimental/Autoencoder/images/infer_data/"))
    scores = []
    for image_name in image_names:
        image_path = os.path.join(
            "experimental/Autoencoder/images/infer_data", image_name
        )
        mse_score, save_path = process_image_with_unet(image_path, model_path)
        print(f"{image_name}: MSE={mse_score:.4f}, saved as {save_path}")
        scores.append(mse_score)

    # Combine reconstructed images
    reconstructed_images = [
        cv2.imread(
            f"experimental/Autoencoder/images/infer_data_reconstructed/{image_name}"
        )
        for image_name in image_names
    ]
    combined_image = combine_images_grid(reconstructed_images, 10)
    cv2.imwrite(
        "experimental/Autoencoder/images/infer_reconstructed_combined.png",
        combined_image,
    )
    print("Reconstructed images saved.")

    # Generate box plot
    labels = ["Inference Data"]
    box_plot_function([scores], labels, "Data Type", "MSE", "reconstruction_scores")
