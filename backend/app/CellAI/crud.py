from __future__ import annotations
import cv2
import numpy as np
from numpy.linalg import eig, inv
from fastapi.responses import StreamingResponse
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
import cv2
import io
import numpy as np
from fastapi.responses import StreamingResponse
from typing import Any
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import cv2
from CellDBConsole.crud import CellCrudBase


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Output layer
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1_pooled = self.pool1(enc1)

        enc2 = self.enc2(enc1_pooled)
        enc2_pooled = self.pool2(enc2)

        enc3 = self.enc3(enc2_pooled)
        enc3_pooled = self.pool3(enc3)

        bottleneck = self.bottleneck(enc3_pooled)

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        out = self.out_conv(dec1)
        return self.sigmoid(out)


class AsyncChores:
    @staticmethod
    async def async_imdecode(data: bytes) -> np.ndarray:
        """
        Decode an image from bytes.

        Parameters:
        - data: Image data in bytes.

        Returns:
        - Image in numpy array format.
        """
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            img = await loop.run_in_executor(
                executor, cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
            )
        return img

    @staticmethod
    async def async_cv2_imencode(img) -> tuple[bool, np.ndarray]:
        """
        Encode an image to PNG format.

        Parameters:
        - img: Image to encode.

        Returns:
        - Tuple containing success status and image buffer.
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            success, buffer = await loop.run_in_executor(
                executor, lambda: cv2.imencode(".png", img)
            )
        return success, buffer


class CellAiCrudBase:

    def __init__(self, db_name: str, model_path: str = "T1"):
        self.db_name = db_name
        self.model = UNet()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model_path = "CellAI/models/T1.pth" if model_path == "T1" else model_path
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    async def predict_contour(
        self,
        cell_id: str,
    ) -> StreamingResponse:
        """
        Predict the contour of a cell.

        Parameters:
        - data: Image data in bytes.

        Returns:
        - Predicted contour image as a StreamingResponse.
        """
        cell = await CellCrudBase(
            self.db_name,
        ).read_cell(cell_id)

        img = await AsyncChores.async_imdecode(cell.img_ph)

        img_resized = cv2.resize(img, (256, 256)) / 255.0
        img_tensor = (
            torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            prediction = self.model(img_tensor)
        prediction = (prediction > 0.5).cpu().numpy().astype(np.uint8) * 255

        prediction_image = Image.fromarray(prediction[0][0])
        buffer = io.BytesIO()
        prediction_image.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    async def predict_contour_draw(self, cell_id: str) -> list[list[float]]:
        """
        Predict the contour of a cell and return the contour image.

        Parameters:
        - cell_id: ID of the cell to predict.

        Returns:
        - Predicted contour image as a StreamingResponse.
        """
        cell = await CellCrudBase(self.db_name).read_cell(cell_id)
        img = await AsyncChores.async_imdecode(cell.img_ph)

        img_resized = cv2.resize(img, (256, 256)) / 255.0
        img_tensor = (
            torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            prediction = self.model(img_tensor)

        prediction = (prediction > 0.5).cpu().numpy().astype(np.uint8) * 255
        prediction = prediction[0][0]
        # resize image to original size
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]))
        edges = cv2.Canny(prediction, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError("No contours found")

        image_center = (edges.shape[1] // 2, edges.shape[0] // 2)

        min_distance = float("inf")
        best_contour = None

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distance = np.sqrt(
                    (cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    best_contour = contour
        if best_contour is None:
            raise ValueError("No valid contours found")

        return best_contour.squeeze().tolist()
