from __future__ import annotations

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from fastapi.responses import StreamingResponse
from PIL import Image

from CellDBConsole.crud import CellCrudBase
from CellAI.schemas import UNet


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
