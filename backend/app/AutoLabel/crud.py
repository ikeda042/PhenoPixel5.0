import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import cv2

import joblib
import numpy as np
from CellDBConsole.crud import CellCrudBase

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = "AutoLabel/svm_cell_classifier.pkl"
TARGET_LEN = 256


class AutoLabelCrud:
    def __init__(self, db_name: str, model_path: Path = MODEL_PATH):
        self.db_name = db_name
        self.model = joblib.load(str(model_path))

    @staticmethod
    def resample_vector(vec: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
        vec = np.asarray(vec).reshape(-1)
        if len(vec) == target_len:
            return vec
        old_x = np.linspace(0.0, 1.0, len(vec))
        new_x = np.linspace(0.0, 1.0, target_len)
        return np.interp(new_x, old_x, vec)

    @staticmethod
    def contour_to_features(contour: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
        center = contour.mean(axis=0)
        dist_vec = np.linalg.norm(contour - center, axis=1)
        profile = AutoLabelCrud.resample_vector(np.sort(dist_vec), target_len)
        hu = cv2.HuMoments(cv2.moments(contour.astype(np.float32))).flatten()
        return np.hstack([profile, hu])

    async def predict_label(self, contour: List[List[float]]) -> str:
        """Predict label for a contour using the loaded SVM model."""
        np_contour = np.array(contour)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            features = await loop.run_in_executor(
                executor, self.contour_to_features, np_contour
            )
            label = await loop.run_in_executor(
                executor,
                self.model.predict,
                features.reshape(1, -1),
            )
        return str(label[0])

    async def autolabel(self) -> None:
        """Automatically label all cells marked as "N/A" using the SVM model."""
        na_ids = await CellCrudBase(self.db_name).read_cell_ids(label="N/A")
        for cell in na_ids:
            contour = await CellCrudBase(self.db_name).get_cell_contour(cell.cell_id)
            label = await self.predict_label(contour)
            await CellCrudBase(self.db_name).update_label(cell.cell_id, label)
