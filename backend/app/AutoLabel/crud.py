import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import joblib
import numpy as np
from CellDBConsole.crud import CellCrudBase

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = "AutoLabel/svm_cell_classifier.pkl"


class AutoLabelCrud:
    def __init__(self, db_name: str, model_path: Path = MODEL_PATH):
        self.db_name = db_name
        self.model = joblib.load(str(model_path))

    @staticmethod
    def contour_to_vector(contour: np.ndarray) -> np.ndarray:
        center = contour.mean(axis=0)
        distances = np.linalg.norm(contour - center, axis=1)
        return np.sort(distances)

    @staticmethod
    def resample_vector(vec: np.ndarray, target_len: int = 256) -> np.ndarray:
        vec = np.asarray(vec).reshape(-1)
        if len(vec) == target_len:
            return vec
        old_x = np.linspace(0.0, 1.0, len(vec))
        new_x = np.linspace(0.0, 1.0, target_len)
        return np.interp(new_x, old_x, vec)

    async def predict_label(self, contour: List[List[float]]) -> str:
        """Predict label for a contour using the loaded SVM model."""
        np_contour = np.array(contour)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            vec = await loop.run_in_executor(
                executor, self.contour_to_vector, np_contour
            )
            resampled = await loop.run_in_executor(
                executor, self.resample_vector, vec
            )
            label = await loop.run_in_executor(
                executor,
                self.model.predict,
                resampled.reshape(1, -1),
            )
        return str(label[0])

    async def autolabel(self) -> None:
        """Automatically label all cells marked as "N/A" using the SVM model."""
        na_ids = await CellCrudBase(self.db_name).read_cell_ids(label="N/A")
        for cell in na_ids:
            contour = await CellCrudBase(self.db_name).get_cell_contour(cell.cell_id)
            label = await self.predict_label(contour)
            await CellCrudBase(self.db_name).update_label(cell.cell_id, label)
