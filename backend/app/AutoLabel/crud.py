import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

import joblib
import numpy as np

from CellDBConsole.crud import CellCrudBase

MODEL_PATH = "experimental/autolabel/autolabel_lda.joblib"
THRESHOLD = 5.0


class AutoLabelCrud:
    def __init__(self, db_name: str, model_path: str = MODEL_PATH):
        self.db_name = db_name
        self.model = joblib.load(model_path)

    @staticmethod
    def contour_to_vector(contour: np.ndarray) -> np.ndarray:
        center = contour.mean(axis=0)
        distances = np.linalg.norm(contour - center, axis=1)
        return np.sort(distances)

    @staticmethod
    def resample_vector(vec: np.ndarray, target_len: int = 256) -> np.ndarray:
        if len(vec) == target_len:
            return vec
        old_x = np.linspace(0.0, 1.0, len(vec))
        new_x = np.linspace(0.0, 1.0, target_len)
        return np.interp(new_x, old_x, vec)

    async def embed_contour(self, contour: List[List[float]]) -> float:
        np_contour = np.array(contour)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            vec = await loop.run_in_executor(executor, self.contour_to_vector, np_contour)
            resampled = await loop.run_in_executor(executor, self.resample_vector, vec)
            emb = await loop.run_in_executor(executor, self.model.transform, resampled.reshape(1, -1))
        return float(emb[0, 0])

    async def autolabel(self) -> None:
        na_ids = await CellCrudBase(self.db_name).read_cell_ids(label="N/A")
        for cell in na_ids:
            contour = await CellCrudBase(self.db_name).get_cell_contour(cell.cell_id)
            emb = await self.embed_contour(contour)
            if emb <= THRESHOLD:
                await CellCrudBase(self.db_name).update_label(cell.cell_id, "1")
            else:
                await CellCrudBase(self.db_name).update_label(cell.cell_id, "N/A")

