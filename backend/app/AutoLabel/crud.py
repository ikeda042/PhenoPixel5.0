from pathlib import Path

import cv2
import numpy as np
from CellDBConsole.crud import CellCrudBase, AsyncChores
from CellAI.crud import CellAiCrudBase

ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET_LEN = 256


class AutoLabelCrud:
    def __init__(self, db_name: str, model_path: Path = ROOT_DIR):
        self.db_name = db_name

    @staticmethod
    def contour_aspect_ratio(contour: np.ndarray) -> float:
        """Return aspect ratio (short side / long side) of contour bounding box."""
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        long_side = max(w, h)
        short_side = min(w, h)
        if long_side == 0:
            return 0.0
        return short_side / long_side

    @staticmethod
    def contour_iou(
        contour_a: np.ndarray, contour_b: np.ndarray, shape: tuple[int, int]
    ) -> float:
        mask_a = np.zeros(shape, dtype=np.uint8)
        mask_b = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask_a, [contour_a.astype(np.int32)], 1)
        cv2.fillPoly(mask_b, [contour_b.astype(np.int32)], 1)
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return intersection / union

    async def autolabel(self) -> None:
        """Label cells comparing contours with the T1 U-Net model."""
        na_ids = await CellCrudBase(self.db_name).read_cell_ids(label="N/A")
        cell_ai = CellAiCrudBase(self.db_name, model_path="T1")

        for cell in na_ids:
            actual_contour = await CellCrudBase(self.db_name).get_cell_contour(
                cell.cell_id
            )
            predicted_contour = await cell_ai.predict_contour_draw(cell.cell_id)

            actual_np = np.array(actual_contour, dtype=np.float32)
            predicted_np = np.array(predicted_contour, dtype=np.float32)

            cell_obj = await CellCrudBase(self.db_name).read_cell(cell.cell_id)
            img = await AsyncChores.async_imdecode(cell_obj.img_ph)
            shape = img.shape[:2]

            iou = self.contour_iou(actual_np, predicted_np, shape)

            if iou >= 0.4:
                await CellCrudBase(self.db_name).update_label(cell.cell_id, "1")
            else:
                await CellCrudBase(self.db_name).update_label(cell.cell_id, "N/A")
