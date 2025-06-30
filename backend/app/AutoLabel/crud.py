from pathlib import Path

import cv2
import numpy as np
from CellDBConsole.crud import CellCrudBase

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

    async def autolabel(self) -> None:
        """Label cells with elongated bounding boxes as label "1"."""
        na_ids = await CellCrudBase(self.db_name).read_cell_ids(label="N/A")
        for cell in na_ids:
            contour = await CellCrudBase(self.db_name).get_cell_contour(cell.cell_id)
            np_contour = np.array(contour, dtype=np.float32)
            ratio = self.contour_aspect_ratio(np_contour)
            if ratio <= 0.5:
                await CellCrudBase(self.db_name).update_label(cell.cell_id, "1")
