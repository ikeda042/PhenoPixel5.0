from fastapi import APIRouter
from CellAI.crud import CellAiCrudBase
from typing import Literal


router_cell_ai = APIRouter(prefix="/cell_ai", tags=["cell_ai"])


@router_cell_ai.get("/{db_name}/{cell_id}")
async def get_predicted_contour(
    db_name: str, cell_id: str, model: Literal["T1"] = "T1"
):
    return await CellAiCrudBase(db_name, model_path=model).predict_contour(cell_id)
