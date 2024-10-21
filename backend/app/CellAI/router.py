from fastapi import APIRouter
from CellAI.crud import CellAiCrudBase
from typing import Literal


router_cell_ai = APIRouter(prefix="/cell_ai", tags=["cell_ai"])


@router_cell_ai.get("/{db_name}/{cell_id}")
async def get_predicted_contour(
    db_name: str, cell_id: str, model: Literal["T1", "T2", "T3"] = "T1"
):
    return await CellAiCrudBase(db_name, model_path=model).predict_contour(cell_id)


@router_cell_ai.get("/{db_name}/{cell_id}/plot_data")
async def get_plot_data(
    db_name: str, cell_id: str, model: Literal["T1", "T2", "T3"] = "T1"
):
    data: list[list[float]] = await CellAiCrudBase(
        db_name, model_path=model
    ).predict_contour_draw(cell_id)
    return data
