from fastapi import APIRouter
from AutoLabel.crud import AutoLabelCrud

router_autolabel = APIRouter(prefix="/autolabel", tags=["autolabel"])

@router_autolabel.post("/{db_name}")
async def autolabel_db(db_name: str):
    await AutoLabelCrud(db_name).autolabel()
    return {"status": "ok"}


