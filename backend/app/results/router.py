from fastapi import APIRouter
from results.crud import ResultsCRUD
from typing import Literal


router_results = APIRouter(prefix="/results", tags=["results"])


@router_results.get("")
async def get_files():
    return await ResultsCRUD.read_result_files()
