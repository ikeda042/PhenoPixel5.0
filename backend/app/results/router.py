from fastapi import APIRouter
from results.crud import ResultsCRUD
from fastapi.responses import FileResponse


router_results = APIRouter(prefix="/results", tags=["results"])


@router_results.get("")
async def get_files():
    return await ResultsCRUD.read_result_files()


@router_results.get("/{file_name}")
async def get_file(file_name: str):
    return FileResponse(f"results/{file_name}")
