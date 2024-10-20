from fastapi import APIRouter, Security
from fastapi.exceptions import HTTPException
from Auth.crud import Auth
import os
from CellDBConsole.crud import CellCrudBase

router_admin = APIRouter(prefix="/admin", tags=["admin"])


@router_admin.delete("/database/{db_name}")
async def delete_database(db_name: str, account: str = Security(Auth.get_account)):
    if not os.path.exists(f"databases/{db_name}"):
        raise HTTPException(status_code=404, detail="File not found")
    if db_name == "test_database.db":
        raise HTTPException(status_code=400, detail="Cannot delete the test database.")
    return await CellCrudBase(db_name=db_name).delete_database()
