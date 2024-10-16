from fastapi import APIRouter
from Dropbox.crud import DropboxCrud


router_dropbox = APIRouter(prefix="/dropbox", tags=["dropbox"])


@router_dropbox.post("/upload")
async def upload_file(file_path: str, file_name: str):
    await DropboxCrud().upload_file(file_path, file_name)
    return {"message": f"File {file_path} uploaded successfully"}


@router_dropbox.get("/list")
async def list_files():
    return {"files": await DropboxCrud().list_files()}


@router_dropbox.post("/databases/backup")
async def backup_databases(file_names: list[str]):
    await DropboxCrud().backup_databases(file_names)
    return {"message": "Backup completed"}
