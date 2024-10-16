from fastapi import APIRouter
from Dropbox.crud import DropboxCrud
import os


router_dropbox = APIRouter(prefix="/dropbox", tags=["dropbox"])


@router_dropbox.post("/upload")
async def upload_file(file_path: str, file_name: str):
    await DropboxCrud().upload_file(file_path, file_name)
    return {"message": f"File {file_path} uploaded successfully"}


@router_dropbox.get("/list")
async def list_files():
    return {"files": await DropboxCrud().list_files()}


@router_dropbox.post("/databases/backup")
async def backup_databases():
    file_names = [
        f"databases/{i}"
        for i in os.listdir("databases")
        if i.endswith(".db") and i != "test_database.db"
    ]
    await DropboxCrud().backup_databases(file_names)
    print(file_names)
    return {"message": file_names}


@router_dropbox.get("/connection_check")
async def connection_check():
    return {"status": await DropboxCrud().connection_check()}
