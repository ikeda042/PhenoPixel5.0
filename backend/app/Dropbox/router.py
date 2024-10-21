from fastapi import APIRouter, Security
from Dropbox.crud import DropboxCrud
import os
from Auth.crud import Auth


router_dropbox = APIRouter(prefix="/dropbox", tags=["dropbox"])


@router_dropbox.post("/upload")
async def upload_file(file_path: str, file_name: str):
    await DropboxCrud().upload_file(file_path, file_name)
    return {"message": f"File {file_path} uploaded successfully"}


@router_dropbox.get("/list")
async def list_files():
    return {"files": await DropboxCrud().list_files()}


@router_dropbox.get("/list_databases")
async def list_databases():
    return {"files": await DropboxCrud().list_databases()}


@router_dropbox.post("/databases/backup")
async def backup_databases():
    file_names = [
        f"databases/{i}"
        for i in os.listdir("databases")
        if i.endswith(".db") and i != "test_database.db" and "-uploaded.db" not in i
    ]
    await DropboxCrud().backup_databases(file_names)
    print(file_names)
    return {"message": file_names}


@router_dropbox.get("/connection_check")
async def connection_check():
    return {"status": await DropboxCrud().connection_check()}


@router_dropbox.get("/access_token")
async def get_access_token(account: str = Security(Auth.get_account)):
    return {"access_token": await DropboxCrud.get_access_token()}


@router_dropbox.post("/download")
async def download_file(file_name: str):
    await DropboxCrud().download_file(file_name)
    return {"message": f"File {file_name} downloaded successfully"}
