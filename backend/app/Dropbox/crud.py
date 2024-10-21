from dotenv import load_dotenv
import aiohttp
import aiofiles
import asyncio
from dropbox import Dropbox
import os

load_dotenv()
REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
CLIEND_ID = os.getenv("DROPBOX_APP_KEY")
CLIENT_SECRET = os.getenv("DROPBOX_APP_SECRET")


class DropboxCrud:
    client_id: str = CLIEND_ID
    refresh_token: str = REFRESH_TOKEN
    client_secret: str = CLIENT_SECRET

    @classmethod
    async def get_access_token(cls) -> str:
        auth_url = "https://api.dropboxapi.com/oauth2/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": cls.refresh_token,
            "client_id": cls.client_id,
            "client_secret": cls.client_secret,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, data=data) as response:
                response_data = await response.json()
                if response.status == 200:
                    return response_data["access_token"]
                else:
                    raise Exception(
                        f"Failed to refresh access token: {response.status}, {response_data}"
                    )

    async def upload_file(self, file_path: str, file_name: str) -> str:
        async with aiofiles.open(file_path, mode="rb") as f:
            data = await f.read()
        async with aiohttp.ClientSession() as session:
            await session.post(
                "https://content.dropboxapi.com/2/files/upload",
                headers={
                    "Authorization": f"Bearer {await DropboxCrud.get_access_token()}",
                    "Dropbox-API-Arg": f'{{"path": "/PhenoPixelDatabases/{file_name}", "mode": "add", "autorename": true, "mute": false, "strict_conflict": false}}',
                    "Content-Type": "application/octet-stream",
                },
                data=data,
            )
        return f"File {file_name} uploaded successfully"

    async def list_files(self) -> list:
        dbx = Dropbox(await DropboxCrud.get_access_token())
        try:
            response = await asyncio.to_thread(
                dbx.files_list_folder, "/PhenoPixelDatabases"
            )
            return [file.name for file in response.entries]
        except Exception as e:
            return []

    async def list_databases(self) -> list:
        dbx = Dropbox(await DropboxCrud.get_access_token())
        try:
            response = await asyncio.to_thread(
                dbx.files_list_folder, "/PhenoPixelDatabases/databases"
            )
            return [file.name for file in response.entries]
        except Exception as e:
            return []

    async def connection_check(self) -> bool:
        if "databases" not in await self.list_files():
            return False
        return True

    async def backup_databases(self, file_names: list[str]):
        for file_name in file_names:
            await self.upload_file(f"{file_name}", file_name)

    async def download_file(self, file_name: str) -> str:
        dbx = Dropbox(await DropboxCrud.get_access_token())
        try:
            files = await self.list_databases()
            print("Downloading...")
            print("#################################################################")
            print(files)
            if file_name not in files:
                raise Exception(f"File {file_name} not found")
            local_file_path = f"databases/{file_name}"
            dropbox_file_path = f"/PhenoPixelDatabases/databases/{file_name}"
            await asyncio.to_thread(
                dbx.files_download_to_file, local_file_path, dropbox_file_path
            )
            return f"File {file_name} downloaded successfully"
        except Exception as e:
            return f"Failed to download file {file_name}: {e}"
