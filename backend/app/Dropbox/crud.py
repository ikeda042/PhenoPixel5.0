from dotenv import load_dotenv
import aiohttp
import aiofiles
import asyncio
from dropbox import Dropbox
import os

load_dotenv()
APP_KEY = os.getenv("DROPBOX_APP_KEY")
APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
ACCESS_CODE = os.getenv("DROPBOX_ACCESS_CODE")


class DropboxCrud:
    app_key: str = APP_KEY
    app_secret: str = APP_SECRET
    access_code: str = ACCESS_CODE

    @classmethod
    async def get_access_token(cls) -> str:
        auth_url = "https://api.dropboxapi.com/oauth2/token"
        data = {
            "grant_type": "authorization_code",
            "code": cls.access_code,
        }

        auth = aiohttp.BasicAuth(login=cls.app_key, password=cls.app_secret)

        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, data=data, auth=auth) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return response_data["access_token"]
                else:
                    raise Exception("Failed to get access token")

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

    async def connection_check(self) -> bool:
        if "databases" not in await self.list_files():
            return False
        return True

    async def backup_databases(self, file_names: list[str]):
        for file_name in file_names:
            await self.upload_file(f"{file_name}", file_name)
