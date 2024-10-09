import dropbox
import aiohttp
import aiofiles
import asyncio
import dropbox
from dropbox.files import WriteMode
from dotenv import load_dotenv
from dropbox import Dropbox
import os

load_dotenv()


class DropboxCrud:
    def __init__(
        self,
        token: str = (
            os.getenv("DROPBOX_ACCESS_TOKEN")
            if os.getenv("DROPBOX_ACCESS_TOKEN")
            else ""
        ),
    ):
        self.dbx: Dropbox = dropbox.Dropbox(token)

    async def upload_file(self, file_path: str, file_name: str) -> None:
        async with aiohttp.ClientSession() as session:
            async with aiofiles.open(file_path, mode="r") as f:
                await session.post(
                    "https://content.dropboxapi.com/2/files/upload",
                    headers={
                        "Authorization": f"Bearer {self.dbx._oauth2_access_token}",
                        "Dropbox-API-Arg": f'{{"path": "/PhenoPixelDatabases/{file_name}", "mode": "add", "autorename": true, "mute": false, "strict_conflict": false}}',
                        "Content-Type": "application/octet-stream",
                    },
                    data=await f.read(),
                )

    async def list_files(self) -> list:
        response = await self.dbx.files_list_folder("/PhenoPixelDatabases")
        return [file.name for file in response.entries]
