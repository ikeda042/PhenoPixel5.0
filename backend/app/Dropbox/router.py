from fastapi import APIRouter
from Dropbox.crud import DropboxCrud
from typing import Literal


router_dropbox = APIRouter(prefix="/dropbox", tags=["dropbox"])
