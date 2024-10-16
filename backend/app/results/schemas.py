from pydantic import BaseModel


class FileName(BaseModel):
    name: str
