from pydantic import BaseModel


class CellExtractionResponse(BaseModel):
    num_tiff: int
    ulid: str
