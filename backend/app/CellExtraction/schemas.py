from pydantic import BaseModel, Field, ValidationInfo, field_validator


class FrameSplitConfig(BaseModel):
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    db_name: str

    @field_validator("db_name")
    @classmethod
    def validate_db_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("db_name cannot be empty")
        return value

    @field_validator("frame_end")
    @classmethod
    def validate_range(cls, value: int, info: ValidationInfo):
        frame_start = info.data.get("frame_start")
        if frame_start is not None and value < frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start")
        return value


class CreatedDatabase(BaseModel):
    frame_start: int
    frame_end: int
    db_name: str


class CellExtractionResponse(BaseModel):
    num_tiff: int
    ulid: str
    db_name: str
    created_databases: list[CreatedDatabase]
