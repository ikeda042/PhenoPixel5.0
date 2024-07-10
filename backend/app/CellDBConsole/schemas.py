from pydantic import BaseModel


class CellDB(BaseModel):
    cell_id: str
    label_experiment: str
    manual_label: int
    perimeter: float
    area: float


class DBInfo(BaseModel):
    file_name: str
    cell_count: int


class CellId(BaseModel):
    cell_id: str


class CellMorhology(BaseModel):
    area: float
    volume: float
    width: float
    length: float
    mean_fluo_intensity: float
    mean_ph_intensity: float
    mean_fluo_intensity_normalized: float
    mean_ph_intensity_normalized: float
    median_fluo_intensity: float
    median_ph_intensity: float
    median_fluo_intensity_normalized: float
    median_ph_intensity_normalized: float


class CellDBAll(BaseModel):
    cell_id: str
    label_experiment: str
    manual_label: str
    perimeter: float
    area: float
    img_ph: bytes
    img_fluo1: bytes | None
    img_fluo2: bytes | None
    contour: bytes
    center_x: float
    center_y: float


class BasicCellInfo(BaseModel):
    cell_id: str
    label_experiment: str
    manual_label: int
    perimeter: float
    area: float


class CellStats(BaseModel):
    basic_cell_info: BasicCellInfo
    ph_max_brightness: float | None = None
    ph_min_brightness: float | None = None
    ph_mean_brightness_raw: float | None = None
    ph_mean_brightness_normalized: float | None = None
    ph_median_brightness_raw: float | None = None
    ph_median_brightness_normalized: float | None = None
    max_brightness: float
    min_brightness: float
    mean_brightness_raw: float
    mean_brightness_normalized: float
    median_brightness_raw: float
    median_brightness_normalized: float


class CellStatsv2(BaseModel):
    cell_id: str
    center_x: float
    center_y: float
    basic_cell_info: BasicCellInfo
    ph_max_brightness: float | None = None
    ph_min_brightness: float | None = None
    ph_mean_brightness_raw: float | None = None
    ph_mean_brightness_normalized: float | None = None
    ph_median_brightness_raw: float | None = None
    ph_median_brightness_normalized: float | None = None
    max_brightness: float
    min_brightness: float
    mean_brightness_raw: float
    mean_brightness_normalized: float
    median_brightness_raw: float
    median_brightness_normalized: float


class ListDBresponse(BaseModel):
    databases: list[str]
