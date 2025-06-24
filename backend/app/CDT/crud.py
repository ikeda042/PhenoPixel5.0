import asyncio
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class DataResponse:
    filename: str
    mean_length: float
    nagg_rate: float


class CdtCrudBase:
    """CRUD utilities for CDT analysis."""

    ctrl_value: float | None = None

    @staticmethod
    def _parse_ctrl(text: str) -> float:
        ys_1 = [
            [float(x) for x in line.split(',') if x]
            for line in text.strip().splitlines()
            if line.strip()
        ]
        ys_norm: list[list[float]] = []
        for arr in ys_1:
            a = np.array(arr)
            if a.max() - a.min() != 0:
                a = (a - a.min()) / (a.max() - a.min())
            ys_norm.append(a.tolist())
        data = [float(np.sum(i) / len(i)) for i in ys_norm]
        data.sort(reverse=True)
        if not data:
            return 0.0
        index = int(len(data) * 0.95)
        return data[index]

    @staticmethod
    def _analyze_csv(text: str, ctrl: float) -> tuple[float, float]:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        cells: list[tuple[float, float]] = []
        for i in range(0, len(lines), 2):
            l = lines[i].split(',')[:-1]
            if len([j for j in l if j != ""]) != 35:
                continue
            length = round((float(l[-1]) - float(l[0])) * 0.065, 2)
            peak_points = np.array([float(x) for x in lines[i + 1].split(',') if x.strip()])
            if peak_points.max() - peak_points.min() != 0:
                peak_points = (peak_points - peak_points.min()) / (
                    peak_points.max() - peak_points.min()
                )
            peak_data = float(np.sum(peak_points) / len(peak_points))
            cells.append((length, peak_data))
        if not cells:
            return 0.0, 0.0
        c = sum(1 for _, p in cells if p < ctrl)
        nagg_rate = c / len(cells)
        mean_length = float(np.mean([l for l, _ in cells]))
        return mean_length, nagg_rate

    @classmethod
    async def set_ctrl(cls, file_content: bytes) -> None:
        cls.ctrl_value = await asyncio.to_thread(cls._parse_ctrl, file_content.decode())

    @classmethod
    async def analyze_files(cls, files: List[tuple[str, bytes]]) -> List[DataResponse]:
        if cls.ctrl_value is None:
            raise ValueError("Control data not set")
        results: List[DataResponse] = []
        for name, content in files:
            mean_length, nagg_rate = await asyncio.to_thread(
                cls._analyze_csv, content.decode(), cls.ctrl_value
            )
            results.append(DataResponse(filename=name, mean_length=mean_length, nagg_rate=nagg_rate))
        return results
