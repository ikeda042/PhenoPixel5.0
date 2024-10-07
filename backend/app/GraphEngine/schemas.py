from __future__ import annotations
from dataclasses import dataclass


@dataclass
class HeatMapVectorAbs:
    index: int
    u1: list[float]
    G: list[float]
    length: float

    def __repr__(self) -> str:
        return f"u1: {self.u1}\nG: {self.G}"

    def __gt__(self, other: HeatMapVectorAbs) -> bool:
        return self.length > other.length


@dataclass
class HeatMapVectorRel:
    index: int
    u1: list[float]
    G: list[float]
    length: float

    def __repr__(self) -> str:
        return f"u1: {self.u1}\nG: {self.G}"

    def __gt__(self, other: HeatMapVectorAbs) -> bool:
        return sum(self.G) > sum(other.G)
