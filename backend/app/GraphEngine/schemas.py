from dataclasses import dataclass


@dataclass
class HeatMapVector:
    index: int
    u1: list[float]
    G: list[float]
    length: float

    def __repr__(self) -> str:
        return f"u1: {self.u1}\nG: {self.G}"

    def __gt__(self, other: "HeatMapVector") -> bool:
        return self.length > other.length
