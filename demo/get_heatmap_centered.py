import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

max_y = 8


@dataclass
class HeatMapVector:
    index: int
    u1: list[float]
    G: list[float]
    length: float

    def __repr__(self) -> str:
        return f"u1: {self.u1}\nG: {self.G}"

    def __gt__(self, other):
        return self.length > other.length


def generate_heatmap_abs(file_name: str):
    with open(file_name) as f:
        data = [
            [
                float(j.replace("\n", "").replace(" ", ""))
                for j in i.split(",")
                if j.strip()
            ]
            for i in f.readlines()
        ]
    heatmap_vectors = sorted(
        [
            HeatMapVector(
                index=i,
                length=max(data[2 * i]) - min(data[2 * i]),
                u1=[d - min(data[2 * i]) for d in data[2 * i]],
                G=data[2 * i + 1],
            )
            for i in range(len(data) // 2)
        ]
    )

    max_length = max(heatmap_vectors).length
    heatmap_vectors = [
        HeatMapVector(
            index=vec.index,
            u1=[
                (d + (max_length - vec.length) / 2 + (max_y / 0.065 - max_length) / 2)
                * 0.065
                - max_y / 2
                for d in vec.u1
            ],
            G=vec.G,
            length=vec.length,
        )
        for vec in heatmap_vectors
    ]

    fig, ax = plt.subplots(figsize=(14, 9))

    u1_min = min(map(min, [vec.u1 for vec in heatmap_vectors]))
    u1_max = max(map(max, [vec.u1 for vec in heatmap_vectors]))

    cmap = plt.cm.jet
    for idx, vec in enumerate(heatmap_vectors):
        u1 = vec.u1
        G_normalized = (np.array(vec.G) - min(vec.G)) / (max(vec.G) - min(vec.G))
        colors = cmap(G_normalized)

        offset = len(heatmap_vectors) - idx - 1
        for i in range(len(u1) - 1):
            ax.plot([offset, offset], u1[i : i + 2], color=colors[i], lw=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Normalized G Value")

    ax.set_ylim([-max_y / 2, max_y / 2])
    ax.set_xlim([-0.5, len(heatmap_vectors) - 0.5])
    ax.set_ylabel("Distance from center (µm)", fontdict={"fontsize": 20})
    ax.set_xlabel("cell number", fontdict={"fontsize": 20})

    # Set y-axis ticks at intervals of 1
    ax.set_yticks(np.arange(-max_y / 2, max_y / 2 + 1, 1))

    plt.savefig("docs_images/stacked_heatmap_abs_centered.png", dpi=800)


generate_heatmap_abs("demo/heatmapdata-bulk.csv")
