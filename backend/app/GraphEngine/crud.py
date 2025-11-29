import matplotlib.pyplot as plt
import numpy as np
import io
import asyncio
import time
import pandas as pd
import seaborn as sns
import concurrent.futures
import matplotlib

matplotlib.use("Agg")
import cv2
from io import BytesIO

from GraphEngine.schemas import HeatMapVectorAbs, HeatMapVectorRel


class SyncChores:
    def process_heatmap_abs(data, dpi: int = 500):
        heatmap_vectors = sorted(
            [
                HeatMapVectorAbs(
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
            HeatMapVectorAbs(
                index=vec.index,
                u1=[d + (max_length - vec.length) / 2 - max_length / 2 for d in vec.u1],
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

        ax.set_ylim([u1_min, u1_max])
        ax.set_xlim([-0.5, len(heatmap_vectors) - 0.5])
        ax.set_ylabel("Cell length (px)")
        ax.set_xlabel("Cell number")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        plt.close(fig)

        return buf

    def process_distribution(data, dpi: int = 500):
        """Create histogram of total intensities for each cell."""
        totals = [sum(data[2 * i + 1]) for i in range(len(data) // 2)]

        fig = plt.figure(figsize=(6, 4))
        plt.hist(totals, bins=20, edgecolor="black", color="skyblue")
        plt.xlabel("Total intensity")
        plt.ylabel("Count")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        plt.close(fig)
        return buf

    def process_distribution_box(data, dpi: int = 500):
        """Create box plot of total intensities for each cell."""
        totals = [sum(data[2 * i + 1]) for i in range(len(data) // 2)]

        fig = plt.figure(figsize=(8, 6))
        x = np.random.normal(1, 0.04, size=len(totals))
        plt.plot(x, totals, "o", alpha=0.5)
        plt.boxplot(totals, flierprops=dict(marker=""))
        plt.ylabel("Total intensity")
        max_total = max(totals) if totals else 0
        plt.ylim(0, max_total * 1.05)
        plt.gca().axes.get_xaxis().set_visible(False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        plt.close(fig)
        return buf

    def process_heatmap_rel(data, dpi: int = 500):
        heatmap_vectors = sorted(
            [
                HeatMapVectorRel(
                    index=i,
                    length=len(data[2 * i]),
                    u1=[i for i in range(len(data[2 * i]))],
                    G=data[2 * i + 1],
                )
                for i in range(len(data) // 2)
            ]
        )

        max_length = max(heatmap_vectors).length
        heatmap_vectors = [
            HeatMapVectorRel(
                index=vec.index,
                u1=[d + (max_length - vec.length) / 2 - max_length / 2 for d in vec.u1],
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

        ax.set_ylim([u1_min, u1_max])
        ax.set_xlim([-0.5, len(heatmap_vectors) - 0.5])
        ax.set_ylabel("Relative position(-)")
        ax.set_xlabel("Cell number")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        plt.close(fig)

        return buf

    def process_boxplot_values(
        values: list[float], title: str, xlabel: str, dpi: int = 200
    ):
        """Create a simple horizontal boxplot with jittered points."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=values, ax=ax, color="#8fb1ff", width=0.25)
        sns.stripplot(
            x=values, ax=ax, color="#1a1a1a", size=3, alpha=0.6, jitter=0.15
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf


class AsyncChores:
    async def process_heatmap_abs(self, data, dpi: int = 500):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, SyncChores.process_heatmap_abs, data, dpi)

    async def process_heatmap_rel(self, data, dpi: int = 500):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, SyncChores.process_heatmap_rel, data, dpi)

    async def process_distribution(self, data, dpi: int = 500):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, SyncChores.process_distribution, data, dpi)

    async def process_distribution_box(self, data, dpi: int = 500):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, SyncChores.process_distribution_box, data, dpi)

    async def process_boxplot_values(
        self, values: list[float], title: str, xlabel: str, dpi: int = 200
    ):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, SyncChores.process_boxplot_values, values, title, xlabel, dpi
        )


class GraphEngineCrudBase:
    async def process_heatmap_abs(data, dpi: int = 500):
        return await AsyncChores().process_heatmap_abs(data, dpi)

    async def process_heatmap_rel(data, dpi: int = 500):
        return await AsyncChores().process_heatmap_rel(data, dpi)

    async def process_distribution(data, dpi: int = 500):
        return await AsyncChores().process_distribution(data, dpi)

    async def process_distribution_box(data, dpi: int = 500):
        return await AsyncChores().process_distribution_box(data, dpi)

    async def boxplot_from_values(
        values: list[float], title: str, xlabel: str, dpi: int = 200
    ):
        return await AsyncChores().process_boxplot_values(values, title, xlabel, dpi)
