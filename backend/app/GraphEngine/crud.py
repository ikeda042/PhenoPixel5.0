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
    def process_heatmap_abs(data):
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
        plt.savefig(buf, format="png", dpi=500)
        buf.seek(0)
        plt.clf()

        return buf

    def process_heatmap_rel(data):
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
        plt.savefig(buf, format="png", dpi=500)
        buf.seek(0)
        plt.clf()

        return buf


class AsyncChores:
    async def process_heatmap_abs(self, data):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, SyncChores.process_heatmap_abs, data)

    async def process_heatmap_rel(self, data):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, SyncChores.process_heatmap_rel, data)


class GraphEngineCrudBase:
    async def process_heatmap_abs(data):
        return await AsyncChores().process_heatmap_abs(data)

    async def process_heatmap_rel(data):
        return await AsyncChores().process_heatmap_rel(data)
