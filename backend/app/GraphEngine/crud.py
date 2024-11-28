import matplotlib.pyplot as plt
import numpy as np
import io
import asyncio
import time
import pandas as pd
import seaborn as sns
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
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

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import StreamingResponse

# app = FastAPI()

class MCPROperation:
    @classmethod
    async def combine_images(cls, image_files, per_row):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, cls._blocking_combine_images, image_files, per_row)

    @classmethod
    def _blocking_combine_images(cls, image_files, per_row):
        images = [cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR) for img in image_files]
        max_width = max(img.shape[1] for img in images)
        max_height = max(img.shape[0] for img in images)
        num_images = len(images)
        num_rows = num_images // per_row
        if num_images % per_row != 0:
            num_rows += 1

        final_image = np.zeros((num_rows * max_height, per_row * max_width, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            top = (i // per_row) * max_height
            left = (i % per_row) * max_width
            final_image[top:top+img.shape[0], left:left+img.shape[1]] = img

        _, buffer = cv2.imencode('.png', final_image)
        return BytesIO(buffer)

    @classmethod
    async def _draw_graph(cls, file_content: bytes, blank_index: str, timespan_sec: int = 180) -> None:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, cls._blocking_draw_graph, file_content, blank_index, timespan_sec)

    @classmethod
    def _blocking_draw_graph(cls, file_content: bytes, blank_index: str, timespan_sec: int = 180) -> list:
        df = pd.read_csv(BytesIO(file_content), encoding='unicode_escape')
        start_index = 0
        end_index = 0
        temp_detected = False
        for i in range(df.shape[0]-1):
            if "Temp" in str(df.iloc[i][0]):
                start_index = i
                temp_detected = True
            if type(df.iloc[i+1][0]) == float and type(df.iloc[i][0]) == str and temp_detected:
                end_index = i+1
                break
        char_i = f"{df.iloc[start_index+1][0][0]}"
        data = {}
        for i in range(start_index+1, end_index):
            data[df.iloc[i][0][0]] = []
        for i in range(start_index+1, end_index):
            data[df.iloc[i][0][0]].append({df.iloc[i][0]: [float(i) for i in df.iloc[i][1:-1]]})

        x = [i * timespan_sec / 3600 for i in range(len(df.iloc[start_index-2]) - 2)]

        blank_index = blank_index.replace(" ", "")
        if blank_index not in [key[1:] for sublist in [[list(i.keys())[0] for i in data[j]] for j in list(data.keys())] for key in sublist]:
            raise ValueError(f"Key {blank_index} not found in data")

        image_buffers = []
        for graph in range(len(list(data.keys()))):
            time.sleep(1)
            fig = plt.figure(figsize=[5, 5])
            sns.set()
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.major.width'] = 1.0
            plt.rcParams['ytick.major.width'] = 1.0
            plt.rcParams['font.size'] = 9
            plt.rcParams['axes.linewidth'] = 1.0
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            plt.xlabel("time(h)")
            plt.ylabel("OD600(-)")
            for i in data[list(data.keys())[graph]]:
                key = [j for j in i.keys()][0]
                y = i[key]
                plt.scatter(x, y, label=f"{key}", s=6)
            plt.legend(title="Series")
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=500)
            buf.seek(0)
            image_buffers.append(buf)
            plt.close()
        return image_buffers
