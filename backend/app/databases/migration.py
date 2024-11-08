from sqlalchemy import Integer, String, BLOB, FLOAT, create_engine, Column
from sqlalchemy.orm import declarative_base
import cv2
import numpy as np
import pickle
from numpy.linalg import inv
from schemas import BasicCellInfo, CellStats, CellDBAll
from database import get_cell_all
from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.integrate import quad
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from scipy.optimize import minimize
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw, ImageFont
from scipy.linalg import eig
from pydantic import BaseModel
import sqlite3
import os


def basis_conversion(
    contour: list[list[int]],
    X: np.ndarray,
    center_x: float,
    center_y: float,
    coordinates_incide_cell: list[list[int]],
) -> list[list[float]]:
    Sigma = np.cov(X)
    eigenvalues, eigenvectors = eig(Sigma)
    if eigenvalues[1] < eigenvalues[0]:
        m = eigenvectors[1][1] / eigenvectors[1][0]
        Q = np.array([eigenvectors[1], eigenvectors[0]])
        U = [Q.transpose() @ np.array([i, j]) for i, j in coordinates_incide_cell]
        U = [[j, i] for i, j in U]
        contour_U = [Q.transpose() @ np.array([j, i]) for i, j in contour]
        contour_U = [[j, i] for i, j in contour_U]
        color = "red"
        center = [center_x, center_y]
        u1_c, u2_c = center @ Q
    else:
        m = eigenvectors[0][1] / eigenvectors[0][0]
        Q = np.array([eigenvectors[0], eigenvectors[1]])
        U = [
            Q.transpose() @ np.array([j, i]).transpose()
            for i, j in coordinates_incide_cell
        ]
        contour_U = [Q.transpose() @ np.array([i, j]) for i, j in contour]
        color = "blue"
        center = [center_x, center_y]
        u2_c, u1_c = center @ Q

    u1 = [i[1] for i in U]
    u2 = [i[0] for i in U]
    u1_contour = [i[1] for i in contour_U]
    u2_contour = [i[0] for i in contour_U]
    min_u1, max_u1 = min(u1), max(u1)
    return u1, u2, u1_contour, u2_contour, min_u1, max_u1, u1_c, u2_c, U, contour_U


def get_cell_stats(dbname: str, include_ph: bool = True) -> list[CellStats]:
    Base = declarative_base()

    class Cell(Base):
        __tablename__ = "cells"
        id = Column(Integer, primary_key=True)
        cell_id = Column(String)
        label_experiment = Column(String)
        manual_label = Column(Integer)
        perimeter = Column(FLOAT)
        area = Column(FLOAT)
        img_ph = Column(BLOB)
        img_fluo1 = Column(BLOB)
        img_fluo2 = Column(BLOB, nullable=True) | None
        contour = Column(BLOB)
        center_x = Column(FLOAT)
        center_y = Column(FLOAT)

    class CellStats(BaseModel):
        cell_id: str
        image_ph: bytes
        image_fluo1: bytes
        image_fluo2: bytes | None
        contour: bytes
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

    engine = create_engine(f"sqlite:///{dbname}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    returnval = []
    with Session() as session:
        cells = session.query(Cell).all()
        for cell in tqdm(cells):
            if cell.manual_label != "N/A" and cell.manual_label != None:
                image_ph = cv2.imdecode(
                    np.frombuffer(cell.img_ph, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                image_fluo = cv2.imdecode(
                    np.frombuffer(cell.img_fluo1, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                contour_raw = cell.contour
                gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

                contour = [[j, i] for i, j in [i[0] for i in pickle.loads(contour_raw)]]
                coords_inside_cell_1, points_inside_cell_1, projected_points = (
                    [],
                    [],
                    [],
                )
                for i in range(image_fluo.shape[1]):
                    for j in range(image_fluo.shape[0]):
                        if (
                            cv2.pointPolygonTest(
                                pickle.loads(contour_raw), (i, j), False
                            )
                            >= 0
                        ):
                            coords_inside_cell_1.append([i, j])
                            points_inside_cell_1.append(gray[j][i])
                X = np.array(
                    [
                        [i[1] for i in coords_inside_cell_1],
                        [i[0] for i in coords_inside_cell_1],
                    ]
                )
                (
                    u1,
                    u2,
                    u1_contour,
                    u2_contour,
                    min_u1,
                    max_u1,
                    u1_c,
                    u2_c,
                    U,
                    contour_U,
                ) = basis_conversion(
                    contour,
                    X,
                    image_fluo.shape[0] // 2,
                    image_fluo.shape[1] // 2,
                    coords_inside_cell_1,
                )
                min_u1, max_u1 = min(u1), max(u1)
                max_brightness = max(points_inside_cell_1)
                min_brightness = min(points_inside_cell_1)
                W = np.array([[i**4, i**3, i**2, i, 1] for i in [i[1] for i in U]])
                f = np.array([i[0] for i in U])
                theta = inv(W.transpose() @ W) @ W.transpose() @ f
                x = np.linspace(min_u1, max_u1, 1000)
                y = [
                    theta[0] * i**4
                    + theta[1] * i**3
                    + theta[2] * i**2
                    + theta[3] * i
                    + theta[4]
                    for i in x
                ]
                mean_brightness_raw = round(np.mean(points_inside_cell_1), 2)
                mean_brightness_normalized = round(
                    np.mean([i / max_brightness for i in points_inside_cell_1]), 2
                )
                median_brightness_raw = round(np.median(points_inside_cell_1), 2)
                median_brightness_normalized = round(
                    np.median([i / max_brightness for i in points_inside_cell_1]), 2
                )
                if include_ph:
                    image_ph = cv2.cvtColor(image_ph, cv2.COLOR_BGR2GRAY)
                    coords_inside_cell_1_ph, points_inside_cell_1_ph = [], []
                    for i in range(image_ph.shape[1]):
                        for j in range(image_ph.shape[0]):
                            if (
                                cv2.pointPolygonTest(
                                    pickle.loads(contour_raw), (i, j), False
                                )
                                >= 0
                            ):
                                coords_inside_cell_1_ph.append([i, j])
                                points_inside_cell_1_ph.append(image_ph[j][i])
                    ph_max_brightness = max(points_inside_cell_1_ph)
                    ph_min_brightness = min(points_inside_cell_1_ph)
                    ph_mean_brightness_raw = round(np.mean(points_inside_cell_1_ph), 4)
                    ph_mean_brightness_normalized = round(
                        np.mean(
                            [i / ph_max_brightness for i in points_inside_cell_1_ph]
                        ),
                        4,
                    )
                    ph_median_brightness_raw = round(
                        np.median(points_inside_cell_1_ph), 4
                    )
                    ph_median_brightness_normalized = round(
                        np.median(
                            [i / ph_max_brightness for i in points_inside_cell_1_ph]
                        ),
                        4,
                    )
                returnval.append(
                    CellStats(
                        basic_cell_info=BasicCellInfo(
                            cell_id=cell.cell_id,
                            label_experiment=cell.label_experiment,
                            manual_label=int(cell.manual_label),
                            perimeter=round(cell.perimeter, 2),
                            area=cell.area,
                        ),
                        ph_max_brightness=ph_max_brightness,
                        ph_min_brightness=ph_min_brightness,
                        ph_mean_brightness_raw=ph_mean_brightness_raw,
                        ph_mean_brightness_normalized=ph_mean_brightness_normalized,
                        ph_median_brightness_raw=ph_median_brightness_raw,
                        ph_median_brightness_normalized=ph_median_brightness_normalized,
                        max_brightness=max_brightness,
                        min_brightness=min_brightness,
                        mean_brightness_raw=mean_brightness_raw,
                        mean_brightness_normalized=mean_brightness_normalized,
                        median_brightness_raw=median_brightness_raw,
                        median_brightness_normalized=median_brightness_normalized,
                        image_ph=cell.img_ph,
                        image_fluo1=cell.img_fluo1,
                        image_fluo2=None,
                        contour=cell.contour,
                        center_x=cell.center_x,
                        center_y=cell.center_y,
                        cell_id=cell.cell_id,
                    )
                )
        return returnval


Base2 = declarative_base()


class Cell2(Base2):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)
    max_brightness = Column(FLOAT)
    min_brightness = Column(FLOAT)
    mean_brightness_raw = Column(FLOAT)
    mean_brightness_normalized = Column(FLOAT)
    median_brightness_raw = Column(FLOAT)
    median_brightness_normalized = Column(FLOAT)
    ph_max_brightness = Column(FLOAT)
    ph_min_brightness = Column(FLOAT)
    ph_mean_brightness_raw = Column(FLOAT)
    ph_mean_brightness_normalized = Column(FLOAT)
    ph_median_brightness_raw = Column(FLOAT)
    ph_median_brightness_normalized = Column(FLOAT)


def migrate(dbname: str) -> None:
    conn = sqlite3.connect(dbname)
    engine = create_engine(f"sqlite:///{dbname}")
    Base2.metadata.create_all(engine)
    c = conn.cursor()

    i = 0
    for cell in get_cell_stats(f"backend/databases/{dbname}"):
        cell_id = cell.basic_cell_info.cell_id
        label_experiment = cell.basic_cell_info.label_experiment
        manual_label = cell.basic_cell_info.manual_label
        perimeter = cell.basic_cell_info.perimeter
        area = cell.basic_cell_info.area
        img_ph = cell.image_ph
        img_fluo1 = cell.image_fluo1
        img_fluo2 = None
        contour = cell.contour
        center_x = cell.center_x
        center_y = cell.center_y
        max_brightness = cell.max_brightness
        min_brightness = cell.min_brightness
        mean_brightness_raw = cell.mean_brightness_raw
        mean_brightness_normalized = cell.mean_brightness_normalized
        median_brightness_raw = cell.median_brightness_raw
        median_brightness_normalized = cell.median_brightness_normalized
        ph_max_brightness = cell.ph_max_brightness
        ph_min_brightness = cell.ph_min_brightness
        ph_mean_brightness_raw = cell.ph_mean_brightness_raw
        ph_mean_brightness_normalized = cell.ph_mean_brightness_normalized
        ph_median_brightness_raw = cell.ph_median_brightness_raw
        ph_median_brightness_normalized = cell.ph_median_brightness_normalized
        id = i
        i += 1
        c.execute(
            """
        INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
            (
                id,
                cell_id,
                label_experiment,
                manual_label,
                perimeter,
                area,
                img_ph,
                img_fluo1,
                img_fluo2,
                contour,
                center_x,
                center_y,
                max_brightness,
                min_brightness,
                mean_brightness_raw,
                mean_brightness_normalized,
                median_brightness_raw,
                median_brightness_normalized,
                ph_max_brightness,
                ph_min_brightness,
                ph_mean_brightness_raw,
                ph_mean_brightness_normalized,
                ph_median_brightness_raw,
                ph_median_brightness_normalized,
            ),
        )
    conn.commit()
    conn.close()


# for i in os.listdir("backend/databases"):
#     if i.endswith(".db"):
#         migrate(f"{i}")


for i in os.listdir("backend/dbold"):
    if i.endswith(".db"):
        migrate(f"{i}")
        print(f"migrated {i}")
