from sqlalchemy import Integer, String, BLOB, FLOAT, create_engine, Column
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT
from sqlalchemy.ext.declarative import declarative_base
import sqlite3
import os


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


for i in os.listdir("backend/app/databases"):
    if i.endswith(".db"):
        migrate(f"{i}")
        print(f"migrated {i}")
