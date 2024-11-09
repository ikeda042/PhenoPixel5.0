import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, FLOAT, BLOB

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
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)


def parse_image(cell: Cell) -> tuple:
    img_fluo = cv2.imdecode(np.frombuffer(cell.img_fluo1, np.uint8), cv2.IMREAD_COLOR)
    contour = pickle.loads(cell.contour)
    mask = np.zeros_like(img_ph)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img_ph, mask)
    masked[mask > 0] = 255
    return img_fluo, masked


dbpath = "sqlite:///experimental/CharVectorMapping/sk326Gen120min.db"
engine = create_engine(dbpath)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == 1).all()

for cell in cells_with_label_1:
    img_ph, masked = parse_image(cell)
    cv2.imwrite(f"images/fluo/{cell.cell_id}.png", img_ph)
    cv2.imwrite(f"images/fluo_masked/{cell.cell_id}.png", masked)
