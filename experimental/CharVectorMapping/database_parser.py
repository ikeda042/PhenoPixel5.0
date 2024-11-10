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
    mask = np.zeros_like(img_fluo)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img_fluo, mask)
    masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    # ユニークな輝度値を取得してソート
    unique_vals = np.unique(masked_gray[masked_gray > 0])
    
    # 最小から2番目の輝度値を取得
    if len(unique_vals) > 1:
        second_min_val = unique_vals[1]  # 2番目に小さい輝度値
    else:
        second_min_val = unique_vals[0]  # 画素が1種類しかない場合
    
    min_val, max_val, _, _ = cv2.minMaxLoc(masked_gray)
    
    # 輝度の正規化処理
    if max_val > min_val:
        normalized = np.clip(masked_gray, second_min_val, max_val)  # 範囲を制限
        normalized = cv2.normalize(
            normalized, None, alpha=1, beta=255, norm_type=cv2.NORM_MINMAX
        )
    else:
        normalized = masked_gray
    
    return img_fluo, normalized



def database_parser(dbname: str):
    dbpath = f"sqlite:///experimental/CharVectorMapping/{dbname}"
    engine = create_engine(dbpath)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == 1).all()

    for cell in cells_with_label_1:
        img_fluo, masked = parse_image(cell)
        cv2.imwrite(
            f"experimental/CharVectorMapping/images/fluo/{dbname}-{cell.cell_id}.png", img_fluo
        )
        cv2.imwrite(
            f"experimental/CharVectorMapping/images/fluo_masked/{dbname}-{cell.cell_id}.png",
            masked,
        )

    session.close()

if __name__ == "__main__":
    database_parser("sk326Gen30min.db")
    database_parser("sk326Gen60min.db")
    database_parser("sk326Gen90min.db")
    database_parser("sk326Gen120min.db")
