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
    user_id = Column(String, nullable=True)


def parse_image(cell: Cell) -> tuple:
    # BGR画像を読み込み
    img_fluo = cv2.imdecode(np.frombuffer(cell.img_fluo1, np.uint8), cv2.IMREAD_COLOR)

    # 緑チャネルのみを抽出
    green_channel = img_fluo[:, :, 1]  # 1は緑チャネルを示す

    # 輪郭情報をデコード
    contour = pickle.loads(cell.contour)

    # マスク作成
    mask = np.zeros_like(green_channel)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # マスクを適用して緑チャネルの画像を作成
    masked_green = cv2.bitwise_and(green_channel, mask)

    # 輝度値の抽出とソート
    unique_vals = np.unique(masked_green[masked_green > 0])

    # 最小から2番目の輝度値を取得
    if len(unique_vals) > 1:
        second_min_val = unique_vals[1]  # 2番目に小さい輝度値
    else:
        second_min_val = unique_vals[0]  # 画素が1種類しかない場合

    min_val, max_val, _, _ = cv2.minMaxLoc(masked_green)

    # 輝度の正規化処理
    if max_val > min_val:
        normalized = np.clip(masked_green, second_min_val, max_val)  # 範囲を制限
        normalized = cv2.normalize(
            normalized, None, alpha=1, beta=255, norm_type=cv2.NORM_MINMAX
        )
    else:
        normalized = masked_green

    # 緑チャネルをカラー画像として出力するためにBGRに変換
    green_bgr = cv2.merge(
        [np.zeros_like(normalized), normalized, np.zeros_like(normalized)]
    )

    return img_fluo, green_bgr


def database_parser(dbname: str) -> list[Cell]:
    dbpath = f"sqlite:///experimental/SK328/{dbname}"
    print(dbpath)
    print("______________________________")
    print("______________________________")
    print("______________________________")
    print("______________________________")
    print("______________________________")
    print("______________________________")
    print("______________________________")     
    engine = create_engine(dbpath)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session.query(Cell).filter(Cell.manual_label == 1).all()
