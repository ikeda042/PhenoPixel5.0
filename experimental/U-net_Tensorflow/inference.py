import cv2
import numpy as np
import tensorflow as tf
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, FLOAT, BLOB
import cv2
import numpy as np
import pickle

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


dbpath = "sqlite:///experimental/U-net_Tensorflow/test_contour_label_data.db"
engine = create_engine(dbpath)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def parse_image(cell: Cell) -> tuple:
    img_ph = cv2.imdecode(np.frombuffer(cell.img_ph, np.uint8), cv2.IMREAD_COLOR)
    contour = pickle.loads(cell.contour)
    mask = np.zeros_like(img_ph)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img_ph, mask)
    masked[mask > 0] = 255
    return img_ph, masked


cells_with_label_1 = session.query(Cell).filter(Cell.manual_label == 1).all()


# モデルを保存したファイルから読み込む
model = tf.keras.models.load_model("experimental/U-net_Tensorflow/model.h5")


# 画像のリサイズ用関数（推論前にリサイズする）
def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)


# 画像のリサイズを元に戻すための関数（推論後にリサイズする）
def resize_to_original_size(img, original_size):
    return cv2.resize(img, original_size)


# 新しい画像に対して推論し、元のサイズに戻す
def predict_contour(model, img_ph):
    original_size = img_ph.shape[:2]  # 元の画像サイズを取得
    img_resized = resize_image(img_ph)  # 256x256にリサイズ
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # モデルで予測
    prediction = model.predict(img_resized)

    # 二値化して255スケールに戻す
    prediction = (prediction > 0.5).astype(np.uint8) * 255

    # 予測結果を元のサイズにリサイズ
    prediction_resized = cv2.resize(prediction[0], (original_size[1], original_size[0]))
    return prediction_resized


# 推論
for cell in cells_with_label_1:
    img_ph = cv2.imdecode(np.frombuffer(cell.img_ph, np.uint8), cv2.IMREAD_COLOR)
    prediction = predict_contour(model, img_ph)
    cv2.imwrite(
        f"experimental/U-net_Tensorflow/images/predicted/{cell.cell_id}.png", prediction
    )
