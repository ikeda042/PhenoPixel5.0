from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, FLOAT, BLOB
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


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

for cell in cells_with_label_1:
    img_ph, masked = parse_image(cell)
    cv2.imwrite(f"experimental/U-net_Tensorflow/images/ph/{cell.cell_id}.png", img_ph)
    cv2.imwrite(
        f"experimental/U-net_Tensorflow/images/masked/{cell.cell_id}.png", masked
    )


# U-Net
def unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)
    # エンコーダー
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # ボトム層
    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c4)

    # デコーダー
    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u5)
    c5 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

    # 出力層
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# モデルを生成
model = unet_model()


# 画像のリサイズ用関数
def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)


# 画像データとラベルデータのリストを作成
input_images = []
mask_images = []

for cell in cells_with_label_1:
    img_ph, masked = parse_image(cell)
    input_images.append(resize_image(img_ph))
    mask_images.append(resize_image(masked))

# リストをnumpy配列に変換
input_images = np.array(input_images)
mask_images = np.array(mask_images)

# データを0-1に正規化
input_images = input_images / 255.0
mask_images = mask_images / 255.0

# ターゲットの形状を (None, 256, 256, 1) に変更
mask_images = np.expand_dims(mask_images[:, :, :, 0], axis=-1)

# データを学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(
    input_images, mask_images, test_size=0.1
)

# 学習
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.1)

model.save("model.h5")


# 新しい画像に対して推論
def predict_contour(model, img_ph):
    img_resized = resize_image(img_ph)
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_resized)
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction[0]


for cell in cells_with_label_1:
    img_ph = cv2.imdecode(np.frombuffer(cell.img_ph, np.uint8), cv2.IMREAD_COLOR)
    prediction = predict_contour(model, img_ph)
    cv2.imwrite(
        f"experimental/U-net_Tensorflow/images/predicted/{cell.cell_id}.png",
        prediction,
    )
