import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras import regularizers

# 画像パスの設定
image_contorols = "experimental/CharVectorMapping/images/dataset/ctrls"
image_positives = "experimental/CharVectorMapping/images/dataset/positives"

image_ctrls_paths = [
    os.path.join(image_contorols, file) for file in os.listdir(image_contorols) if file.endswith(".png")
]
image_positives_paths = [
    os.path.join(image_positives, file) for file in os.listdir(image_positives) if file.endswith(".png")
]

# 画像の前処理関数
def load_and_preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  # 正規化
    return image

# コントロール群とポジティブ群の画像を読み込み
X_train = np.array([load_and_preprocess_image(path) for path in image_ctrls_paths])
X_test = np.array([load_and_preprocess_image(path) for path in image_positives_paths])

# 形状を(サンプル数, 高さ, 幅, チャネル)に変更
X_train = X_train.reshape((X_train.shape[0], 64, 64, 1))
X_test = X_test.reshape((X_test.shape[0], 64, 64, 1))

# オートエンコーダーの構築
input_img = Input(shape=(64, 64, 1))
x = Flatten()(input_img)
x = Dense(256, activation="relu", activity_regularizer=regularizers.l1(10e-5))(x)
encoded = Dense(64, activation="relu")(x)
x = Dense(256, activation="relu")(encoded)
x = Dense(64 * 64, activation="sigmoid")(x)
decoded = Reshape((64, 64, 1))(x)

autoencoder = Model(inputs=input_img, outputs=decoded)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# モデルの学習
autoencoder.fit(X_train, X_train, epochs=50, batch_size=8, shuffle=True, validation_split=0.2, verbose=1)

# テストデータをオートエンコーダーに通す
reconstructed = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructed, 2), axis=(1, 2, 3))

# MSEが大きい順に並べた画像ファイル名のリストを取得
mse_with_filenames = sorted(zip(mse, image_positives_paths), key=lambda x: x[0], reverse=True)
mse_filenames_sorted = [os.path.basename(filename).replace(".png", "") for _, filename in mse_with_filenames]

# MSEの大きい順のファイル名を表示
print("MSEが大きい順の画像ファイル名:")
print(mse_filenames_sorted)

# 画像をMSE順に並べたリスト
loaded_images = [cv2.imread(path) for _, path in mse_with_filenames]

# 各画像が同じサイズであることを前提に、画像サイズを取得
img_height, img_width, _ = loaded_images[0].shape

# 結合後の正方形に近い画像サイズを決定
num_images = len(loaded_images)
side_length = int(np.ceil(np.sqrt(num_images)))  # 正方形にするための行・列の数

# 正方形のキャンバスを作成
canvas_height = side_length * img_height
canvas_width = side_length * img_width
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# 画像をキャンバスに配置
for idx, image in enumerate(loaded_images):
    row = idx // side_length
    col = idx % side_length
    y_start = row * img_height
    x_start = col * img_width
    canvas[y_start:y_start + img_height, x_start:x_start + img_width] = image

# 結合した画像を保存
output_path = "experimental/CharVectorMapping/images/combined_image.png"
cv2.imwrite(output_path, canvas)

print(f"結合した画像が {output_path} に保存されました。")
