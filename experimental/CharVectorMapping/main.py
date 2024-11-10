import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from skimage import io, color, img_as_ubyte
from mahotas.features import haralick

# 画像パスの設定
image_contorols = "experimental/CharVectorMapping/images/dataset/ctrls"
image_positives = "experimental/CharVectorMapping/images/dataset/positives"

image_ctrls_paths = [
    os.path.join(image_contorols, file) for file in os.listdir(image_contorols) if file.endswith(".png")
]
image_positives_paths = [
    os.path.join(image_positives, file) for file in os.listdir(image_positives) if file.endswith(".png")
]

# Haralick特徴量抽出関数
def extract_haralick_features(image_path):
    image = io.imread(image_path)
    if image.ndim == 3:  # RGBの場合はグレースケールに変換
        image = color.rgb2gray(image)
    image = img_as_ubyte(image)
    haralick_features = haralick(image).mean(axis=0)  # 平均をとって特徴量を取得
    return haralick_features

# 特徴量の抽出（Haralick）
features_ctrls = [extract_haralick_features(path) for path in image_ctrls_paths]
features_positives = [extract_haralick_features(path) for path in image_positives_paths]

# コントロール群をトレーニングデータとして使用
X_train = np.array(features_ctrls)
X_test = np.array(features_positives)

# データの正規化
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_train.min()) / (X_train.max() - X_train.min())

# オートエンコーダーの構築
input_dim = X_train.shape[1]
encoding_dim = 10  # 圧縮する次元数

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(10e-5))(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# モデルの学習
autoencoder.fit(X_train, X_train, epochs=50, batch_size=8, shuffle=True, validation_split=0.2, verbose=1)

# テストデータをオートエンコーダーに通す
reconstructed = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)

# MSEが大きい順に並べた画像ファイル名のリストを取得
mse_with_filenames = sorted(zip(mse, image_positives_paths), key=lambda x: x[0], reverse=True)
mse_filenames_sorted = [os.path.basename(filename).replace(".png", "") for _, filename in mse_with_filenames]

# MSEの大きい順のファイル名を表示
print("MSEが大きい順の画像ファイル名:")
print(mse_filenames_sorted)
a = ['11', '28', '3', '48', '26', '1', '124', '101', '32', '49', '15', '19', '62', '16', '110', '5', '12', '35', '25', '2', '94', '63', '21', '86', '60', '145', '117', '31', '20', '154', '0', '178', '47', '64', '92', '55', '7', '9', '135', '17', '57', '143', '33', '37', '93', '134', '175', '8', '30', '84', '65', '75', '149', '146', '138', '123', '39', '6', '22', '129', '179', '137', '168', '114', '177', '112', '53', '46', '69', '105', '163', '176', '23', '51', '157', '70', '108', '132', '91', '98', '107', '111', '155', '42', '158', '72', '102', '151', '4', '36', '73', '125', '161', '174', '83', '139', '61', '29', '133', '119', '136', '79', '80', '126', '77', '150', '103', '165', '160', '104', '144', '96', '118', '142', '140', '171', '27', '170', '130', '156', '59', '172', '106', '180', '116', '99', '43', '109', '71', '81', '167', '122', '90', '153', '38', '68', '82', '56', '152', '34', '89', '159', '100', '44', '147', '127', '169', '97', '131', '164', '76', '173', '162', '141', '66', '87', '148', '50', '54', '113', '121', '120', '45', '41', '128', '52', '67', '115', '40', '95', '74', '85', '58', '88', '78', '24', '13', '166', '10', '14', '18']