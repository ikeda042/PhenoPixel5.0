import cv2
import numpy as np
import os

files = [
    i for i in os.listdir("experimental/detect_dot_test_field") if i.endswith(".png")
]


def detect_dot(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)

    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 輝度の正規化: グレースケール画像の最小値と最大値を用いて0～255に正規化
    norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    ret, thresh = cv2.threshold(norm_gray, 180, 255, cv2.THRESH_BINARY)

    # 　結果を保存 (image_pathの拡張子を除いた名前で保存)
    cv2.imwrite(f"{image_path[:-4]}_thresh.png", thresh)


if __name__ == "__main__":
    for file in files:
        detect_dot(f"experimental/detect_dot_test_field/{file}")
