import cv2
import numpy as np
import os

files = [
    i
    for i in os.listdir("experimental/detect_dot_test_field")
    if i.endswith(".png") and not i.endswith("_thresh.png")
]


def detect_dot(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)

    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    median_val = np.median(norm_gray)
    top5_val = np.percentile(norm_gray, 97)
    diff = top5_val - median_val
    print(diff)
    dot_diff_threshold = 70

    if diff > dot_diff_threshold:
        # ドットがある場合：しきい値180で２値化
        ret, thresh = cv2.threshold(norm_gray, 180, 255, cv2.THRESH_BINARY)
    else:
        # ドットがない場合：全て黒の画像を生成
        thresh = np.zeros_like(norm_gray)

    # 結果の保存（元ファイル名に _thresh を付加）
    cv2.imwrite(f"{image_path[:-4]}_thresh.png", thresh)
    return thresh


if __name__ == "__main__":
    for file in files:
        detect_dot(f"experimental/detect_dot_test_field/{file}")
