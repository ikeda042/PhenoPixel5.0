import cv2
import numpy as np
import os

files = [
    i
    for i in os.listdir("experimental/detect_dot_test_field")
    if i.endswith(".png") and "_" not in i
]


def detect_dot(image_path: str) -> list[tuple[int, int]]:
    image = cv2.imread(image_path)

    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    median_val = np.median(norm_gray)
    top97_val = np.percentile(norm_gray, 97)
    diff = top97_val - median_val
    print(f"diff: {diff}")
    dot_diff_threshold = 70

    coordinates: list[tuple[int, int]] = []

    if diff > dot_diff_threshold:
        # ドットがある場合：しきい値180で2値化
        ret, thresh = cv2.threshold(norm_gray, 180, 255, cv2.THRESH_BINARY)

        # 輪郭検出（外側の輪郭のみ取得）
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            # モーメントを計算し、重心を求める
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coordinates.append((cX, cY))

        # 検出された輪郭をそのまま表示する画像を生成
        detected_img = np.zeros_like(image)  # 黒背景の画像
        cv2.drawContours(detected_img, contours, -1, (255, 255, 255), 2)

        cv2.imwrite(f"{image_path[:-4]}_detected.png", detected_img)
        cv2.imwrite(f"{image_path[:-4]}_thresh.png", thresh)
    else:
        # ドットがない場合：全て黒の画像を生成して保存
        thresh = np.zeros_like(norm_gray)
        cv2.imwrite(f"{image_path[:-4]}_thresh.png", thresh)

    return coordinates


if __name__ == "__main__":
    for file in files:
        coords = detect_dot(f"experimental/detect_dot_test_field/{file}")
        print(f"Detected dot centers in {file}: {coords}")
