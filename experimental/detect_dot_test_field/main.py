import cv2
import numpy as np
import os

files = [
    i
    for i in os.listdir("experimental/detect_dot_test_field")
    if i.endswith(".png") and not i.endswith("_thresh.png")
]


def detect_dot(image_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    image = cv2.imread(image_path)

    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    median_val = np.median(norm_gray)
    top5_val = np.percentile(norm_gray, 97)
    diff = top5_val - median_val
    print(f"diff: {diff}")
    dot_diff_threshold = 70

    if diff > dot_diff_threshold:
        # ドットがある場合：しきい値180で２値化
        ret, thresh = cv2.threshold(norm_gray, 180, 255, cv2.THRESH_BINARY)
    else:
        # ドットがない場合：全て黒の画像を生成
        thresh = np.zeros_like(norm_gray)

    # 結果の保存（元ファイル名に _thresh を付加）
    cv2.imwrite(f"{image_path[:-4]}_thresh.png", thresh)

    # 輪郭の検出（外部輪郭を抽出）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    closed_contour = None
    if contours:
        # 最大の輪郭を選択
        cnt = max(contours, key=cv2.contourArea)
        # 輪郭の周長を計算
        arc_len = cv2.arcLength(cnt, True)
        # 近似精度の設定
        epsilon = 0.01 * arc_len  # 近似精度の計算: epsilon = 0.01 * arcLength(cnt)
        # LaTeXの生コード:
        # \epsilon = 0.01 \times \text{arcLength}(cnt)
        # 輪郭の近似（閉じた輪郭となる）
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        closed_contour = approx

        # 輪郭を画像に描画して保存（デバッグ用）
        image_contour = image.copy()
        cv2.drawContours(image_contour, [closed_contour], -1, (0, 255, 0), 2)
        cv2.imwrite(f"{image_path[:-4]}_contour.png", image_contour)
    else:
        print("輪郭が見つかりませんでした")

    return thresh, closed_contour


if __name__ == "__main__":
    for file in files:
        thresh, contour = detect_dot(f"experimental/detect_dot_test_field/{file}")
