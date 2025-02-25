import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread("experimental/detect_dot_test_field/image.png")
if image is None:
    raise FileNotFoundError("image.pngが見つかりません")

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 輝度の正規化: グレースケール画像の最小値と最大値を用いて0～255に正規化
norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


ret, thresh = cv2.threshold(norm_gray, 200, 255, cv2.THRESH_BINARY)

# 　結果を保存
cv2.imwrite("experimental/detect_dot_test_field/result.png", thresh)
