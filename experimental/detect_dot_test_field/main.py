import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread("image.png")
if image is None:
    raise FileNotFoundError("image.pngが見つかりません")

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 輝度が高い画素を抽出するための閾値処理
# 式: I(x, y) > T  （ここでは T = 200）
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# LaTeXコード:
# I(x, y) > T

# 輪郭抽出
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# 輪郭を描画（緑色）
image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

# 　結果を保存
cv2.imwrite("image_contours.png", image_contours)
