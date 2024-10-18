import cv2
import numpy as np
import json

# 画像をグレースケールで読み込む
image = cv2.imread("experimental/3d/test.png", cv2.IMREAD_GRAYSCALE)

height, width = image.shape

point_cloud = []

for y in range(height):
    for x in range(width):
        z = image[y, x]
        if z > 15:
            point_cloud.append([x, y, z])

point_cloud = np.array(point_cloud)

# point_cloudをCSVファイルに保存
np.savetxt("experimental/3d/point_cloud.csv", point_cloud, delimiter=",", fmt="%d")
with open("experimental/3d/point_cloud.json", "w") as f:
    json.dump([{"x": int(x), "y": int(y), "z": int(z)} for x, y, z in point_cloud], f)
# point_cloudをプロット
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# マーカーサイズを最小に設定
ax.scatter(
    point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="r", marker="o", s=1
)

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")

plt.show()
