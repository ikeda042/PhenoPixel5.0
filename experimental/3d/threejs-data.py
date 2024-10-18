import cv2
import numpy as np


image = cv2.imread("microscope_image.png", cv2.IMREAD_GRAYSCALE)

height, width = image.shape

point_cloud = []

for y in range(height):
    for x in range(width):
        z = image[y, x]
        point_cloud.append([x, y, z])

point_cloud = np.array(point_cloud)

np.savetxt("point_cloud.csv", point_cloud, delimiter=",", fmt="%d")
