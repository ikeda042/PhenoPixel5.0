import numpy as np
import os
import nd2reader
from PIL import Image
import cv2


class SyncChores:
    @staticmethod
    def correct_drift(reference_image, target_image):
        """
        基準フレームと比較して、対象フレームのドリフトを補正する。
        """
        # ORB（特徴量検出）を用いて、フレーム間の特徴点をマッチング
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(reference_image, None)
        kp2, des2 = orb.detectAndCompute(target_image, None)

        # BFMatcherで特徴点をマッチング
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # マッチング点を基に、フレーム間の変換行列を計算
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # 対象フレームを基準フレームに合わせて変換（位置補正）
        aligned_image = cv2.warpAffine(
            target_image, matrix, (target_image.shape[1], target_image.shape[0])
        )

        return aligned_image

    @staticmethod
    def process_image(array):
        """
        画像処理関数：正規化とスケーリングを行う。
        """
        array = array.astype(np.float32)  # Convert to float
        array -= array.min()  # Normalize to 0
        array /= array.max()  # Normalize to 1
        array *= 255  # Scale to 0-255
        return array.astype(np.uint8)
