import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import cv2
import numpy as np

class ImagePlaygroundCrud:
    @staticmethod
    def _apply_canny(data: bytes, threshold1: int, threshold2: int) -> bytes:
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        success, buffer = cv2.imencode(".png", edges)
        if not success:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()

    @classmethod
    async def canny(cls, data: bytes, threshold1: int, threshold2: int) -> bytes:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, cls._apply_canny, data, threshold1, threshold2
            )

    @staticmethod
    def _apply_sobel(data: bytes, ksize: int, dx: int, dy: int) -> bytes:
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        sobel = cv2.convertScaleAbs(sobel)
        success, buffer = cv2.imencode(".png", sobel)
        if not success:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()

    @classmethod
    async def sobel(cls, data: bytes, ksize: int, dx: int, dy: int) -> bytes:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, cls._apply_sobel, data, ksize, dx, dy
            )

    @staticmethod
    def _apply_gaussian(data: bytes, ksize: int, sigma_x: float, sigma_y: float) -> bytes:
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma_x, sigma_y)
        success, buffer = cv2.imencode(".png", blurred)
        if not success:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()

    @classmethod
    async def gaussian(cls, data: bytes, ksize: int, sigma_x: float, sigma_y: float) -> bytes:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, cls._apply_gaussian, data, ksize, sigma_x, sigma_y
            )

    @staticmethod
    def _apply_histogram(data: bytes, bins: int, normalize: bool) -> bytes:
        import matplotlib.pyplot as plt
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image data")
        hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
        if normalize:
            hist = hist / hist.sum() if hist.sum() > 0 else hist
        fig = plt.figure(figsize=(4, 3))
        plt.plot(hist, color="gray")
        plt.xlim([0, bins])
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

    @classmethod
    async def histogram(cls, data: bytes, bins: int, normalize: bool) -> bytes:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, cls._apply_histogram, data, bins, normalize
            )

    @staticmethod
    def _apply_cell_contour(
        data: bytes, threshold: int, min_area: int
    ) -> bytes:
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh, 0, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        success, buffer = cv2.imencode(".png", img)
        if not success:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()

    @classmethod
    async def cell_contour(
        cls, data: bytes, threshold: int, min_area: int
    ) -> bytes:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, cls._apply_cell_contour, data, threshold, min_area
            )
