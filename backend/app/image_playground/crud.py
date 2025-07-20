import asyncio
from concurrent.futures import ThreadPoolExecutor
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
