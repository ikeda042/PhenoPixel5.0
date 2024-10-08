from __future__ import annotations
from CellDBConsole.schemas import CellId, CellMorhology, ListDBresponse
from database import get_session, Cell
from sqlalchemy.future import select
from exceptions import CellNotFoundError
import cv2
import numpy as np
from numpy.linalg import eig, inv
from fastapi.responses import StreamingResponse
import io
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from dataclasses import dataclass
from fastapi import UploadFile
import aiofiles
import os
import pandas as pd
from sqlalchemy import update
import shutil
from typing import Literal


class AsyncChores:
    @staticmethod
    async def async_imdecode(data: bytes) -> np.ndarray:
        """
        Decode an image from bytes.

        Parameters:
        - data: Image data in bytes.

        Returns:
        - Image in numpy array format.
        """
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            img = await loop.run_in_executor(
                executor, cv2.imdecode, np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
            )
        return img

    @staticmethod
    async def async_cv2_imencode(img) -> tuple[bool, np.ndarray]:
        """
        Encode an image to PNG format.

        Parameters:
        - img: Image to encode.

        Returns:
        - Tuple containing success status and image buffer.
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            success, buffer = await loop.run_in_executor(
                executor, lambda: cv2.imencode(".png", img)
            )
        return success, buffer
