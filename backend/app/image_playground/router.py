import io
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from .crud import ImagePlaygroundCrud

router_image_playground = APIRouter(prefix="/image_playground", tags=["image_playground"])


@router_image_playground.post("/canny")
async def canny_image(file: UploadFile, threshold1: int = 100, threshold2: int = 200):
    try:
        data = await file.read()
        processed = await ImagePlaygroundCrud.canny(data, threshold1, threshold2)
        return StreamingResponse(io.BytesIO(processed), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_image_playground.post("/sobel")
async def sobel_image(
    file: UploadFile,
    ksize: int = 3,
    dx: int = 1,
    dy: int = 1,
):
    try:
        data = await file.read()
        processed = await ImagePlaygroundCrud.sobel(data, ksize, dx, dy)
        return StreamingResponse(io.BytesIO(processed), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_image_playground.post("/gaussian")
async def gaussian_image(
    file: UploadFile,
    ksize: int = 5,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
):
    try:
        data = await file.read()
        processed = await ImagePlaygroundCrud.gaussian(data, ksize, sigma_x, sigma_y)
        return StreamingResponse(io.BytesIO(processed), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_image_playground.post("/histogram")
async def histogram_image(
    file: UploadFile,
    bins: int = 256,
    normalize: bool = False,
):
    try:
        data = await file.read()
        processed = await ImagePlaygroundCrud.histogram(data, bins, normalize)
        return StreamingResponse(io.BytesIO(processed), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
