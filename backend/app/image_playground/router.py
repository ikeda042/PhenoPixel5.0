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
