from fastapi import APIRouter, Depends
from .schemas import Tokens
from .login_manager import authorize


router_oauth2 = APIRouter(prefix="/oauth2", tags=["oauth2"])


@router_oauth2.post(
    "/token", response_model=Tokens, description="トークン取得用エンドポイント"
)
async def get_token(token=Depends(authorize)):
    return token
