from fastapi import APIRouter, Depends
from .schemas import Tokens
from .login_manager import authorize
from .crud import UserCrud

router_oauth2 = APIRouter(prefix="/oauth2", tags=["oauth2"])


@router_oauth2.post(
    "/token", response_model=Tokens, description="トークン取得用エンドポイント"
)
async def get_token(token=Depends(authorize)):
    return token


@router_oauth2.post("/register", description="ユーザー登録用エンドポイント")
async def register_user(user=Depends(UserCrud.create)):
    return user
