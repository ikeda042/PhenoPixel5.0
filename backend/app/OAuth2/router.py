from fastapi import APIRouter, Depends
from .schemas import Tokens, UserCreate
from .login_manager import authorize
from .crud import UserCrud
from .database import get_session

router_oauth2 = APIRouter(prefix="/oauth2", tags=["oauth2"])


@router_oauth2.post(
    "/token", response_model=Tokens, description="トークン取得用エンドポイント"
)
async def get_token(token=Depends(authorize)):
    return token


@router_oauth2.post("/register", description="ユーザー登録用エンドポイント")
async def register(user: UserCreate, session=Depends(get_session)):
    await UserCrud.create(session, **user.dict())
    user = await UserCrud.get_by_handle(session, user.handle_id)
    return user
