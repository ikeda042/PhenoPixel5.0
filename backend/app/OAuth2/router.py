from fastapi import APIRouter, Depends
from .schemas import Tokens, UserCreate
from .login_manager import authorize, get_account
from .crud import UserCrud
from .database import get_session
from fastapi.security import SecurityScopes

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


@router_oauth2.get("/protected", description="保護されたエンドポイント")
async def protected(account=Depends(get_account)):
    return {"account": account.dict()}


@router_oauth2.get(
    "/protected_optional", description="保護されたエンドポイント(オプション)"
)
async def protected_optional(account=Depends(get_account)):
    if account is None:
        return {"account": None}
    return {"account": account.dict()}


@router_oauth2.get("/me", description="自分の情報を取得するエンドポイント")
async def me(account=Depends(get_account)):
    return {"account": account.dict()}
