from fastapi import APIRouter
from fastapi.responses import JSONResponse
from Auth.crud import AuthCrud

router_auth = APIRouter(prefix="/auth", tags=["auth"])


@router_auth.post("/generate_password_hash")
def generate_password_hash(password: str):
    password_hash = AuthCrud.hash_password(password)
    return JSONResponse(content={"password_hash": password_hash})


@router_auth.post("/login")
def verify_password(plain_password: str, hashed_password: str):
    is_verified = AuthCrud.verify_password(plain_password, hashed_password)
    return JSONResponse(content={"is_verified": is_verified})
