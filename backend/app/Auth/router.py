from fastapi import APIRouter, Security
from fastapi.responses import JSONResponse
from Auth.crud import Auth

router_auth = APIRouter(prefix="/auth", tags=["auth"])


@router_auth.post("/generate_password_hash")
def generate_password_hash(password: str):
    password_hash = Auth.hash_password(password)
    return JSONResponse(content={"password_hash": password_hash})


@router_auth.post("/login")
def login(plain_password: str, hashed_password: str):
    is_verified = Auth.verify_password(plain_password, hashed_password)
    return JSONResponse(content={"is_verified": is_verified})


@router_auth.get("/account")
def protected_route(account: str = Security(Auth.get_account)):
    return JSONResponse(content={"account": account})
