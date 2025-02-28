from typing import Literal
from fastapi import Form
from .types import Scope, OAuthGrantType, TokenType
from pydantic import constr, BaseModel


class BaseModelImmutable(BaseModel):
    class Config:
        allow_mutation = False


class BaseModelImmutableOrm(BaseModel):
    class Config:
        allow_mutation = False
        orm_mode = True


class Account(BaseModelImmutable):
    id: str
    handle_id: str
    scopes: set[Scope] = set()


class Tokens(BaseModelImmutable):
    token_type: Literal["bearer"] = "bearer"
    access_token: str
    refresh_token: str


class AccessToken(BaseModelImmutable):
    type: Literal[TokenType.access]
    jti: str
    sub: str
    exp: int
    scopes: set[Scope]
    hid: str  # handle_id


class RefreshToken(BaseModelImmutable):
    type: Literal[TokenType.refresh]
    jti: str
    sub: str
    exp: int
    scopes: set[Scope]
    hid: str  # handle_id


class OAuth2RequestForm:
    def __init__(
        self,
        grant_type: OAuthGrantType = Form(),
        username: str | None = Form(default=None),
        password: str | None = Form(default=None),
        scope: str = Form(default=""),
        refresh_token: str | None = Form(default=None),
    ):
        self.grant_type = grant_type
        self.username = username
        self.password = password
        self.scopes = scope.split()
        self.refresh_token = refresh_token


class OAuth2PasswordRequest(BaseModelImmutableOrm):
    grant_type: Literal[OAuthGrantType.password]
    username: str
    password: str
    scopes: set[Scope]


class OAuth2RefreshRequest(BaseModelImmutableOrm):
    grant_type: Literal[OAuthGrantType.refresh_token]
    scopes: set[Scope]
    refresh_token: str


class UserCreate(BaseModel):
    handle_id: str
    password: str
