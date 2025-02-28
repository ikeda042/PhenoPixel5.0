from enum import StrEnum
from typing import TypedDict


class OAuthGrantType(StrEnum):
    password = "password"
    refresh_token = "refresh_token"


class TokenType(StrEnum):
    access = "access"
    refresh = "refresh"


class Scope(StrEnum):
    admin = "admin"
    me = "me"


class AccessTokenCreate(TypedDict):
    sub: str
    scopes: list[Scope]
    hid: str


class RefreshTokenCreate(TypedDict):
    sub: str
    scopes: list[Scope]
    hid: str
