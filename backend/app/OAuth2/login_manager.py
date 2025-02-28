from __future__ import annotations
import asyncio
from datetime import datetime, timedelta
from fastapi import Depends, Header
from fastapi.security import SecurityScopes, OAuth2PasswordBearer
from pydantic import parse_obj_as
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from ..database.database import get_db
from ..settings import settings
from .schemas import (
    OAuth2RequestForm,
    OAuth2RefreshRequest,
    Tokens,
    Account,
    OAuth2PasswordRequest,
)
from .token_manager import (
    invalidate_refresh_token,
    parse_and_validate_refresh_token,
    parse_and_validate_access_token,
    create_access_token_from_account,
    create_refresh_token_from_account,
)
from .exceptions import (
    InvalidTokenRequest,
    AccountLocked,
    InvalidPassword,
    NoToken,
    NotEnoughPermissions,
    InvalidOrigin,
)
from .types import Scope
from .exceptions import UserNotFound
from ..database import models

auth_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_path}/token",
    auto_error=False,
    scopes={"admin": Scope.admin, "me": Scope.me},
)


async def auth_form(form_data: OAuth2RequestForm = Depends()):
    return parse_obj_as(OAuth2PasswordRequest | OAuth2RefreshRequest, form_data)


ph = PasswordHasher()


async def verify_password(
    password_hash: str, password: str, lock_until: datetime | None
):
    if lock_until is not None and lock_until > datetime.now():
        raise AccountLocked
    try:
        await asyncio.to_thread(ph.verify, password_hash, password)
    except VerifyMismatchError:
        raise InvalidPassword


async def verify_user(db: AsyncSession, handle_id: str, password: str) -> str:
    user = (
        await db.scalars(select(models.User).where(models.User.handle_id == handle_id))
    ).one_or_none()
    if user is None:
        raise UserNotFound
    user_id = user.id
    try:
        await verify_password(user.password_hash, password, user.lock_until)
        user.login_fail_count = 0
        user.lock_until = None
    except (AccountLocked, InvalidPassword) as err:
        user.login_fail_count = user.login_fail_count + 1
        if user.login_fail_count >= settings.login_fail_count_max:
            user.lock_until = datetime.now() + timedelta(hours=settings.login_lock_hour)
        await db.commit()
        raise err
    await db.commit()
    return user_id


async def create_account(db: AsyncSession, user_id: str, scopes: set[Scope]) -> Account:
    user = await db.get(models.User, user_id)
    if user is None:
        raise UserNotFound

    is_admin = user.is_admin

    account_scopes = set()

    if Scope.admin in scopes and is_admin:
        account_scopes.add(Scope.admin)
    if Scope.me in scopes:
        account_scopes.add(Scope.me)

    return Account(id=user.id, handle_id=user.handle_id, scopes=account_scopes)


async def authorize(
    form_data: OAuth2PasswordRequest | OAuth2RefreshRequest = Depends(auth_form),
    db: AsyncSession = Depends(get_db),
    origin: str = Header(),
) -> Tokens:
    user_id: str
    allowed_scopes: set[Scope] | None = None
    exp_limit: datetime | None = None

    if origin != settings.server_origin:
        raise InvalidOrigin

    # Parse form data
    if isinstance(form_data, OAuth2PasswordRequest):
        user_id = await verify_user(db, form_data.username, form_data.password)
    elif isinstance(form_data, OAuth2RefreshRequest):
        refresh_token = await parse_and_validate_refresh_token(
            db, form_data.refresh_token
        )
        await invalidate_refresh_token(db, refresh_token)
        allowed_scopes = set(refresh_token.scopes)
        exp_limit = datetime.fromtimestamp(refresh_token.exp)
        user_id = refresh_token.sub
    else:
        raise InvalidTokenRequest

    required_scopes = form_data.scopes
    if allowed_scopes is not None:
        if len(required_scopes) == 0:
            required_scopes = allowed_scopes
        else:
            required_scopes = allowed_scopes & required_scopes

    account = await create_account(db, user_id, required_scopes)
    # Create tokens and save to DB
    access_token = create_access_token_from_account(account)
    refresh_token = await create_refresh_token_from_account(db, account, exp_limit)
    return Tokens(access_token=access_token, refresh_token=refresh_token)


async def get_account(
    security_scopes: SecurityScopes, token: str | None = Depends(auth_scheme)
) -> Account:
    if token is None:
        raise NoToken
    payload = parse_and_validate_access_token(token)
    for scope in security_scopes.scopes:
        if scope not in payload.scopes:
            raise NotEnoughPermissions
    return Account(id=payload.sub, handle_id=payload.hid, scopes=payload.scopes)


async def get_account_optional(
    security_scopes: SecurityScopes, token: str | None = Depends(auth_scheme)
) -> Account | None:
    if token:
        return await get_account(security_scopes, token)
    return None
