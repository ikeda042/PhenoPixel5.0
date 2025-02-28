from datetime import datetime, timedelta
from time import time
from jose import jwt, JWTError
from sqlalchemy import delete, or_, and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from ..settings import settings
from ..database.utils import get_ulid
from ..database import models
from .schemas import AccessToken, RefreshToken, Account
from .types import AccessTokenCreate, RefreshTokenCreate, TokenType
from .exceptions import InvalidRefreshToken, InvalidAccessToken


def create_access_token(data: AccessTokenCreate) -> tuple[str, AccessToken]:
    to_encode = {**data}
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_exp)
    jti = get_ulid()
    to_encode.update({"exp": expire, "jti": jti, "type": TokenType.access})
    encoded_jwt: str = jwt.encode(to_encode, settings.jwt_secret, algorithm="HS256")
    return encoded_jwt, AccessToken(**to_encode)


def create_refresh_token(
    data: RefreshTokenCreate, expire_limit: datetime | None = None
) -> tuple[str, RefreshToken]:
    to_encode = {**data}
    expire = datetime.utcnow() + timedelta(minutes=settings.refresh_token_exp)
    if expire_limit is not None:
        expire = expire if expire < expire_limit else expire_limit
    jti = get_ulid()
    to_encode.update({"exp": expire, "jti": jti, "type": TokenType.refresh})
    encoded_jwt: str = jwt.encode(to_encode, settings.jwt_secret, algorithm="HS256")
    return encoded_jwt, RefreshToken(**to_encode)


def create_access_token_from_account(account: Account) -> str:
    access_token, _ = create_access_token(
        data={
            "sub": account.id,
            "scopes": list(account.scopes),
            "hid": account.handle_id,
        }
    )
    return access_token


async def create_refresh_token_from_account(
    db: AsyncSession, account: Account, expire_limit: datetime | None
) -> str:
    refresh_token, refresh_token_payload = create_refresh_token(
        data={
            "sub": account.id,
            "scopes": list(account.scopes),
            "hid": account.handle_id,
        },
        expire_limit=expire_limit,
    )
    db.add(
        models.RefreshToken(
            id=refresh_token_payload.jti,
            exp=refresh_token_payload.exp,
            user_id=refresh_token_payload.sub,
        )
    )
    await db.commit()
    return refresh_token


def parse_and_validate_access_token(token: str) -> AccessToken:
    try:
        payload = AccessToken(
            **jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        )
    except JWTError:
        raise InvalidAccessToken
    return payload


async def parse_and_validate_refresh_token(
    db: AsyncSession, token: str
) -> RefreshToken:
    try:
        payload = RefreshToken(
            **jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        )
    except JWTError:
        raise InvalidRefreshToken
    stmt = select(select(models.RefreshToken.id).filter_by(id=payload.jti).exists())
    ex = (await db.execute(stmt)).scalar()
    if ex is not True:
        raise InvalidRefreshToken
    return payload


async def invalidate_refresh_token(
    db: AsyncSession, refresh_token: RefreshToken
) -> None:
    await db.execute(
        delete(models.RefreshToken)
        .where(
            or_(
                and_(
                    models.RefreshToken.user_id == refresh_token.sub,
                    models.RefreshToken.exp < int(time()),
                ),
                models.RefreshToken.id == refresh_token.jti,
            )
        )
        .execution_options(synchronize_session=False)
    )
    await db.commit()
