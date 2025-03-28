from datetime import datetime, timedelta
from time import time
from jose import jwt, JWTError
from sqlalchemy import delete, or_, and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from settings import settings
from .database import get_ulid
from .database import RefreshToken as RefreshTokenModel
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
            "scopes": list(account.scopes),  # ここで list に変換していますが…
            "hid": account.handle_id,
        },
        expire_limit=expire_limit,
    )
    # refresh_token_payload.scopes が list だけでなく set の場合にも対応するため、
    # どちらの場合も list に変換してカンマ区切りの文字列に変換
    scopes_str = (
        ",".join(list(refresh_token_payload.scopes))
        if isinstance(refresh_token_payload.scopes, (list, set))
        else refresh_token_payload.scopes
    )
    db.add(
        RefreshTokenModel(
            id=refresh_token_payload.jti,
            exp=refresh_token_payload.exp,
            user_id=refresh_token_payload.sub,
            scopes=scopes_str,
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
    stmt = select(select(RefreshTokenModel.id).filter_by(id=payload.jti).exists())
    ex = (await db.execute(stmt)).scalar()
    if ex is not True:
        raise InvalidRefreshToken
    return payload


async def invalidate_refresh_token(
    db: AsyncSession, refresh_token: RefreshToken
) -> None:
    await db.execute(
        delete(RefreshTokenModel)
        .where(
            or_(
                and_(
                    RefreshTokenModel.user_id == refresh_token.sub,
                    RefreshTokenModel.exp < int(time()),
                ),
                RefreshTokenModel.id == refresh_token.jti,
            )
        )
        .execution_options(synchronize_session=False)
    )
    await db.commit()
