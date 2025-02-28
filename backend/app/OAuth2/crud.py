from typing import Optional
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from .database import (
    User,
    get_ulid,
)


class UserCrud:
    @classmethod
    async def create(
        cls,
        session: AsyncSession,
        handle_id: str,
        password_hash: str,
        lock_until: Optional[datetime] = None,
        is_admin: bool = False,
        login_fail_count: int = 0,
    ) -> User:
        """ユーザーを新規作成する"""
        user = User(
            id=get_ulid(),
            handle_id=handle_id,
            password_hash=password_hash,
            lock_until=lock_until,
            is_admin=is_admin,
            login_fail_count=login_fail_count,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

    @classmethod
    async def get_by_id(cls, session: AsyncSession, user_id: str) -> Optional[User]:
        """ユーザーIDからユーザーを取得する"""
        result = await session.execute(select(User).filter_by(id=user_id))
        user = result.scalar_one_or_none()
        return user

    @classmethod
    async def get_by_handle(
        cls, session: AsyncSession, handle_id: str
    ) -> Optional[User]:
        """handle_idからユーザーを取得する"""
        result = await session.execute(select(User).filter_by(handle_id=handle_id))
        user = result.scalar_one_or_none()
        return user

    @classmethod
    async def update(
        cls, session: AsyncSession, user_id: str, **kwargs
    ) -> Optional[User]:
        """
        ユーザーの指定フィールドを更新する
        存在しないキーは無視されるので、Userモデルに定義されたカラムのみが更新対象となる
        """
        result = await session.execute(select(User).filter_by(id=user_id))
        user = result.scalar_one_or_none()
        if user is None:
            return None
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        await session.commit()
        await session.refresh(user)
        return user

    @classmethod
    async def delete(cls, session: AsyncSession, user_id: str) -> bool:
        """ユーザーを削除する。削除できた場合はTrue、ユーザーが存在しなければFalseを返す"""
        result = await session.execute(select(User).filter_by(id=user_id))
        user = result.scalar_one_or_none()
        if user is None:
            return False
        await session.delete(user)
        await session.commit()
        return True
