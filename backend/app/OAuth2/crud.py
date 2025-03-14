from typing import Optional
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from .database import User, get_ulid
import asyncio
from .utils import ph


class UserCrud:
    @classmethod
    async def create(
        cls,
        session: AsyncSession,
        handle_id: str,
        password: str,
        lock_until: Optional[datetime] = None,
        is_admin: bool = False,
        login_fail_count: int = 0,
    ) -> User:
        """ユーザーを新規作成する (handle_idの重複チェックあり)"""
        result = await session.execute(select(User).filter_by(handle_id=handle_id))
        if result.scalar_one_or_none():
            raise ValueError("User with this handle_id already exists")

        hashed_password = await asyncio.to_thread(ph.hash, password)

        user = User(
            id=get_ulid(),
            handle_id=handle_id,
            password_hash=hashed_password,
            lock_until=lock_until,
            is_admin=is_admin,
            login_fail_count=login_fail_count,
            updated_at=datetime.now(),
            created_at=datetime.now(),
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

    @classmethod
    async def change_password(
        cls,
        session: AsyncSession,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> Optional[User]:
        """
        ユーザーのパスワードを変更する

        Args:
            session (AsyncSession): データベースセッション
            user_id (str): ユーザーID
            old_password (str): 現在のパスワード
            new_password (str): 新しいパスワード

        Raises:
            ValueError: 古いパスワードが正しくない場合

        Returns:
            Optional[User]: 更新されたユーザー情報、またはユーザーが存在しない場合はNone
        """
        result = await session.execute(select(User).filter_by(id=user_id))
        user = result.scalar_one_or_none()
        if user is None:
            return None

        try:
            # 古いパスワードの検証
            await asyncio.to_thread(ph.verify, user.password_hash, old_password)
        except Exception:
            raise ValueError("古いパスワードが正しくありません")

        # 新しいパスワードのハッシュ化
        new_hashed_password = await asyncio.to_thread(ph.hash, new_password)
        user.password_hash = new_hashed_password
        user.updated_at = datetime.now()
        await session.commit()
        await session.refresh(user)
        return user
