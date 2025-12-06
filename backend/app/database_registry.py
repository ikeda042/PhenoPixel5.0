import asyncio
import os
from pathlib import Path
from typing import Iterable

import aiosqlite


class DatabaseRegistry:
    """
    Lightweight registry of available databases to avoid repeatedly scanning
    the entire databases/ directory when listing entries.
    """

    REGISTRY_FILENAME = "databases.db"
    _DATABASE_TABLE = "database_names"
    _METADATA_TABLE = "metadata"
    _DIR_MTIME_KEY = "dir_mtime"
    _lock: asyncio.Lock | None = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def _db_dir(cls) -> Path:
        return Path(__file__).resolve().parent / "databases"

    @classmethod
    def _registry_path(cls) -> Path:
        return cls._db_dir() / cls.REGISTRY_FILENAME

    @classmethod
    def _ensure_dir(cls) -> None:
        cls._db_dir().mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_dir_mtime(cls) -> float:
        try:
            return os.path.getmtime(cls._db_dir())
        except FileNotFoundError:
            return 0.0

    @classmethod
    def _scan_filesystem(cls) -> list[str]:
        """
        Return .db files inside databases/ excluding the registry itself.
        """
        db_dir = cls._db_dir()
        if not db_dir.exists():
            return []

        entries: list[str] = []
        with os.scandir(db_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if not entry.name.endswith(".db"):
                    continue
                if entry.name == cls.REGISTRY_FILENAME:
                    continue
                entries.append(entry.name)
        entries.sort()
        return entries

    @classmethod
    async def _connect(cls) -> aiosqlite.Connection:
        cls._ensure_dir()
        conn = await aiosqlite.connect(cls._registry_path())
        await conn.execute("PRAGMA journal_mode=WAL;")
        await cls._ensure_schema(conn)
        return conn

    @classmethod
    async def _ensure_schema(cls, conn: aiosqlite.Connection) -> None:
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {cls._DATABASE_TABLE} (
                name TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {cls._METADATA_TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        await conn.commit()

    @classmethod
    async def list_databases(cls) -> list[str]:
        """
        Return cached database names stored in databases/databases.db.
        The registry is refreshed when the directory mtime changes or when
        the registry is empty.
        """
        lock = cls._get_lock()
        async with lock:
            conn = await cls._connect()
            try:
                dir_mtime = cls._get_dir_mtime()
                names = await cls._get_names(conn)
                stored_mtime = await cls._get_metadata(conn, cls._DIR_MTIME_KEY)

                needs_rescan = stored_mtime is None or not names
                if stored_mtime is not None:
                    try:
                        if abs(float(stored_mtime) - dir_mtime) > 1e-9:
                            needs_rescan = True
                    except ValueError:
                        needs_rescan = True

                if needs_rescan:
                    names = await cls._rescan(conn, dir_mtime)
                return names
            finally:
                await conn.close()

    @classmethod
    async def register_database(cls, name: str) -> None:
        """
        Insert or update a single database name in the registry.
        """
        if name == cls.REGISTRY_FILENAME:
            return

        lock = cls._get_lock()
        async with lock:
            conn = await cls._connect()
            try:
                await cls._upsert_names(conn, [name])
                await cls._update_dir_mtime(conn, cls._get_dir_mtime())
                await conn.commit()
            finally:
                await conn.close()

    @classmethod
    async def sync_from_filesystem(cls) -> list[str]:
        """
        Force a full rescan of databases/ and rewrite the registry.
        """
        lock = cls._get_lock()
        async with lock:
            conn = await cls._connect()
            try:
                names = cls._scan_filesystem()
                await conn.execute(f"DELETE FROM {cls._DATABASE_TABLE}")
                await cls._upsert_names(conn, names)
                await cls._update_dir_mtime(conn, cls._get_dir_mtime())
                await conn.commit()
                return names
            finally:
                await conn.close()

    @classmethod
    async def replace_database(cls, old_name: str, new_name: str) -> None:
        """
        Replace one entry in the registry (e.g., when renaming a DB).
        """
        if new_name == cls.REGISTRY_FILENAME:
            return

        lock = cls._get_lock()
        async with lock:
            conn = await cls._connect()
            try:
                await conn.execute(
                    f"DELETE FROM {cls._DATABASE_TABLE} WHERE name = ?", (old_name,)
                )
                await cls._upsert_names(conn, [new_name])
                await cls._update_dir_mtime(conn, cls._get_dir_mtime())
                await conn.commit()
            finally:
                await conn.close()

    @classmethod
    async def remove_database(cls, name: str) -> None:
        """
        Remove an entry from the registry (used if a DB is deleted).
        """
        if name == cls.REGISTRY_FILENAME:
            return

        lock = cls._get_lock()
        async with lock:
            conn = await cls._connect()
            try:
                await conn.execute(
                    f"DELETE FROM {cls._DATABASE_TABLE} WHERE name = ?", (name,)
                )
                await cls._update_dir_mtime(conn, cls._get_dir_mtime())
                await conn.commit()
            finally:
                await conn.close()

    @classmethod
    async def _rescan(cls, conn: aiosqlite.Connection, dir_mtime: float) -> list[str]:
        names = cls._scan_filesystem()
        await conn.execute(f"DELETE FROM {cls._DATABASE_TABLE}")
        await cls._upsert_names(conn, names)
        await cls._update_dir_mtime(conn, dir_mtime)
        await conn.commit()
        return names

    @classmethod
    async def _upsert_names(
        cls, conn: aiosqlite.Connection, names: Iterable[str]
    ) -> None:
        filtered_names = [name for name in names if name != cls.REGISTRY_FILENAME]
        if not filtered_names:
            return
        await conn.executemany(
            f"""
            INSERT INTO {cls._DATABASE_TABLE} (name, updated_at)
            VALUES (?, CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            """,
            ((name,) for name in filtered_names),
        )

    @classmethod
    async def _get_names(cls, conn: aiosqlite.Connection) -> list[str]:
        cursor = await conn.execute(
            f"SELECT name FROM {cls._DATABASE_TABLE} ORDER BY name"
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    @classmethod
    async def _get_metadata(cls, conn: aiosqlite.Connection, key: str) -> str | None:
        cursor = await conn.execute(
            f"SELECT value FROM {cls._METADATA_TABLE} WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    @classmethod
    async def _update_dir_mtime(
        cls, conn: aiosqlite.Connection, dir_mtime: float
    ) -> None:
        await cls._set_metadata(conn, cls._DIR_MTIME_KEY, str(dir_mtime))

    @classmethod
    async def _set_metadata(
        cls, conn: aiosqlite.Connection, key: str, value: str
    ) -> None:
        await conn.execute(
            f"""
            INSERT INTO {cls._METADATA_TABLE} (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
