import os
import asyncio
import aiohttp
import logging
import contextlib
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from OAuth2.database import BaseAuth, dispose_auth_engine, get_engine
from settings import settings

from CellAI.router import router_cell_ai
from AutoLabel.router import router_autolabel
from CellDBConsole.router import router_cell, router_database
from CellExtraction.router import router_cell_extraction
from Dev.router import router_dev
from GraphEngine.router import router_graphengine
from CDT.router import router_cdt
from TimeLapseEngine.router import router_tl_engine
from results.router import router_results
from FileManager.router import router_file_manager
from image_playground.router import router_image_playground
from Auth.router import router_auth
from Admin.router import router_admin
from OAuth2.router import router_oauth2
from settings import settings
from OAuth2.crud import UserCrud
from sqlalchemy.ext.asyncio import AsyncSession
from TimeLapseEngine.crud import shutdown_timelapse_process_pool
from database import dispose_cached_engines
from database_registry import DatabaseRegistry

api_title = settings.API_TITLE
api_prefix = settings.API_PREFIX
test_env = settings.TEST_ENV

logger = logging.getLogger(__name__)
_watchdog_task: asyncio.Task | None = None


app = FastAPI(
    title=api_title,
    docs_url=f"{api_prefix}/docs",
    openapi_url=f"{api_prefix}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(BaseAuth.metadata.create_all)


@app.on_event("startup")
async def startup_event():
    await init_db()
    engine = get_engine()

    async with AsyncSession(engine) as session:
        existing_user = await UserCrud.get_by_handle(session, settings.admin_handle_id)
        if existing_user is None:
            try:
                await UserCrud.create(
                    session,
                    handle_id=settings.admin_handle_id,
                    password=settings.admin_password,
                    is_admin=True,
                )
                print(f"Default user created with handle: {settings.admin_handle_id}")
            except Exception as e:
                print(f"Failed to create default user: {e}")
        else:
            print(f"Default user with handle {settings.admin_handle_id} already exists")
    await DatabaseRegistry.sync_from_filesystem()

    if settings.watchdog_enabled:
        global _watchdog_task
        _watchdog_task = asyncio.create_task(event_loop_watchdog())
        logger.info(
            "Watchdog started (interval=%.2fs, max_delay=%.2fs, threshold=%d)",
            settings.watchdog_interval_sec,
            settings.watchdog_max_delay_sec,
            settings.watchdog_failure_threshold,
        )


@app.on_event("shutdown")
async def on_shutdown():
    await shutdown_timelapse_process_pool()
    await dispose_cached_engines()
    await dispose_auth_engine()
    global _watchdog_task
    if _watchdog_task:
        _watchdog_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _watchdog_task
        _watchdog_task = None


@app.get("/api")
async def root():
    return {
        "title": api_title,
        "api_prefix": api_prefix,
        "docs_url": f"{api_prefix}/docs",
        "openapi_url": f"{api_prefix}/openapi.json",
        "status": "ok",
        "code": 200,
    }


@app.get(f"{api_prefix}/healthcheck")
async def healthcheck():
    return {"status": "ok"}


@app.get(f"{api_prefix}/env")
async def get_env():
    return {
        "API_TITLE": api_title,
        "API_PREFIX": api_prefix,
        "TEST_ENV": test_env,
        "Slack_Webhook_URL": settings.slack_webhook_url,
    }


async def check_internet_connection() -> bool:
    url = settings.internet_healthcheck_url
    timeout = aiohttp.ClientTimeout(total=settings.internet_healthcheck_timeout)
    connector = aiohttp.TCPConnector(ssl=settings.internet_healthcheck_verify_ssl)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        try:
            async with session.get(url) as response:
                return 200 <= response.status < 400
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False


@app.get(f"{api_prefix}/internet-connection")
async def internet_connection():
    return {"status": await check_internet_connection()}


async def event_loop_watchdog() -> None:
    """
    Periodically check event-loop latency; if it stalls repeatedly, exit so
    the process manager can restart the server.
    """
    interval = max(0.1, settings.watchdog_interval_sec)
    max_delay = max(0.0, settings.watchdog_max_delay_sec)
    threshold = max(1, settings.watchdog_failure_threshold)
    loop = asyncio.get_running_loop()
    consecutive = 0

    while True:
        start = loop.time()
        await asyncio.sleep(interval)
        elapsed = loop.time() - start
        delay = elapsed - interval

        if delay > max_delay:
            consecutive += 1
            logger.warning(
                "Watchdog: event loop delay %.3fs (> %.3fs) [%d/%d]",
                delay,
                max_delay,
                consecutive,
                threshold,
            )
            if consecutive >= threshold:
                logger.error(
                    "Watchdog: event loop stalled; exiting to trigger restart."
                )
                os._exit(1)
        else:
            consecutive = 0


@app.post(f"{api_prefix}/replace_env")
async def replace_env(file: UploadFile):
    contents = await file.read()
    with open(".env", "wb") as f:
        f.write(contents)
    return {"status": "ok"}


app.include_router(router_admin, prefix=api_prefix)
app.include_router(router_auth, prefix=api_prefix)
app.include_router(router_dev, prefix=api_prefix)
app.include_router(router_cell, prefix=api_prefix)
app.include_router(router_database, prefix=api_prefix)
app.include_router(router_cell_extraction, prefix=api_prefix)
app.include_router(router_cell_ai, prefix=api_prefix)
app.include_router(router_tl_engine, prefix=api_prefix)
app.include_router(router_graphengine, prefix=api_prefix)
app.include_router(router_cdt, prefix=api_prefix)
app.include_router(router_results, prefix=api_prefix)
app.include_router(router_file_manager, prefix=api_prefix)
app.include_router(router_image_playground, prefix=api_prefix)
app.include_router(router_autolabel, prefix=api_prefix)
app.include_router(router_oauth2, prefix=api_prefix)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
