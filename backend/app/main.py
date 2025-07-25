import os
import aiohttp
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine
from OAuth2.database import BaseAuth
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

api_title = settings.API_TITLE
api_prefix = settings.API_PREFIX
test_env = settings.TEST_ENV


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
    dbname = "users.db"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(base_dir, "OAuth2")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, dbname)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}?timeout=30", echo=False
    )
    async with engine.begin() as conn:
        await conn.run_sync(BaseAuth.metadata.create_all)
    await engine.dispose()


@app.on_event("startup")
async def startup_event():
    await init_db()
    dbname = "users.db"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(base_dir, "OAuth2")
    db_path = os.path.join(db_dir, dbname)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}?timeout=30", echo=False
    )

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
    await engine.dispose()


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
    }


async def check_internet_connection():
    url = "https://www.google.com"
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url):
                return True
        except aiohttp.ClientError:
            return False


@app.get(f"{api_prefix}/internet-connection")
async def internet_connection():
    return {"status": await check_internet_connection()}


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
