from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_TITLE: str = "PhenoPixel5.0API"
    API_PREFIX: str = "/api"
    TEST_ENV: str = ""

    ADMIN_PASSWORD_HASH: str | None = None

    EMAIL: str | None = None
    PASSWORD: str | None = None
    HINET_URL: str | None = None

    DROPBOX_REFRESH_TOKEN: str | None = None
    DROPBOX_APP_KEY: str | None = None
    DROPBOX_APP_SECRET: str | None = None
    login_fail_count_max: int = 5
    login_lock_hour: int = 1
    jwt_secret: str = ""
    refresh_token_exp: int = 60 * 60 * 24 * 30
    access_token_exp: int = 60 * 60

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
