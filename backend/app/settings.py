from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    API_TITLE: str = "PhenoPixel5.0API"
    API_PREFIX: str = "/api"
    TEST_ENV: str = ""

    ADMIN_PASSWORD_HASH: str | None = None

    EMAIL: str | None = None
    PASSWORD: str | None = None
    HINET_URL: str | None = None
    internet_healthcheck_url: str = "https://1.1.1.1/"
    internet_healthcheck_timeout: float = 5.0
    internet_healthcheck_verify_ssl: bool = True

    login_fail_count_max: int = 5
    login_lock_hour: int = 1
    jwt_secret: str = ""
    refresh_token_exp: int = 60 * 60 * 24 * 30
    access_token_exp: int = 60 * 60
    server_origin: str = "http://localhost:8000"
    admin_handle_id: str = "ikeda042"
    admin_password: str = "default"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
