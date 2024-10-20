from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import bcrypt
from dotenv import load_dotenv
import os

load_dotenv()


class AuthCrud:
    security = HTTPBasic()

    password_hash_hard_coded = os.getenv("PASSWORD_HASH_SECRET")

    @classmethod
    def hash_password(cls, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    @classmethod
    def verify_password(cls, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )

    @classmethod
    def get_account(cls, credentials: HTTPBasicCredentials = Depends(security)) -> str:
        correct_username = "admin"
        correct_password_hash = cls.password_hash_hard_coded
        if correct_password_hash is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password hash not found",
            )
        if credentials.username != correct_username or not cls.verify_password(
            credentials.password, correct_password_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username
