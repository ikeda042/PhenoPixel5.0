from fastapi import status


class PhenoPixelException(Exception):
    def __init__(
        self,
        message: str = "Internal Error",
        code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message)
        self.code = code
        self.message = message


class AuthorizationException(PhenoPixelException):
    def __init__(
        self,
        message: str = "Authorization error.",
        code: int = status.HTTP_401_UNAUTHORIZED,
    ):
        super().__init__(message, code)


class InvalidAccessToken(AuthorizationException):
    def __init__(
        self,
        message: str = "Access token is invalid.",
        code: int = status.HTTP_401_UNAUTHORIZED,
    ):
        super().__init__(message, code)


class InvalidRefreshToken(AuthorizationException):
    def __init__(
        self,
        message: str = "Refresh token is invalid.",
        code: int = status.HTTP_401_UNAUTHORIZED,
    ):
        super().__init__(message, code)


class InvalidTokenRequest(AuthorizationException):
    def __init__(
        self,
        message: str = "Token request parameters are invalid.",
        code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, code)


class NoToken(AuthorizationException):
    def __init__(
        self, message: str = "No Token", code: int = status.HTTP_401_UNAUTHORIZED
    ):
        super().__init__(message, code)


class NotEnoughPermissions(AuthorizationException):
    def __init__(
        self,
        message: str = "Not enough permissions",
        code: int = status.HTTP_401_UNAUTHORIZED,
    ):
        super().__init__(message, code)


class InvalidOrigin(AuthorizationException):
    def __init__(
        self, message: str = "Invalid origin.", code: int = status.HTTP_401_UNAUTHORIZED
    ):
        super().__init__(message, code)


class UserNotFound(AuthorizationException):
    def __init__(
        self, message: str = "User not found.", code: int = status.HTTP_404_NOT_FOUND
    ):
        super().__init__(message, code)


class AccountLocked(AuthorizationException):
    def __init__(
        self,
        message: str = "Your account is locked.",
        code: int = status.HTTP_401_UNAUTHORIZED,
    ):
        super().__init__(message, code)


class InvalidPassword(AuthorizationException):
    def __init__(
        self,
        message: str = "Invalid password",
        code: int = status.HTTP_401_UNAUTHORIZED,
    ):
        super().__init__(message, code)
