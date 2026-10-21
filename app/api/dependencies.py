import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from app.core.config import get_settings
from app.db import session_factory
from app.models import User

bearer = HTTPBearer(auto_error=False)


def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    expected = get_settings().api_token.get_secret_value()
    if not credentials or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(401, "Invalid API token", headers={"WWW-Authenticate": "Bearer"})
    # Explicit single-principal mock adapter. Never trust a user_id from the request body.
    return "mock-employee"


def get_session():
    with session_factory()() as session, session.begin():
        yield session


def current_user(external_id=Depends(authenticate), session=Depends(get_session)):
    user = session.scalar(select(User).where(User.external_id == external_id))
    if not user:
        raise HTTPException(503, "Bootstrap has not completed")
    return user.id
