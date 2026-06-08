import re
import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _guest_email(session_id: str) -> str:
    return f"guest-{session_id}@local"


def get_or_create_guest_user(db: Session, session_id: str) -> User:
    email = _guest_email(session_id)
    user = db.query(User).filter(User.email == email).first()
    if user:
        return user

    user = User(
        email=email,
        hashed_password=secrets.token_hex(32),
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        user = db.query(User).filter(User.email == email).first()
        if user:
            return user
        raise


def get_session_user(
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    db: Session = Depends(get_db),
) -> User:
    if not x_session_id or not UUID_PATTERN.match(x_session_id.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Valid X-Session-Id header required",
        )
    return get_or_create_guest_user(db, x_session_id.strip())
