import re
import secrets
import threading
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


_guest_user_locks: dict[str, threading.Lock] = {}
_guest_user_locks_guard = threading.Lock()


def _guest_user_lock(email: str) -> threading.Lock:
    with _guest_user_locks_guard:
        lock = _guest_user_locks.get(email)
        if lock is None:
            lock = threading.Lock()
            _guest_user_locks[email] = lock
        return lock


def get_or_create_guest_user(db: Session, session_id: str) -> User:
    email = _guest_email(session_id)
    with _guest_user_lock(email):
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
            db.expunge(user)
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
