import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.session import get_session_user
from app.database import get_db
from app.models import Conversation, Message, User

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationOut(BaseModel):
    id: int
    title: str

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[list] = None

    model_config = {"from_attributes": True}


class ConversationCreate(BaseModel):
    title: str = "New conversation"


@router.get("", response_model=List[ConversationOut])
def list_conversations(
    current_user: User = Depends(get_session_user),
    db: Session = Depends(get_db),
):
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_session_user),
    db: Session = Depends(get_db),
):
    conv = Conversation(user_id=current_user.id, title=payload.title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("/{conversation_id}/messages", response_model=List[MessageOut])
def get_messages(
    conversation_id: int,
    current_user: User = Depends(get_session_user),
    db: Session = Depends(get_db),
):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    result = []
    for msg in messages:
        sources = json.loads(msg.sources_json) if msg.sources_json else None
        result.append(MessageOut(id=msg.id, role=msg.role, content=msg.content, sources=sources))
    return result


def save_message(
    db: Session,
    user_id: int,
    conversation_id: int,
    role: str,
    content: str,
    sources: Optional[list] = None,
) -> Message:
    msg = Message(
        user_id=user_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources_json=json.dumps(sources) if sources else None,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_or_create_conversation(db: Session, user_id: int, conversation_id: Optional[int]) -> Conversation:
    if conversation_id:
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .first()
        )
        if conv:
            return conv

    conv = Conversation(user_id=user_id, title="New conversation")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv
