import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.session import get_session_user
from app.config import get_settings
from app.core.rag import ask_question, ask_question_stream
from app.core.vectorstore import get_vectorstore
from app.database import get_db
from app.models import User
from app.routes.conversations import get_or_create_conversation, save_message

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])
limiter = Limiter(key_func=get_remote_address)
settings = get_settings()


class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[int] = None
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []
    conversation_id: int
    usage: dict = {}


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat_endpoint(
    request: Request,
    payload: ChatRequest,
    current_user: User = Depends(get_session_user),
    db: Session = Depends(get_db),
):
    if not payload.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty")

    store = get_vectorstore(current_user.id)
    if store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents indexed. Please upload PDFs first.",
        )

    if payload.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use POST /chat/stream for streaming responses",
        )

    try:
        conv = get_or_create_conversation(db, current_user.id, payload.conversation_id)
        save_message(db, current_user.id, conv.id, "user", payload.question)

        result = ask_question(current_user.id, payload.question, str(conv.id))
        save_message(db, current_user.id, conv.id, "assistant", result["answer"], result.get("sources"))

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            conversation_id=conv.id,
            usage=result.get("usage", {}),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error("Chat error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream_endpoint(
    request: Request,
    payload: ChatRequest,
    current_user: User = Depends(get_session_user),
    db: Session = Depends(get_db),
):
    if not payload.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty")

    store = get_vectorstore(current_user.id)
    if store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents indexed. Please upload PDFs first.",
        )

    conv = get_or_create_conversation(db, current_user.id, payload.conversation_id)
    save_message(db, current_user.id, conv.id, "user", payload.question)

    return StreamingResponse(
        ask_question_stream(current_user.id, payload.question),
        media_type="text/event-stream",
    )


@router.get("/status")
async def get_status(current_user: User = Depends(get_session_user)):
    store = get_vectorstore(current_user.id)
    return {
        "vectorstore_loaded": store.count() > 0,
        "document_count": store.count(),
        "anthropic_api_configured": bool(settings.anthropic_api_key),
        "user_id": current_user.id,
    }
