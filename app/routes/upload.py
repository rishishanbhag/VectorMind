import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.session import get_session_user
from app.config import get_settings
from app.core.tasks import create_upload_task, get_upload_task, process_upload_task
from app.models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])
limiter = Limiter(key_func=get_remote_address)
settings = get_settings()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("5/minute")
async def upload_pdfs(
    request: Request,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_session_user),
):
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded")

    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Maximum {settings.max_files_per_upload} files per upload",
        )

    file_data = []
    total_size = 0

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.filename}. Only PDF files are allowed.",
            )

        content = await file.read()
        file_size = len(content)

        if file_size > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {file.filename} exceeds {settings.max_file_size_mb}MB limit",
            )

        total_size += file_size
        if total_size > settings.max_total_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Total upload exceeds {settings.max_total_upload_mb}MB limit",
            )

        file_data.append((content, file.filename))

    task_id = create_upload_task(current_user.id, len(files))
    # Process synchronously — in-memory tasks are lost when Render restarts (OOM/redeploy),
    # which caused 404 "Task not found" on status polling.
    process_upload_task(task_id, current_user.id, file_data)
    task = get_upload_task(task_id)

    return {
        **task,
        "message": "Processing complete" if task["status"] == "completed" else "Processing failed",
    }


@router.get("/status/{task_id}")
async def upload_status(
    task_id: str,
    current_user: User = Depends(get_session_user),
):
    task = get_upload_task(task_id)
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    if task["user_id"] != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return task
