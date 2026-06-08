import logging
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, List

from langchain_core.documents import Document

from app.core.pdf_parser import extract_pdf_pages
from app.core.rag import chunk_documents, index_documents

logger = logging.getLogger(__name__)

# In-memory upload task store
upload_tasks: Dict[str, dict] = {}


def create_upload_task(user_id: int, file_count: int) -> str:
    task_id = str(uuid.uuid4())
    upload_tasks[task_id] = {
        "task_id": task_id,
        "user_id": user_id,
        "status": "pending",
        "file_count": file_count,
        "chunks_created": 0,
        "text_length": 0,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    return task_id


def get_upload_task(task_id: str) -> dict | None:
    return upload_tasks.get(task_id)


def process_upload_task(task_id: str, user_id: int, files: List[tuple]):
    """Process uploaded PDFs in background. files = [(bytes, filename), ...]"""
    task = upload_tasks.get(task_id)
    if not task:
        return

    task["status"] = "processing"
    try:
        all_documents: List[Document] = []
        total_text = 0

        for content, filename in files:
            pdf_file = BytesIO(content)
            pdf_file.name = filename
            pages = extract_pdf_pages(pdf_file, filename)
            total_text += sum(len(p[0]) for p in pages)
            docs = chunk_documents(pages, filename)
            all_documents.extend(docs)

        if not all_documents:
            raise ValueError("No text could be extracted from the PDFs")

        chunks_created = index_documents(user_id, all_documents)

        task["status"] = "completed"
        task["chunks_created"] = chunks_created
        task["text_length"] = total_text
        task["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Upload task %s completed: %d chunks", task_id, chunks_created)

    except Exception as e:
        logger.error("Upload task %s failed: %s", task_id, e)
        task["status"] = "failed"
        task["error"] = str(e)
        task["completed_at"] = datetime.now(timezone.utc).isoformat()
