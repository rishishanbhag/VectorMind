import time

from fastapi import APIRouter

router = APIRouter(tags=["health"])

_start_time = time.time()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 2),
    }
