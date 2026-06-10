import logging

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import get_settings
from app.database import init_db
from app.routes.chat import limiter, router as chat_router
from app.routes.conversations import router as conversations_router
from app.routes.health import router as health_router
from app.routes.upload import router as upload_router

settings = get_settings()

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
logger = structlog.get_logger()


def create_app() -> FastAPI:
    app = FastAPI(title="VectorMind API", version="2.0.0")

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(upload_router)
    app.include_router(conversations_router)

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.on_event("startup")
    async def startup():
        init_db()
        logger.info("app_started", cors=settings.cors_origin_list)

        if settings.qdrant_api_key and "cloud.qdrant.io" in settings.qdrant_url:
            try:
                from app.core.vectorstore import _get_qdrant_client

                _get_qdrant_client().get_collections()
                logger.info("qdrant_startup_ok", url=settings.qdrant_url.split("://")[-1][:40])
            except Exception as e:
                logger.error("qdrant_startup_failed", error=str(e))

    @app.get("/")
    async def root():
        return {
            "message": "VectorMindbot API v2",
            "endpoints": {
                "health": "GET /health",
                "upload": "POST /upload (async, returns task_id)",
                "upload_status": "GET /upload/status/{task_id}",
                "chat": "POST /chat",
                "chat_stream": "POST /chat/stream",
                "conversations": "GET /conversations",
                "metrics": "GET /metrics",
            },
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
