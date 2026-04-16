"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent.config import settings
from agent.queue.nats_client import NATSClient

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

nats_client = NATSClient(settings.NATS_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    try:
        await nats_client.connect()
        await nats_client.ensure_stream(
            settings.NATS_STREAM_NAME,
            [settings.NATS_TASK_SUBJECT],
        )
        logger.info("NATS connected and stream configured")
    except ConnectionError:
        logger.warning("NATS not available — running without queue")

    yield

    try:
        await nats_client.disconnect()
        logger.info("NATS disconnected")
    except Exception:
        pass


app = FastAPI(
    title="Feature Agent Core",
    description="Backend brain of an autonomous AI coding agent",
    version="1.0.0",
    lifespan=lifespan,
)

web_dir = Path(__file__).parent / "web"
app.mount("/app", StaticFiles(directory=str(web_dir)), name="web")


@app.get("/")
async def serve_index():
    """Serve the web client index page."""
    return FileResponse(str(web_dir / "index.html"))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    nats_status = "connected" if nats_client.is_connected else "disconnected"
    return {
        "status": "ok",
        "nats": nats_status,
        "version": "1.0.0",
    }
