"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.config import settings
from agent.event_emitter import EventEmitter
from agent.orchestrator import Orchestrator
from agent.queue.consumer import TaskConsumer
from agent.queue.nats_client import NATSClient
from agent.skills.issue_reader import IssueReaderSkill
from agent.skills.clarifier import ClarifierSkill
from agent.skills.codebase_explorer import CodebaseExplorerSkill
from agent.skills.code_writer import CodeWriterSkill
from agent.skills.test_writer import TestWriterSkill
from agent.skills.test_runner import TestRunnerSkill
from agent.skills.pr_creator import PRCreatorSkill
from agent.state_manager import StateManager
from agent.storage.local_volume import LocalVolumeStorage

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

nats_client = NATSClient(settings.NATS_URL)
storage = LocalVolumeStorage(settings.DATA_PATH)
state_manager = StateManager(storage)
event_emitter = EventEmitter(state_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    consumer = None
    try:
        await nats_client.connect()
        await nats_client.ensure_stream(
            settings.NATS_STREAM_NAME,
            [settings.NATS_TASK_SUBJECT],
        )
        logger.info("NATS connected and stream configured")

        # Build skill pipeline and orchestrator
        skills = [
            IssueReaderSkill(),
            ClarifierSkill(),
            CodebaseExplorerSkill(),
            CodeWriterSkill(),
            TestWriterSkill(),
            TestRunnerSkill(),
            PRCreatorSkill(),
        ]
        orchestrator = Orchestrator(skills, state_manager, event_emitter, storage)

        # Start consumer
        consumer = TaskConsumer(
            nats_client, state_manager, orchestrator, settings.NATS_TASK_SUBJECT
        )
        await consumer.start()
        logger.info("Task consumer started — agent is ready")

    except ConnectionError:
        logger.warning("NATS not available — running without queue (tasks will not be processed)")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a safe 500 response."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Register API routers
from agent.api.routes.tasks import router as tasks_router  # noqa: E402
from agent.api.routes.stream import router as stream_router  # noqa: E402
from agent.api.routes.clarify import router as clarify_router  # noqa: E402
from agent.api.routes.benchmarks import router as benchmarks_router  # noqa: E402

app.include_router(tasks_router)
app.include_router(stream_router)
app.include_router(clarify_router)
app.include_router(benchmarks_router)

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
