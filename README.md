# Feature Agent Core

The backend brain of an autonomous AI coding agent. Accepts feature requests, explores a target codebase, writes code and tests, and opens GitHub PRs — autonomously.

Built for the "Building Agentic AI Systems" course by Adnan Khan.

## What it does

1. Accepts a task (GitHub issue or description)
2. Detects ambiguity — asks up to 3 clarifying questions with suggested options
3. Clones and explores the target codebase
4. Writes code changes following existing patterns
5. Writes comprehensive tests
6. Runs tests — retries if they fail (max 2x)
7. Opens a GitHub PR with full context
8. Streams every step to the client in real time

## Prerequisites

- Python 3.10+
- Docker Desktop
- Git
- Claude API key (console.anthropic.com)
- GitHub account + personal access token

## Setup

1. Clone this repo
   ```bash
   git clone <repo-url>
   cd feature-agent-core
   ```

2. Create virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Configure environment
   ```bash
   cp .env.example .env
   ```
   Edit .env with your API keys and GitHub repo

4. Start the system
   ```bash
   docker compose up
   ```

5. Open the API docs
   http://localhost:8000/docs

For the web client UI:
See feature-agent-client repo.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Client (UI)                           │
│                                                              │
│   POST /api/tasks ──────┐    GET /api/stream/{id} ◄─── SSE  │
│   POST /api/tasks/{id}/ │                                    │
│         clarify ────┐   │                                    │
└─────────────────────┼───┼────────────────────────────────────┘
                      │   │
                      ▼   ▼
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI (agent/main.py)                   │
│                                                              │
│   /api/tasks    /api/stream/{id}    /api/tasks/{id}/clarify  │
│   /api/health   /api/benchmarks     /api/tasks/{id}/benchmark│
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   NATS JetStream Queue                        │
│                 subject: agent.tasks.incoming                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Orchestrator (agent worker)                 │
│                                                              │
│   ┌─────────────┐   ┌───────────┐   ┌──────────────────┐    │
│   │ IssueReader  │──▶│ Clarifier │──▶│ CodebaseExplorer │    │
│   │  (skill 1)   │   │ (skill 2) │   │    (skill 3)     │    │
│   └─────────────┘   └─────┬─────┘   └────────┬─────────┘    │
│                            │                   │              │
│                  ┌─────────┘                   ▼              │
│                  │ PAUSE if              ┌───────────┐        │
│                  │ unclear               │CodeWriter │        │
│                  │                       │ (skill 4) │        │
│                  ▼                       └─────┬─────┘        │
│         ┌────────────────┐                     │              │
│         │  AWAITING_     │               ┌─────▼─────┐        │
│         │ CLARIFICATION  │               │TestWriter │        │
│         │  (wait for     │               │ (skill 5) │        │
│         │   POST /clarify│               └─────┬─────┘        │
│         │   to resume)   │                     │              │
│         └────────────────┘               ┌─────▼─────┐        │
│                                          │TestRunner │        │
│                                          │ (skill 6) │        │
│                                          └─────┬─────┘        │
│                                                │              │
│                                    ┌───── PASS?──────┐        │
│                                    │ yes          no │        │
│                                    ▼     (retry 2x)  │        │
│                              ┌───────────┐    ▲      │        │
│                              │ PRCreator │    │      │        │
│                              │ (skill 7) │    └──────┘        │
│                              └─────┬─────┘                    │
│                                    │                          │
│                                    ▼                          │
│                              GitHub PR opened                 │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                Docker Volume (/data)                          │
│                                                              │
│   /data/tasks/{id}/state.json      Task state                │
│   /data/tasks/{id}/events.jsonl    SSE event log             │
│   /data/tasks/{id}/benchmark.json  Timing & cost data        │
│   /data/benchmarks.jsonl           Cumulative benchmark log  │
└──────────────────────────────────────────────────────────────┘
```

See [CLAUDE.md](CLAUDE.md) for full architecture documentation.

## Scaling Limitations

This system processes one task at a time. The agent worker is a single consumer on the NATS queue. Concurrent tasks are queued and processed serially.

For higher throughput see the v2 architecture discussion at the end of the course:
- Multiple agent worker containers
- Per-task NATS output subjects
- UI subscribes directly to NATS WebSocket

## Running Tests

```bash
pytest tests/ -v --cov=agent
```
