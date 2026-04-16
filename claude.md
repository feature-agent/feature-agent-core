# CLAUDE.md — Feature Agent Core

## Purpose
Feature Agent Core is the backend brain of an
autonomous AI coding agent. It accepts a task
(GitHub issue URL or free text description),
understands the target codebase, implements the
feature, and raises a GitHub PR — with zero human
intervention after the initial task submission.

Built as the backend course project for
"Building Agentic AI Systems: From Zero to
Production" by Adnan Khan.

## Build Prompt
This repository was built using the following
prompt with Claude CLI. Update this section when
requirements change and re-run to iterate.

---
See claude.md for the full build prompt.
---

## Tech Stack
  Language:    Python 3.10+
  API:         FastAPI
  Queue:       NATS JetStream
  LLM:         Claude API via Anthropic SDK
  Model:       claude-opus-4-5
  GitHub:      PyGithub
  Container:   Docker + Docker Compose
  Storage:     Docker named volume
  Testing:     pytest + pytest-asyncio
  Config:      pydantic-settings

## Architecture

### Overview
  UI → POST /tasks → FastAPI → NATS queue
                                    ↓
                              Agent worker
                              (consumes task)
                                    ↓
                              Skills execute
                              in sequence
                                    ↓
                              SSE events stored
                              + emitted to client
                              via GET /stream/{id}

### Why this architecture (v1)
FastAPI sits in front of NATS. The agent worker
consumes tasks from the queue and processes them
one at a time. SSE events flow back through FastAPI
to the client.

This works well for moderate throughput. The known
scaling limitation: one agent worker processes one
task at a time. Concurrent tasks queue up. This is
intentional for the course — v2 architecture where
the agent publishes events directly to NATS output
subjects is discussed at course end.

### Folder Structure
  agent/
    main.py              FastAPI app entry point
    config.py            pydantic-settings config
    llm_client.py        Anthropic SDK wrapper
    benchmark.py         timing and cost tracking
    state_manager.py     task state on Docker volume
    orchestrator.py      skill pipeline coordinator
    skill_base.py        abstract Skill base class
    event_emitter.py     SSE event emission + storage

    storage/
      interface.py       StorageInterface abstract class
      local_volume.py    LocalVolumeStorage (Docker vol)
      efs_stub.py        EFSStorage stub (3-line swap)

    skills/
      issue_reader.py    reads GitHub issue or structures
                         free text into requirement
      clarifier.py       detects ambiguity, generates
                         questions with options
      codebase_explorer.py maps repo, finds relevant files
      code_writer.py     writes code changes via Claude
      test_writer.py     writes pytest tests via Claude
      test_runner.py     runs pytest via subprocess
      pr_creator.py      creates GitHub branch and PR

    queue/
      nats_client.py     NATS JetStream wrapper
      consumer.py        task queue consumer worker

    api/
      routes/
        tasks.py         POST/GET /tasks endpoints
        stream.py        GET /stream/{id} SSE endpoint
        clarify.py       POST /tasks/{id}/clarify
        benchmarks.py    GET /benchmarks endpoints
      models.py          Pydantic API models

    web/
      index.html         served statically at /

  tests/
    conftest.py
    test_llm_client.py
    test_state_manager.py
    test_orchestrator.py
    test_skills.py
    test_api.py

## Determinism Rules
ALL LLM calls in this system use:
  temperature: 0.0
  top_p: 1.0
Every prompt specifies exact JSON output schema.
Every JSON response validated via Pydantic immediately.
Parse failures trigger one retry with correction prompt.
Second failure raises SkillError — never silently ignored.

## Clarification Rules
Maximum 3 questions per task — enforced in code.
questions = questions[:3] — always sliced in clarifier.
Pipeline MUST NOT continue until all answers received.
Task stays AWAITING_CLARIFICATION until POST /clarify.
Clarifier runs ONCE per task — never re-runs on retry.
Each question has 3 suggested options + Other option.
Keyboard shortcuts a/b/c/d select options (web client).

## Benchmark Tracking
Every skill records:
  started_at, ended_at, elapsed_ms
  llm_calls: list with per-call timing and tokens
  input_tokens, output_tokens, cached_tokens
  estimated_cost_usd (calculated from Anthropic rates)
  status: success | failed | retried
Saved to /data/tasks/{id}/benchmark.json after task done.
Appended to /data/benchmarks.jsonl cumulative log.

## Scaling Note
v1 architecture: one agent worker, one task at a time.
Concurrent tasks queue in NATS — processed serially.
v2 solution: multiple agent workers consuming same
subject, events published to per-task NATS subjects,
UI subscribes directly to NATS output subjects.
Covered at end of course — not implemented here.

## For Students Extending This System
To add a new skill:
  1. Create agent/skills/your_skill.py
  2. Inherit from Skill base class
  3. Implement execute() method
  4. Add to orchestrator skill sequence
  5. Add tests in tests/test_skills.py

To swap Docker volume for EFS:
  See agent/storage/efs_stub.py
  Change one line in agent/config.py
  Change one line in agent/state_manager.py

## Phase Build Plan

### Phase 1 — Foundation [ ]
  PR: "feat: project scaffold, Docker Compose,
       NATS, config, and health endpoint"

### Phase 2 — State management and task API [ ]
  PR: "feat: task state manager, POST /tasks,
       GET /tasks, SSE event storage"

### Phase 3 — SSE streaming endpoint [ ]
  PR: "feat: SSE streaming, event replay,
       and keep-alive"

### Phase 4 — LLM client and all 7 skills [ ]
  PR: "feat: LLM client, benchmark tracker,
       and all 7 skill implementations"

### Phase 5 — Orchestrator and clarification [ ]
  PR: "feat: orchestrator, full pipeline,
       clarification flow, POST /clarify"

### Phase 6 — Benchmark endpoints and polish [ ]
  PR: "feat: benchmark endpoints, error hardening,
       scaling notes, final CLAUDE.md"

Mark each phase [DONE] after PR is merged.
