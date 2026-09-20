---
title: DESIGN
type: note
permalink: workbuddy/2026-09-20-19-28-08/aml-handoff/design
---

# Design notes, changes relative to a baseline, and attribution

This accompanies the source in `service/app.py`. It states what the baseline
approach looks like, what this implementation changes and why, and which prior
work it builds on.

## 1. Scope

The two endpoints the evaluation harness calls — `POST /add` and `POST /search` —
plus a `GET /health` liveness probe that deliberately does not read the database,
so a cheap probe cannot contend with evaluation traffic.

Everything runs self-hosted in a single container with no outbound network
dependency at request time.

## 2. Baseline

The straightforward way to build a service of this shape is:

- store each incoming session as one opaque blob keyed by session;
- retrieve with substring or `LIKE` matching;
- call a hosted embedding API for semantic search;
- leave authentication and rate limiting to the deployment layer.

That baseline has three problems in an offline judging sandbox: it has no lexical
ranking, it requires network egress for every query, and it exposes an
unauthenticated endpoint.

## 3. What this implementation changes

| Area | Baseline | Here | Why |
|---|---|---|---|
| Retrieval | substring match; hosted embedding API | BM25 over tokenised chunks, fused with local embedding similarity when available | Works with no network, and keeps lexical precision on rare terms where dense vectors are weak |
| Embeddings | remote API | local Ollama (`bge-m3`), with BM25-only degradation | No egress; degrades instead of failing |
| Write semantics | write then acknowledge, best effort | the chunk is committed before the 200; a `request_id` replay is idempotent | The contract requires a chunk to be persisted and searchable before its 200 |
| Isolation | none | every statement is scoped by `user_id` | Cross-user retrieval is prohibited |
| Auth | none | shared secret, four accepted header forms | An open endpoint would expose every sample |
| Rate limiting | none | per-client sliding window, `X-Forwarded-For`-aware | Survives a reverse proxy without collapsing into one global bucket |

BM25 is implemented directly (`bm25_scores`, `k1 = 1.2`, `b = 0.75`) rather than
pulled in as a dependency, so the ranking function is auditable in one screen and
the image keeps three runtime dependencies in total.

## 4. Failure behaviour

- The embedding endpoint is probed once; if it does not answer, the flag latches
  and retrieval proceeds on BM25 alone rather than retrying on every request.
- In a managed deployment (a platform-injected `PORT`), the process refuses to
  start when no secret is configured.
- Malformed request bodies are rejected before touching the database.

## 5. Known limits

Stated plainly rather than left for a reader to discover:

- `/add` and `/search` are `async def` handlers that call synchronous SQLite and
  synchronous HTTP. Under sustained concurrency this blocks the event loop and
  degrades into 502/503 responses. Measured on a 16-thread machine: 64 concurrent
  searches complete (p50 12.2 s), while 256 concurrent searches fail in bulk. The
  advertised concurrency is therefore kept deliberately conservative. Moving the
  blocking sections into a thread pool is the obvious next step and has not been
  done.
- On a free platform tier the filesystem is ephemeral. The keep-alive described
  in `README.md` mitigates that constraint; it does not remove it.

## 6. Attribution — prior work this builds on

- **BM25** — S. Robertson and H. Zaragoza, *The Probabilistic Relevance
  Framework: BM25 and Beyond*, Foundations and Trends in Information Retrieval,
  3(4), 2009. The scoring function here follows the standard form with the
  conventional defaults `k1 = 1.2`, `b = 0.75`.
- **BGE-M3** — J. Chen et al., *BGE M3-Embedding: Multi-Lingual,
  Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge
  Distillation*, 2024. Used as published for the optional dense leg; the model is
  neither modified nor redistributed.
- **Ollama** — local inference server used to host the embedding model.
- **FastAPI** / **Starlette** / **Uvicorn** — web framework and ASGI server.
- **httpx** — HTTP client used for embedding calls.
- **SQLite** — storage engine (WAL mode); the schema and statements are ours.

No prior work is redistributed in modified form. Model weights are not included
in this repository, and every dependency arrives from its own upstream package.