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
plus two read-only endpoints: `GET /health`, a liveness probe that deliberately
does not read the database so that a cheap probe cannot contend with evaluation
traffic, and `GET /version`, which reports the live revision and the retrieval
leg actually backing `/search`. The latter exists because a system whose
retrieval path cannot be identified from the outside cannot be version-reviewed.

Storage and lexical ranking are entirely in-process. The only outbound call made
at request time is the embedding call described in §3; when that leg is
unavailable the service degrades to lexical-only rather than failing.

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
| Retrieval | substring match; hosted embedding API | BM25 over tokenised chunks fused with a dense cosine term (`0.65·dense + 0.35·lexical`) whenever the embedding leg answers | Keeps lexical precision on rare terms, where dense vectors are weak, while the dense term carries paraphrase and cross-lingual recall |
| Embeddings | remote API with no fallback | hosted `text-embedding-v4` (Model Studio / DashScope), with Ollama and then BM25-only degradation | The open-method division pins the embedding model to this exact name; the two fallback legs mean an API fault degrades retrieval instead of failing the request |
| Version identity | none | `GET /version` echoes the build revision, the git commit and the active dense leg | The evaluation rules treat interface configuration and system version as reviewable artefacts |
| Write semantics | write then acknowledge, best effort | the chunk is committed before the 200; a `request_id` replay is idempotent | The contract requires a chunk to be persisted and searchable before its 200 |
| Isolation | none | every statement is scoped by `user_id` | Cross-user retrieval is prohibited |
| Auth | none | shared secret, four accepted header forms | An open endpoint would expose every sample |
| Rate limiting | none | per-client sliding window, `X-Forwarded-For`-aware | Survives a reverse proxy without collapsing into one global bucket |

The dense leg encodes queries and documents asymmetrically — `text_type` is set to
`query` on the search side and `document` on the write side — which is the
intended operating mode of this model family for short-query/long-document
matching. It is carried on the native endpoint; the OpenAI-compatible route is
kept as a second attempt because it accepts the same key but ignores `text_type`.

BM25 is implemented directly (`bm25_scores`, `k1 = 1.2`, `b = 0.75`) rather than
pulled in as a dependency, so the ranking function is auditable in one screen and
the image keeps three runtime dependencies in total.

## 4. Failure behaviour

- A failing embedding leg is **parked, not latched off**. The first failure starts
  a 90-second cooldown, after which the leg is retried. Latching it off would mean
  that a single transient fault at start-up silently degrades an entire evaluation
  run to lexical-only, with nothing visible from the outside; the cooldown keeps
  recovery automatic and bounded.
- A missing or invalid embedding key is not fatal: the service still starts and
  serves lexical-only. That is a silent quality loss, so the leg actually in use
  is published on `GET /version` and is meant to be checked from outside before a
  run is submitted.
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
- **`text-embedding-v4`** — the hosted embedding model served by Alibaba Cloud
  Model Studio (DashScope), called over its public HTTP API with a per-region key.
  Used as published through the vendor endpoint; the model is neither modified nor
  redistributed. In the deployed configuration this is the dense leg.
- **BGE-M3** — J. Chen et al., *BGE M3-Embedding: Multi-Lingual,
  Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge
  Distillation*, 2024. Retained as the self-hosted fallback leg so the service can
  still run fully offline; it is not the leg used in the deployed configuration.
  Used as published; the model is neither modified nor redistributed.
- **Ollama** — local inference server used to host the fallback embedding model.
- **FastAPI** / **Starlette** / **Uvicorn** — web framework and ASGI server.
- **httpx** — HTTP client used for embedding calls.
- **SQLite** — storage engine (WAL mode); the schema and statements are ours.

No prior work is redistributed in modified form. Model weights are not included
in this repository, and every dependency arrives from its own upstream package.