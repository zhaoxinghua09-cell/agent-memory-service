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

That baseline has three problems. It has no lexical ranking, so a rare literal
token carries no weight; it makes the success of every query depend on a single
outbound call, so one bad minute at a provider degrades or fails a whole run; and
it exposes an unauthenticated endpoint. The dense leg is still used here — the
open-method division pins the model — but it is fused with lexical scoring and is
treated as a leg that can fail rather than as the only way to answer.

## 3. What this implementation changes

| Area | Baseline | Here | Why |
|---|---|---|---|
| Retrieval | substring match; hosted embedding API | BM25 over tokenised chunks fused with a dense cosine term (`0.65·dense + 0.35·lexical`) whenever the embedding leg answers | Keeps lexical precision on rare terms, where dense vectors are weak, while the dense term carries paraphrase and cross-lingual recall |
| Embeddings | remote API with no fallback | hosted `text-embedding-v4` (Model Studio / DashScope), with Ollama and then BM25-only degradation | The open-method division pins the embedding model to this exact name; the two fallback legs mean an API fault degrades retrieval instead of failing the request |
| Version identity | none | `GET /version` echoes the build revision, the git commit and the active dense leg | The evaluation rules treat interface configuration and system version as reviewable artefacts |
| Write semantics | write then acknowledge, best effort | the chunk is committed before the 200; a `request_id` replay is idempotent | The contract requires a chunk to be persisted and searchable before its 200 |
| Isolation | none | every statement is scoped by `user_id` | Cross-user retrieval is prohibited |
| Auth | none | shared secret, four accepted header forms | An open endpoint would expose every sample |
| Rate limiting | none | sliding window on **rejected** authentications only | Brute-force protection without a counter in front of the scored run |
| Blocking work | n/a | embedding calls run in a worker thread | The handler is `async`; calling blocking HTTP inline would serialise concurrent requests |

The dense leg encodes queries and documents asymmetrically — `text_type` is set to
`query` on the search side and `document` on the write side — which is the
intended operating mode of this model family for short-query/long-document
matching. It is carried on the native endpoint; the OpenAI-compatible route is
kept as a second attempt because it accepts the same key but ignores `text_type`.

BM25 is implemented directly (`bm25_scores`, `k1 = 1.2`, `b = 0.75`) rather than
pulled in as a dependency, so the ranking function is auditable in one screen and
the image keeps three runtime dependencies in total.

## 3b. v0.5.0 retrieval upgrades (zero-LLM, hot path)

Cycle 1 of this challenge published its leaderboard in August 2026: the top
open-source entry (InvMem, 45.06) ran a four-stage pipeline — per-message
chunking with sliding windows over long messages, dual dense/BM25 recall fused
by **weighted reciprocal-rank fusion**, and **same-session adjacency
expansion** around the top seeds. Analysis of the cycle-1 boards also showed
temporal reasoning was the weakest dimension across all entries. v0.5.0
adopts those findings; every technique below is deterministic and adds no LLM
call to the request path.

| Area | v0.4.0 | v0.5.0 | Why |
|---|---|---|---|
| Chunking | ≤20 messages or ≤2000 words merged into one chunk | one chunk per message; long messages split into ~300-word windows with 50-word overlap | coarse chunks dilute topics and cut conditions away from their conclusions; message boundaries also give adjacency natural neighbours (InvMem) |
| Fusion | linear mix `0.65·dense + 0.35·lexical` | weighted RRF: `1.0/(K+rank_dense) + 0.5/(K+rank_lex)`, K=60 | rank-level fusion is scale-free; mixing raw similarity and BM25 magnitudes is brittle (InvMem; standard RRF literature) |
| Evidence set | top-k of the fused list only | top-20 seeds expanded with same-session neighbours (±1 ordinal) re-entering at seed×0.9 | pulls rule premises, matching questions and referents back into the evidence list — the mechanism behind the cycle-1 lead on rule-following and multi-hop |
| Temporal ranking | none | event dates parsed from content at add time (`event_time` column); queries containing "now/currently/现在/目前" style signals boost the newest chunks (`W_NOW·recency`), a mild always-on recency nudge (`W_REC`), and a query-year match boost (`W_DATE`) | cycle-1 analysis: temporal reasoning is the field's weakest dimension; mem0 v3 showed retrieval-time recency beats write-time overwriting; this is graphiti's soft-invalidation intuition without the LLM |
| Dedup | none | exact sha256(content per user) suppressed at add time | MemOS stage-1; duplicates pollute the evidence list |

All weights are environment-tunable (`AML_W_*`, `AML_ADJ_*`, `AML_RRF_*`) with
the defaults above; `GET /version` reports the active retrieval description.

Deliberately **not** done in v0.5.0, recorded as candidates for a later cycle:
LLM-based fact extraction or query rewriting (hot-path latency and cost on
/add with up to 500 messages), a knowledge graph (operational weight), and
cosine near-duplicate suppression (O(n) vector parsing per add). v0.6.0 spent
its budget on the rerank leg instead (§3c): the measurements pointed at an
ordering problem rather than a recall problem, which made a cross-encoder the
cheaper win than any of the three above.

## 3c. v0.6.0 cross-encoder rerank leg (retrieve-then-rerank)

The fused retriever's remaining failures were predominantly **ordering**
failures, not recall failures: on single-hop questions the answer chunk was
usually already inside the retrieved pool but ranked below the cut. That is the
signature of a bi-encoder setup — query and document are encoded independently,
so fine-grained relevance has to be inferred from two vectors — and it is
exactly what a cross-encoder is for (Nogueira & Cho, arXiv:1901.04085). The
challenge's FAQ 05 places rerankers outside its restricted-model list, so a
hosted reranking model is admissible.

| Area | v0.5.0 | v0.6.0 | Why |
|---|---|---|---|
| Ranking | one-stage: fused order is the answer order | two-stage: top 100 of the fused order reordered by a cross-encoder (`qwen3-rerank`), response cut from the reranked order, candidates beyond the pool keeping their fused order | the pool already contains the answer; a cross-encoder separates it from near-miss distractors |
| Fusion weights | `1.0/(K+rank_dense) + 0.5/(K+rank_lex)`, K=60 | `0.8/(K+rank_dense) + 0.8/(K+rank_lex)`, K=60 | with the rerank leg in place the dense leg had been over-weighted; measured over 9 configurations, 2:1 → 1:1 was the largest single gain |
| Adjacency window | ±1 ordinal around top-20 seeds | ±2 | mild but consistent; the effect beyond ±2 flattens |
| Diagnosis surface | `dense_leg` only | `dense_leg` + `rerank_leg` on `GET /version` | the leg fails silently by design (see §4), so it has to be visible from outside |

Measured on the local bench (challenge dataset, 10 conversations, 1382
questions, recall@20), same database and same evaluation harness throughout:

| Category | n | v0.5.0 | v0.6.0 | Δ |
|---|---|---|---|---|
| overall | 1382 | 0.6819 | **0.7770** | **+9.51 pt** |
| single-hop | 213 | 0.3465 | 0.4566 | +11.01 pt |
| temporal | 299 | 0.7246 | 0.8021 | +7.75 pt |
| multi-hop | 66 | 0.4066 | 0.4493 | +4.27 pt |
| open-domain | 802 | 0.7776 | 0.8797 | +10.21 pt |

Attribution of the total: the rerank leg alone (fused weights unchanged at 2:1)
accounted for 0.7603, and the retuned weights and adjacency window supplied the
remaining **+1.67 pt** to 0.7770.

That last figure is stated with its uncertainty rather than as a settled fact:
+1.67 pt on 1382 questions is ≈23 questions, one standard deviation is ≈1.15 pt,
so the effect is ≈1.4σ — **suggestive, not conclusive on its own**. It was
reproduced independently on a 417-question subset (+1.96 pt at a 1.15σ scale of
its own) under the same harness, and all four categories were non-regressing at
full scale (single-hop +0.69 pt, i.e. no sign of the subset's apparent
single-hop loss). The change is adopted on that combined evidence, not on a
single run.

Cost and latency: the leg is one extra hosted round trip per `/search`. Full
1382-question runs on this machine took 1468 s (v0.6.0) and 1978 s (rerank leg
without the retuned weights) against 1426 s for the v0.5.0 configuration, so the
added cost is small next to the embedding round trip, while the spread between
the two reranked runs (≈35%) is provider-side latency variance rather than a
property of the configuration.

An **RRF blend** of the reranked rank with the fused rank exists
(`AML_RERANK_BLEND`), written for the failure mode where a confident fused
match is pushed out of the top-20 by the cross-encoder. Measured +0.13 pt on a
417-question subset — inside noise — so it is **off by default** and kept only
as a documented switch.

## 4. Failure behaviour

- A failing embedding leg is **parked, not latched off**. The first failure starts
  a 90-second cooldown, after which the leg is retried. Latching it off would mean
  that a single transient fault at start-up silently degrades an entire evaluation
  run to lexical-only, with nothing visible from the outside; the cooldown keeps
  recovery automatic and bounded.
- The rerank leg fails **open**: a missing key, a timeout, a non-2xx answer or an
  empty result list all leave the fused order untouched, so the service cannot
  fail because of it. The cost of that choice is invisibility, and it is not
  hypothetical: on 2026-09-25 the provider account went into arrears
  (`HTTP 400 Arrearage`), a 30-minute evaluation run silently measured the
  no-rerank path, and the result read like "reranking brings nothing". The last
  call's outcome is therefore published on `GET /version` as
  `rerank_leg.healthy` / `rerank_leg.last_error` (e.g. `HTTP 400 Arrearage`),
  and checking it from outside is a pre-submission step, not a nicety.
- The concurrency limiter deliberately does **not** count authenticated traffic. It
  is a brute-force guard: only rejected authentications advance the window, and the
  key that passes is never throttled. The evaluator is the only credentialled client,
  so counting its requests could only ever fire at the actor the service exists to
  serve — and a 429 here answers `Retry-After: 60`, spending a minute of evaluation
  wall-clock per false trigger against a budget of two Full runs per track, the
  second of which is locked for 30 days. Protecting the scored run was worth more
  than a request ceiling that no untrusted party can reach anyway.
- Embedding calls are dispatched to a worker thread. They are blocking HTTP with a
  20-second timeout, and the handlers are `async def`, so calling them inline would
  have turned N concurrent requests into N sequential ones. Measured against a stub
  that sleeps 1 s per call, 16 concurrent `/add` requests with the work offloaded
  finish in **1.42 s total**; inline they would have taken ≈16 s.
- A body larger than `AML_MAX_BODY_BYTES` (default 16 MiB, declared via
  `Content-Length`) is answered with `413` before it is read into memory. The
  container has a hard memory ceiling, and being OOM-killed mid-run costs a scored
  run, whereas `413` is an ordinary contract answer.
- A missing or invalid embedding key is not fatal: the service still starts and
  serves lexical-only. That is a silent quality loss, so the leg actually in use
  is published on `GET /version` and is meant to be checked from outside before a
  run is submitted.
- In a managed deployment (a platform-injected `PORT`), the process refuses to
  start when no secret is configured.
- Malformed request bodies are rejected before touching the database.

## 5. Known limits

Stated plainly rather than left for a reader to discover:

- SQLite access remains synchronous inside `async def` handlers. Each statement is
  sub-millisecond and is serialised by a process-level lock, so it has not been
  moved off the event loop; the blocking network call, which is the slow one, has.
- The advertised concurrency is kept deliberately conservative because the service
  runs a single uvicorn worker: past a point, requests queue rather than scale. Load
  tests against the currently deployed build gave 64 concurrent searches completing
  (p50 12.2 s) and 256 failing in bulk; that build still routed embeddings through
  the event loop, so the offload removes one bottleneck but has not been re-measured
  end to end, and the conservative declaration stands until it is.
- On a free platform tier the filesystem is ephemeral. The keep-alive described
  in `README.md` mitigates that constraint; it does not remove it.
- The rerank leg adds a **paid, external dependency to the hot path**, and the
  published score depends on it. If the key loses quota or the account is in
  arrears the service stays up and answerable while scoring like v0.5.0. This is
  a deliberate trade — never fail a scored run — but it means the deployed
  quality is a function of an account's billing state, and the only defence is
  the `GET /version` check above. No figure in this document was measured with
  that leg degraded.

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
- **Qwen3-Reranker** — Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang,
  P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, J. Zhou, *Qwen3 Embedding:
  Advancing Text Embedding and Reranking Through Foundation Models*,
  arXiv:2506.05176, 2025 (Apache-2.0). Called as a hosted reranking endpoint of
  Alibaba Cloud Model Studio; used as published, neither modified nor
  redistributed. In the deployed configuration this is the rerank leg.
- **Retrieve-then-rerank** — R. Nogueira and K. Cho, *Passage Re-ranking with
  BERT*, arXiv:1901.04085, 2019. The two-stage arrangement used here (cheap
  recall, then expensive cross-encoder over the pool) follows this pattern; the
  implementation is ours.
- **Ollama** — local inference server used to host the fallback embedding model.
- **FastAPI** / **Starlette** / **Uvicorn** — web framework and ASGI server.
- **httpx** — HTTP client used for embedding calls.
- **SQLite** — storage engine (WAL mode); the schema and statements are ours.

No prior work is redistributed in modified form. Model weights are not included
in this repository, and every dependency arrives from its own upstream package.