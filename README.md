# Agent Memory Service — self-hosted Add / Search API

![Stars](https://img.shields.io/github/stars/zhaoxinghua09-cell/agent-memory-service?style=flat-square)
![License](https://img.shields.io/github/license/zhaoxinghua09-cell/agent-memory-service?style=flat-square)
![Last commit](https://img.shields.io/github/last-commit/zhaoxinghua09-cell/agent-memory-service/main?style=flat-square)
![Retrieval](https://img.shields.io/badge/retrieval-BM25%20%2B%20dense%20RRF-2ea44f?style=flat-square)
![Fallback](https://img.shields.io/badge/fallback-Ollama%20%E2%86%92%20BM25-orange?style=flat-square)
![Deploy](https://img.shields.io/badge/deploy-Docker%20%2B%20Render%20Blueprint-0068ff?style=flat-square)

A minimal, dependency-light memory service exposing the two endpoints an
evaluation harness calls. Retrieval is hybrid: BM25 over per-message chunks
fused with a dense cosine term by weighted reciprocal-rank fusion, followed by
same-session adjacency expansion and temporal-aware reranking. The dense leg is
the hosted `text-embedding-v4` model; if it is unreachable the service falls
back to a local Ollama server and then to BM25 alone, so it keeps answering
instead of failing.

## Endpoints

| Method | Path      | Purpose |
|--------|-----------|---------|
| `GET`  | `/health` | liveness probe — returns `{"status":"ok"}`; does not touch the database |
| `GET`  | `/version`| read-only build identity: revision, git commit, and the dense leg actually in use |
| `POST` | `/add`    | ingest a session (`messages[]`, optional `user_id`, `session_id`, metadata) |
| `POST` | `/search` | retrieve memories for a query (`query`, optional `user_id`, `top_k`) |

`/health` and `/version` are unauthenticated and read nothing but process state;
they carry no key, path or sample data. The two `POST` routes require the shared
secret. Accepted forms:

```
X-API-Key: <key>
Authorization: Bearer <key>
Authorization: token <key>
api-key: <key>
```

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `AML_API_KEY` | *(unset)* | shared secret; takes precedence when non-empty |
| `AML_MEMORY_KEY` | *(unset)* | fallback secret, typically a platform-generated value |
| `AML_DB_PATH` | `service/data/memory.db` | SQLite file |
| `AML_DASHSCOPE_KEY` | *(unset)* | Model Studio API key; unset ⇒ dense leg skipped |
| `AML_DASHSCOPE_BASE` | `https://dashscope.aliyuncs.com` | Model Studio region endpoint; a key is valid in **one** region only |
| `AML_DASHSCOPE_MODEL` | `text-embedding-v4` | dense embedding model |
| `AML_EMBED_DIM` | `1024` | embedding width; must match `AML_DASHSCOPE_MODEL` |
| `AML_EMBED_TIMEOUT` | `20` | per-embedding-call timeout, seconds |
| `AML_EMBED_COOLDOWN` | `90` | how long a failed dense leg is parked before being retried |
| `AML_EMBED_MODEL` | `bge-m3` | Ollama fallback embedding model |
| `AML_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama fallback endpoint; unreachable ⇒ dense legs skipped |
| `AML_RERANK` | `0` | set to `1` to enable the cross-encoder rerank leg (hosted `qwen3-rerank`); needs `AML_DASHSCOPE_KEY` |
| `AML_RERANK_MODEL` | `qwen3-rerank` | rerank model; FAQ 05 of the challenge places rerankers outside the restricted-model list |
| `AML_RERANK_CANDIDATES` | `100` | how many top fused candidates are sent to the cross-encoder |
| `AML_RERANK_TIMEOUT` | `10` | per-rerank-call timeout, seconds |
| `AML_RERANK_BLEND` / `AML_RERANK_BLEND_W` | `0` / `0.7` | blend the reranked order with the fused order by RRF instead of replacing it |
| `AML_RATE_LIMIT` | `240` | per-minute ceiling applied to **rejected** authentications only; a valid key is never throttled |
| `AML_MAX_BODY_BYTES` | `16777216` | request bodies larger than this are answered `413` unread |
| `AML_RRF_K` / `AML_W_RRF_DENSE` / `AML_W_RRF_LEX` | `60` / `1.0` / `0.5` | weighted-RRF fusion constant and dense/lexical weights |
| `AML_ADJ_SEEDS` / `AML_ADJ_WINDOW` / `AML_ADJ_FACTOR` | `20` / `1` / `0.9` | adjacency expansion: seed count, ±neighbour window, neighbour score factor |
| `AML_W_NOW` / `AML_W_REC` / `AML_W_DATE` | `0.6` / `0.12` / `0.35` | temporal reranking: current-state boost, mild recency nudge, query-year match |
| `AML_RECENCY_HALF_LIFE_DAYS` / `AML_W_STALE` / `AML_STALE_YEARS` | `30` / `0.4` / `1` | current-state reranking: ingest-recency half-life (days), soft penalty for chunks anchored to an old year, that year threshold |
| `AML_W_CHANGE` | `0.5` | additive boost for chunks whose content announces a state change ("just moved", "last month") on current-state queries |
| `AML_COS_REL_GATE` | `0.5` | temporal factors apply only inside the semantic relevance band: raw cosine >= gate * best cosine; with no dense leg temporal reordering is off |
| `AML_CHUNK_WORDS` / `AML_CHUNK_OVERLAP` | `300` / `50` | sliding-window split for long messages |
| `AML_KEEPALIVE_URL` | *(unset)* | optional self-ping; keeps a free-tier instance resident |
| `AML_KEEPALIVE_SECONDS` | `600` | self-ping interval, floored at 60 |

The secret resolves to the first non-empty of `AML_API_KEY`, `AML_MEMORY_KEY`;
a blank or whitespace-only value counts as unset.

## Run locally

```bash
pip install -r service/requirements.txt
cd service && uvicorn app:app --host 127.0.0.1 --port 8000
curl -s localhost:8000/health
```

With no secret configured on loopback, an ephemeral key is generated and printed
once at startup.

## Run in a container

```bash
docker build -t agent-memory-service .
docker run --rm -p 8000:8000 -e AML_API_KEY=your-key agent-memory-service
```

The entrypoint honours a platform-injected `PORT` and falls back to `8000`, so it
drops into any PaaS that assigns a port at runtime. It also passes
`--forwarded-allow-ips=*` to the ASGI server, because behind the platform proxy
the default setting would ignore `X-Forwarded-For` and collapse per-client rate
limiting into a single global bucket.

## Deploy

`render.yaml` declares a complete Blueprint, so no value has to be typed during
setup:

1. Sign in to Render with GitHub and grant it access to this repository.
2. **New + → Blueprint** → pick this repository → **Apply**.
3. Done. The auth secret is declared `generateValue: true`, so the platform
   generates it while provisioning. Copy it from **Dashboard → service →
   Environment → `AML_MEMORY_KEY`** when a client needs it.

Any host that accepts a Dockerfile works equally well — nothing vendor-specific
is involved.

## Keeping a free tier awake

A free instance spins down after 15 minutes without inbound traffic, and the free
filesystem is ephemeral: a spin-down, restart or redeploy wipes the SQLite file,
which is to say the memories themselves. Since `/add` promises that a chunk is
persisted before its 200 response, that is a correctness problem rather than a
latency one.

Two independent layers cover it:

- **`.github/workflows/keepalive.yml`** pings `/health` every 10 minutes. The
  external timer is the layer that matters: an in-process self-ping only runs
  while the instance is already awake, so it can never wake a sleeping one.
- **`AML_KEEPALIVE_URL`** (opt-in, off by default) makes the process ping its own
  public address on an interval.

⚠️ Point keep-alive probes at a real route such as `/health`, never at
`/robots.txt` — while a free instance is asleep the platform answers that path
itself, so a probe there always looks healthy and wakes nothing.

## Design notes

- **Storage** — a single SQLite file in WAL mode; no external database.
- **Retrieval** — hybrid: BM25 over per-message chunks (long messages split into
  ~300-word sliding windows) fused with a dense cosine term by **weighted RRF**
  (`0.8/(K+rank_dense) + 0.8/(K+rank_lex)`, K = 60); the top-20 seeds are then
  expanded with same-session neighbours (±2), and a temporal-aware factor boosts
  recent chunks when the query asks about the present ("now / 最近 / 目前") or
  matches a year named in the query. The top 100 candidates of that fused order
  are then reordered by a **cross-encoder** (hosted `qwen3-rerank`) and the
  response is cut from the reranked order. Queries and documents are encoded
  asymmetrically (`text_type=query` on search, `document` on write).
  The rerank leg is optional (`AML_RERANK=1`) and treats every failure as a
  fallback to the fused order, so it can never break a scored run — which also
  means it can degrade invisibly. `GET /version` therefore reports
  `rerank_leg.healthy` and `rerank_leg.last_error`; **check it from outside
  before submitting a run** (a provider account in arrears answers
  `HTTP 400 Arrearage` and the whole leg silently disappears).
- **Writes are synchronous and idempotent** — a chunk is committed before the 200
  is returned, and a replay of the same `request_id` is accepted without
  duplicating data.
- **Isolation** — every read and write is scoped by `user_id`; a search never
  crosses that boundary.
- **Hardening** — strict auth, per-client rate limiting that honours
  `X-Forwarded-For` behind a trusted proxy, response-shape guards, and a
  request-size ceiling.

`DESIGN.md` describes the method, the changes relative to a naive baseline, and
attribution of the prior work this builds on.

## License

GNU AGPL-3.0 (see `LICENSE`).

## Security

No credentials, tokens or private keys are stored in this repository; runtime
configuration arrives through environment variables only.

There is no built-in default API key. When a platform-injected `PORT` is present
— that is, the process is running as a managed deployment — the service **refuses
to start** unless a secret is configured, so an unauthenticated or default-key
endpoint cannot be exposed by accident.
