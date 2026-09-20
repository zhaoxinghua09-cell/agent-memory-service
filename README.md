---
title: README
type: note
permalink: workbuddy/2026-09-20-19-28-08/aml-handoff/readme-1
---

# Agent Memory Service — self-hosted Add / Search API

A minimal, dependency-light memory service exposing the two endpoints an
evaluation harness calls. It runs fully offline: retrieval uses local embeddings
when an Ollama server is reachable and falls back to BM25 otherwise, so it still
answers inside a no-network judging sandbox.

## Endpoints

| Method | Path      | Purpose |
|--------|-----------|---------|
| `GET`  | `/health` | liveness probe — returns `{"status":"ok"}`; does not touch the database |
| `POST` | `/add`    | ingest a session (`messages[]`, optional `user_id`, `session_id`, metadata) |
| `POST` | `/search` | retrieve memories for a query (`query`, optional `user_id`, `top_k`) |

Both `POST` routes require the shared secret. Accepted forms:

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
| `AML_EMBED_MODEL` | `bge-m3` | Ollama embedding model |
| `AML_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint; unreachable ⇒ BM25 only |
| `AML_RATE_LIMIT` | `240` | requests per minute per client IP |
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
- **Retrieval** — hybrid: BM25 over tokenized chunks always, fused with local
  embedding similarity when the Ollama endpoint answers. No outbound calls.
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