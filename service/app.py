"""AML Add/Search memory adapter.

Contract (per AML API Guide, cycle 2):
- POST add:   {request_id, messages:[{role, timestamp?, content}], user_id, session_id}
              -> 200 {success:true, request_id, user_id, session_id}   (synchronous)
- POST search:{query, options?, user_id, top_k}
              -> 200 {data:[{id, content, score?, created_at?}]}       (sorted, <= top_k, per-user scope)
- GET health: unauthenticated, any 2xx.
- Auth: Token / Bearer / X-Api-Key (one chosen at registration).

v0.5.0 retrieval upgrades (all hot-path, zero-LLM):
1. Fine-grained chunking: one chunk per message; long messages split into
   ~300-word windows with 50-word overlap (pattern validated by InvMem, the
   cycle-1 open-source #1: coarse multi-message chunks dilute topics and cut
   conditions away from their conclusions).
2. Weighted RRF fusion instead of linear score mixing (dense 1.0 / BM25 0.5,
   k=60): rank-level fusion is scale-free, which the top cycle-1 entries found
   more stable than mixing raw similarity and BM25 magnitudes.
3. Same-session adjacency expansion (+/-1 chunk around top seeds): pulls rule
   premises, Q/A pairs and referents back into the evidence set -- the
   mechanism behind the cycle-1 leaders' lead on multi-hop and rule-following.
4. Temporal-aware reranking (zero LLM): a query with "now/currently/最近/目前"
   style signals (a) boosts chunks by time-decayed ingest recency (exponential,
   half-life in days -- corpus-rank recency breaks at both corpus extremes) and
   (b) soft-penalises chunks whose content carries an explicit OLD date anchor
   ("since 2020"), so superseded facts (moved city, changed job) sink below
   their updates (mem0 v3's "ADD-only + retrieval-time recency" finding,
   graphiti's soft-invalidation intuition).
5. Exact-duplicate suppression at add time (MemOS stage-1 dedup).

The add path stays synchronous-persist-then-ack per the API Guide: by the time
/add returns 200 the raw chunks are queryable; enrichment (event-time parsing)
is deterministic and happens inline -- no background worker, nothing to lose.
"""

import asyncio
import os
import re
import json
import time
import math
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("AML_DB_PATH", APP_DIR / "data" / "memory.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_MANAGED = bool(os.environ.get("PORT"))  # 平台注入 PORT == 托管环境（PaaS / 容器平台）

# 鉴权密钥取「第一个非空」的环境变量，按优先级依次尝试：
#   AML_API_KEY    —— 人工填写的密钥（若平台侧已配置，优先采用）
#   AML_MEMORY_KEY —— 平台自动生成的密钥（Blueprint: generateValue: true）
# 之所以支持两个名字：Render 的 sync:false 在 Blueprint 首次创建时会静默留下空变量，
# 漏填即 fail-fast，表现为「部署失败」；改用 generateValue 可全程免人工输入。
API_KEY = ""
KEY_SOURCE = ""
for _name in ("AML_API_KEY", "AML_MEMORY_KEY"):
    _val = (os.environ.get(_name) or "").strip()
    if _val:
        API_KEY, KEY_SOURCE = _val, _name
        break

if not API_KEY:
    if _MANAGED:
        # 托管环境禁止用空密钥或内置默认密钥对外提供服务：
        # 否则任何人可读写他人记忆，直接违反 AML「样本隔离」红线。
        raise RuntimeError(
            "No API key configured. Refusing to start in managed/PaaS mode: an "
            "unauthenticated or default-key endpoint violates the AML sample-isolation "
            "rule. Set AML_MEMORY_KEY (or AML_API_KEY) in the platform environment "
            "before deploying."
        )
    # 本地开发：生成一次性随机密钥，且服务仅绑回环地址，不落任何固定凭据
    import secrets as _secrets

    API_KEY = _secrets.token_urlsafe(24)
    print(f"[dev] AML_API_KEY not set -> generated ephemeral key (loopback only): {API_KEY}")

# --- dense retrieval legs ---------------------------------------------------
# Ollama leg: a local inference server. Kept for self-hosted runs; on a PaaS
# free tier nothing listens on loopback, so this leg is inert there unless
# AML_OLLAMA_URL is pointed at an external host.
EMBED_MODEL = os.environ.get("AML_EMBED_MODEL", "bge-m3")
OLLAMA_URL = os.environ.get("AML_OLLAMA_URL", "http://127.0.0.1:11434")

# DashScope leg: the hosted text-embedding-v4 model. The open-method division
# pins the embedding model to this exact name, so when a key is present this
# leg takes priority over Ollama.
#
# The default is the Beijing region, not Singapore, because a Model Studio API
# key is issued per region and is not interchangeable: the free v4 quota is
# also granted per region. A key minted in one region returns
# `InvalidApiKey` against the other, so the default has to match the region the
# operator actually provisioned.
DASHSCOPE_KEY = (os.environ.get("AML_DASHSCOPE_KEY") or "").strip()
DASHSCOPE_BASE = (
    os.environ.get("AML_DASHSCOPE_BASE") or "https://dashscope.aliyuncs.com"
).rstrip("/")
DASHSCOPE_MODEL = os.environ.get("AML_DASHSCOPE_MODEL", "text-embedding-v4")
EMBED_DIM = int(os.environ.get("AML_EMBED_DIM") or "1024")
EMBED_BATCH = 10  # text-embedding-v4 accepts at most 10 texts per call
EMBED_TIMEOUT = float(os.environ.get("AML_EMBED_TIMEOUT") or "20")
# A transient network fault must not disable the dense leg for the whole run,
# so a failure parks it for this many seconds instead of latching it off.
EMBED_COOLDOWN = float(os.environ.get("AML_EMBED_COOLDOWN") or "90")

# --- retrieval tuning knobs (v0.5.0) ----------------------------------------
# Weighted-RRF fusion: score = w_dense/(K+rank_dense) + w_lex/(K+rank_lex).
RRF_K = int(os.environ.get("AML_RRF_K", "60"))
W_RRF_DENSE = float(os.environ.get("AML_W_RRF_DENSE", "1.0"))
W_RRF_LEX = float(os.environ.get("AML_W_RRF_LEX", "0.5"))

# Adjacency expansion: this many top-scored chunks act as seeds; each seed's
# same-session neighbours within AML_ADJ_WINDOW of its ordinal are pulled into
# the result with score * AML_ADJ_FACTOR (kept below the seed, above noise).
ADJ_SEEDS = int(os.environ.get("AML_ADJ_SEEDS", "20"))
ADJ_WINDOW = int(os.environ.get("AML_ADJ_WINDOW", "1"))
ADJ_FACTOR = float(os.environ.get("AML_ADJ_FACTOR", "0.9"))

# Temporal reranking, multiplicative on the fused score:
#   current-state query ("where do they live *now*") -> newer chunks boosted
#   by up to W_NOW * recency_norm (recency_norm 0..1 within the user's corpus)
#   a plain recency nudge of W_REC * recency_norm always applies
#   a year/date found in the query boosts chunks containing that year by W_DATE
W_NOW = float(os.environ.get("AML_W_NOW", "0.6"))
W_REC = float(os.environ.get("AML_W_REC", "0.12"))
W_DATE = float(os.environ.get("AML_W_DATE", "0.35"))
# Current-state queries: ingest recency decays with a half-life in days
# (corpus-rank recency breaks at both extremes -- see search()), and chunks
# whose CONTENT carries an explicit old date anchor ("since 2020") are soft-
# penalised as likely-supersided evidence (graphiti's soft invalidation,
# minus the LLM).
RECENCY_HALF_LIFE_DAYS = float(os.environ.get("AML_RECENCY_HALF_LIFE_DAYS", "30"))
W_STALE = float(os.environ.get("AML_W_STALE", "0.4"))
STALE_YEARS = int(os.environ.get("AML_STALE_YEARS", "1"))
# Additive boost for chunks whose content announces a state change ("just
# moved", "last month") when the query asks about the current state.
W_CHANGE = float(os.environ.get("AML_W_CHANGE", "0.5"))
# Temporal factors only reorder chunks inside the SEMANTIC relevance band
# (cosine >= COS_REL_GATE * best cosine). RRF compresses score differences so
# hard that an unconditioned multiplier reshuffles arbitrarily (an irrelevant
# note outranked the actual answer); gating on raw cosine keeps temporal as a
# tie-breaker among genuinely competing facts. With no dense leg there is no
# relevance scale at all, so temporal reordering stays off entirely.
COS_REL_GATE = float(os.environ.get("AML_COS_REL_GATE", "0.5"))

# Fine-grained chunking: long messages become sliding windows.
CHUNK_WORDS = int(os.environ.get("AML_CHUNK_WORDS", "300"))
CHUNK_OVERLAP = int(os.environ.get("AML_CHUNK_OVERLAP", "50"))

# Render injects RENDER_GIT_COMMIT on every build, so the live revision stays
# externally verifiable without anyone maintaining a version string by hand.
VERSION = os.environ.get("AML_VERSION") or "0.6.0"
COMMIT = (os.environ.get("AML_COMMIT")
          or os.environ.get("RENDER_GIT_COMMIT")
          or "unknown")

MAX_TOP_K = 200
MAX_MESSAGES = 500
MAX_BODY_BYTES = int(os.environ.get("AML_MAX_BODY_BYTES") or str(16 * 1024 * 1024))

# --- v0.6.0: cross-encoder rerank leg (FAQ 05: Reranker is NOT restricted) --
# Pipeline: hybrid RRF + temporal + adjacency -> merged ranking; then the top
# AML_RERANK_CANDIDATES chunks go to a DashScope cross-encoder (qwen3-rerank)
# which reorders them by query-document relevance; final top_k is cut from
# the reranked order. Measured on LoCoMo single-hop misses (n=187, local
# bench 2026-09-25): recall@20 0.2557 -> 0.3681 (+11.2pt) at pool=50;
# pool=100 adds +5.4pt more on the 86 recall-leg misses (0.2293 -> 0.2833).
# Safety: any API error/timeout/empty result silently falls back to the
# merged order, so the service never fails because of the rerank leg.
RERANK_ON = os.environ.get("AML_RERANK", "0") == "1"
RERANK_MODEL = os.environ.get("AML_RERANK_MODEL", "qwen3-rerank")
RERANK_CANDIDATES = int(os.environ.get("AML_RERANK_CANDIDATES", "100"))
RERANK_TIMEOUT = float(os.environ.get("AML_RERANK_TIMEOUT", "10"))
# 纯替换式精排会丢掉词法/稠密侧的强信号（2026-09-25 实测 417 题子集：涨 72 / 跌 48，
# 其中多题 gold 从 recall 1.0 直接掉出 top-20）。置 1 则把「精排名次」与「原融合名次」
# 做 RRF 混合后再截断 —— 保留精排的排序增益，同时不让原序里的强命中被一脚踢出。
RERANK_BLEND = os.environ.get("AML_RERANK_BLEND", "0") == "1"
RERANK_BLEND_W = float(os.environ.get("AML_RERANK_BLEND_W", "0.7"))

# Visibility for a leg that fails silently by design (2026-09-25 incident): when
# the rerank call fails the service keeps the merged order, so an evaluation run
# would be scored on a *different* retrieval path with nothing visible from the
# outside. That actually happened once -- the provider account went into
# arrears (HTTP 400 `Arrearage`) and a full 30-minute run silently measured the
# no-rerank order. The last call's outcome and a short reason are therefore
# published on the read-only `GET /version`, so a run can be validated from
# outside without log access. Carries no key, no query text and no documents.
_rerank_state = {"called": 0, "ok": None, "error": None}


def _rerank_fail(reason: str) -> None:
    _rerank_state["called"] += 1
    _rerank_state["ok"] = False
    _rerank_state["error"] = reason


def _rerank_scores(query: str, docs: list[str]) -> Optional[list[int]]:
    """Return doc indices sorted by cross-encoder relevance (desc).

    Returns None on any failure -- caller then keeps the merged order.
    Uses httpx (already a dependency) inside a worker thread.
    """
    api_key = os.environ.get("AML_DASHSCOPE_KEY", "")
    if not api_key or not docs:
        _rerank_fail("no_key" if not api_key else "no_docs")
        return None
    url = ("https://dashscope.aliyuncs.com/api/v1/services/"
           "rerank/text-rerank/text-rerank")
    body = {"model": RERANK_MODEL,
            "input": {"query": query, "documents": docs},
            "parameters": {"return_documents": False,
                           "top_n": len(docs)}}
    try:
        r = httpx.post(url, json=body, timeout=RERANK_TIMEOUT,
                       headers={"Authorization": "Bearer " + api_key,
                                "User-Agent": "aml-memory-service/0.6"})
        r.raise_for_status()
        results = r.json().get("output", {}).get("results")
        if not isinstance(results, list) or not results:
            _rerank_fail("empty_result")
            return None
        order = [x["index"] for x in results
                 if isinstance(x, dict) and isinstance(x.get("index"), int)]
        if not order:
            _rerank_fail("no_index_field")
            return None
        _rerank_state["called"] += 1
        _rerank_state["ok"] = True
        _rerank_state["error"] = None
        return order
    except Exception as e:
        # Status code plus the vendor error code is what makes a failure
        # diagnosable; the request body (query/documents) never appears here.
        code = getattr(getattr(e, "response", None), "status_code", None)
        detail = ""
        if code is not None:
            try:
                payload = e.response.json()
            except Exception:
                payload = {}
            detail = str(payload.get("code") or payload.get("message") or "")[:80]
        _rerank_fail(f"HTTP {code} {detail}".strip() if code else type(e).__name__)
        return None

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

_db_lock = threading.Lock()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _table_columns(c, table):
    return {row[1] for row in c.execute(f"PRAGMA table_info({table})")}


def init_db():
    with _db_lock, db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              session_id TEXT NOT NULL,
              request_id TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at TEXT NOT NULL,
              words INTEGER NOT NULL,
              vec BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);
            CREATE TABLE IF NOT EXISTS seen_requests (
              request_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              session_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
            """
        )
        # v0.5.0 columns, added in-place so an existing production DB migrates
        # on first boot instead of forcing a wipe (the Starter-disk DB holds
        # real ingested history).
        cols = _table_columns(c, "chunks")
        if "sess_seq" not in cols:
            c.execute("ALTER TABLE chunks ADD COLUMN sess_seq INTEGER")
        if "event_time" not in cols:
            c.execute("ALTER TABLE chunks ADD COLUMN event_time TEXT")
        if "content_hash" not in cols:
            c.execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_user_session "
            "ON chunks(user_id, session_id, sess_seq)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(user_id, content_hash)"
        )


init_db()


# ---------------------------------------------------------------- text utils

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Absolute dates/years in content, kept as the chunk's event-time anchor:
# ISO/ slashes/dotted dates, bare years, and 年-月-日 CJK forms.
DATE_RE = re.compile(
    r"\b((?:19|20)\d{2})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?\b"
    r"|((?:19|20)\d{2})\s*年(?:\s*(\d{1,2})\s*月)?(?:\s*(\d{1,2})\s*[日号])?"
    r"|\b((?:19|20)\d{2})\b"
)
# Queries implying "what is true right now" -> recency should matter.
NOW_HINT_RE = re.compile(
    r"\b(now|currently|current|nowadays|these days|latest|today|tonight|present|as of|so far)\b"
    r"|现在|如今|目前|当前|最近|近来|当下|现今|这阵子|眼下",
    re.IGNORECASE,
)
# Content announcing a STATE CHANGE ("just moved", "last month") -- the
# update-language invalidation signal (mem0/graphiti) without an LLM. For
# current-state queries such chunks are the fresh side of an update.
RECENT_CHANGE_RE = re.compile(
    r"\b(just|recently|newly)\s+(moved|started|joined|switched|changed|upgraded|left|quit|bought|sold|got|became)\b"
    r"|\b(last month|last week|yesterday|this month|this year|a few days ago|new job|new city|new home)\b"
    r"|刚刚|最近|刚搬|新工作|上个月|上周|昨天|今年",
    re.IGNORECASE,
)


def tokenize(text: str):
    toks = [t.lower() for t in WORD_RE.findall(text)]
    toks += CJK_RE.findall(text)  # single CJK chars act as weak tokens
    return toks


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text)) + len(CJK_RE.findall(text))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_event_time(text: str) -> Optional[str]:
    """First absolute date/year found in the content, normalised.

    This is the chunk's world-time anchor (graphiti's `valid_at` intuition,
    minus the LLM): used by search to tell "stated in 2023" from "stated
    yesterday" without trusting the ingest clock alone.
    """
    m = DATE_RE.search(text)
    if not m:
        return None
    year = m.group(1) or m.group(4) or m.group(7)
    month = m.group(2) or m.group(5)
    day = m.group(3) or m.group(6)
    if month and day:
        return f"{year}-{int(month):02d}-{int(day):02d}"
    if month:
        return f"{year}-{int(month):02d}"
    return year


def query_years(query: str):
    """Years the query itself is asking about (e.g. 'What happened in 2024?')."""
    return sorted({m.group(0) for m in re.finditer(r"\b(?:19|20)\d{2}\b", query)})


def split_long_message(text: str, limit: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP):
    """Sliding-window split for one long message (InvMem: ~320/40 windows)."""
    words = text.split()
    if len(words) <= limit:
        return [text]
    step = max(1, limit - overlap)
    out = []
    for i in range(0, len(words), step):
        win = words[i:i + limit]
        if win:
            out.append(" ".join(win))
        if i + limit >= len(words):
            break
    return out


def build_chunks(messages):
    """One unit per message; long messages -> overlapping windows.

    Keeps every message a separate retrieval unit so adjacency expansion has
    natural neighbours (InvMem's message-boundary chunking).
    """
    units = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        units.extend(split_long_message(content.strip()))
    return units


# ---------------------------------------------------------------- embeddings

_embed_state = {"ok": None, "backend": None, "fail_until": 0.0}


def _l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dashscope_headers():
    return {
        "Authorization": f"Bearer {DASHSCOPE_KEY}",
        "Content-Type": "application/json",
    }


def _via_dashscope_native(batch, text_type):
    """`text-embedding-v4` through the native DashScope endpoint.

    The native route is preferred over the OpenAI-compatible one because only
    it exposes `text_type`, and the query side here is a short question matched
    against much longer stored chunks -- exactly the asymmetry that parameter
    exists to correct. Returns None on any non-2xx or unparseable body, so the
    caller can fall through to the compatible route.
    """
    url = f"{DASHSCOPE_BASE}/api/v1/services/embeddings/text-embedding/text-embedding"
    body = {
        "model": DASHSCOPE_MODEL,
        "input": {"texts": batch},
        "parameters": {"dimension": EMBED_DIM, "text_type": text_type},
    }
    try:
        r = httpx.post(url, headers=_dashscope_headers(), json=body,
                       timeout=EMBED_TIMEOUT)
        if r.status_code >= 400:
            return None
        embs = ((r.json() or {}).get("output") or {}).get("embeddings") or []
        embs = sorted(embs, key=lambda e: e.get("text_index", 0))
        vecs = [e.get("embedding") for e in embs]
        if vecs and all(vecs):
            return [_l2(v) for v in vecs]
    except Exception:
        pass
    return None


def _via_dashscope_compatible(batch):
    """Same model through the OpenAI-compatible route (no `text_type`)."""
    url = f"{DASHSCOPE_BASE}/compatible-mode/v1/embeddings"
    body = {"model": DASHSCOPE_MODEL, "input": batch, "dimensions": EMBED_DIM}
    try:
        r = httpx.post(url, headers=_dashscope_headers(), json=body,
                       timeout=EMBED_TIMEOUT)
        r.raise_for_status()
        data = (r.json() or {}).get("data") or []
        data = sorted(data, key=lambda d: d.get("index", 0))
        vecs = [d.get("embedding") for d in data]
        if vecs and all(vecs):
            return [_l2(v) for v in vecs]
    except Exception:
        pass
    return None


def _via_dashscope(texts, text_type):
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        vecs = _via_dashscope_native(batch, text_type)
        if vecs is None:
            vecs = _via_dashscope_compatible(batch)
        if vecs is None or len(vecs) != len(batch):
            return None
        out.extend(vecs)
    return out


def _via_ollama(texts):
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=60.0,
        )
        r.raise_for_status()
        embs = r.json().get("embeddings")
        if not embs or len(embs) != len(texts):
            return None
        return [_l2(v) for v in embs]
    except Exception:
        return None


def embed_texts(texts, text_type="document"):
    """Dense leg. Returns L2-normalised vectors, or None to fall back to BM25.

    Order: hosted `text-embedding-v4` when a key is configured, then local
    Ollama. A failure parks the leg for EMBED_COOLDOWN seconds instead of
    latching it off for the process lifetime -- an earlier permanent latch meant
    one transient blip at startup silently downgraded every later request of
    the run, and that is invisible from the outside.
    """
    if not texts:
        return None
    if _embed_state["fail_until"] > time.time():
        return None
    vecs, backend = None, None
    if DASHSCOPE_KEY:
        vecs = _via_dashscope(texts, text_type)
        if vecs is not None:
            backend = "dashscope"
    if vecs is None:
        vecs = _via_ollama(texts)
        if vecs is not None:
            backend = "ollama"
    if vecs is None:
        _embed_state.update(ok=False, backend=None,
                            fail_until=time.time() + EMBED_COOLDOWN)
        return None
    _embed_state.update(ok=True, backend=backend, fail_until=0.0)
    return vecs


# ---------------------------------------------------------------- BM25

def bm25_scores(query_toks, docs_toks, k1=1.2, b=0.75):
    n_docs = len(docs_toks)
    if n_docs == 0:
        return [0.0] * n_docs
    avgdl = sum(len(d) for d in docs_toks) / n_docs or 1.0
    df = {}
    for d in docs_toks:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = []
    for d in docs_toks:
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in query_toks:
            if t not in tf:
                continue
            idf = math.log(1 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores


def _ranks(scores):
    """0-based ranks, best first; ties keep stable order."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


# ---------------------------------------------------------------- auth / limits

def auth_ok(request: Request) -> bool:
    key = request.headers.get("x-api-key")
    if key:
        return key == API_KEY
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() == API_KEY
    if auth.lower().startswith("token "):
        return auth[6:].strip() == API_KEY
    if auth and not auth.lower().startswith(("basic ", "digest ")):
        return auth.strip() == API_KEY
    return False


_RATE = {}
RATE_LIMIT = int(os.environ.get("AML_RATE_LIMIT", "240"))  # req/min per IP


def rate_ok(ip: str) -> bool:
    now = time.time()
    win = _RATE.setdefault(ip, [])
    while win and now - win[0] > 60:
        win.pop(0)
    if len(win) >= RATE_LIMIT:
        return False
    win.append(now)
    return True


def guard(request: Request):
    """Auth + brute-force throttle for the two POST routes.

    Order matters, and this order is deliberate: an authenticated caller is
    **never** throttled, and the window is only charged for *rejected*
    authentications. The evaluator is the only credentialled client, so a
    limiter that also counted its traffic could only ever fire at the one actor
    we need to serve -- and a 429 answers with `Retry-After: 60`, i.e. sixty
    seconds of evaluation wall-clock per false trigger, against a budget of two
    Full runs per track whose second one is locked for 30 days. Keeping the
    limiter on failed auth keeps the brute-force protection (and the abuse
    ceiling) without putting the scored run behind a counter.

    Returns None when the request may proceed, or the response to send back.
    """
    if auth_ok(request):
        return None
    ip = request.client.host if request.client else "?"
    if not rate_ok(ip):
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "60"},
            content={"detail": {"reason": "rate limited"}},
        )
    return err(401, "invalid key")


@app.middleware("http")
async def harden(request: Request, call_next):
    # Reject an oversized body before it is read into memory: the container has a
    # hard memory ceiling, and being OOM-killed mid-evaluation costs a scored run,
    # whereas a 413 is an ordinary contract answer the harness already understands.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": {"reason": "payload too large"}},
        )
    resp = await call_next(request)
    resp.headers["Server"] = "ws"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


def err(status: int, reason: str):
    return JSONResponse(status_code=status, content={"detail": {"reason": reason}})


# ---------------------------------------------------------------- endpoints

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    """Read-only build/configuration identity.

    Exists so a reviewer can confirm from outside which revision is live and
    which embedding leg actually backs `/search` -- a system whose retrieval
    path cannot be identified from the outside cannot be version-reviewed.
    Deliberately carries no secret and no request data: it names the model and
    the leg in use, never a key, a filesystem path or a sample.
    """
    if DASHSCOPE_KEY:
        leg, model = "dashscope", DASHSCOPE_MODEL
    elif _embed_state["backend"] == "ollama":
        leg, model = "ollama", EMBED_MODEL
    else:
        leg, model = "bm25-only", None
    return {
        "version": VERSION,
        "commit": COMMIT,
        "contract": ["POST /add", "POST /search", "GET /health", "GET /version"],
        # The description is assembled from the values actually in force, so it
        # cannot drift from behaviour the way a hand-maintained string does.
        "retrieval": ("bm25 + dense weighted-RRF "
                      f"({W_RRF_DENSE:g}/{W_RRF_LEX:g}, k={RRF_K}) "
                      "+ temporal rerank + adjacency expansion"
                      + (f" + cross-encoder rerank ({RERANK_MODEL}, "
                         f"pool={RERANK_CANDIDATES}"
                         + (", blended" if RERANK_BLEND else "") + ")"
                         if RERANK_ON else "")),
        "dense_leg": {
            "backend": leg,
            "model": model,
            "dim": EMBED_DIM if model else None,
            "healthy": _embed_state["ok"],
        },
        # Exposed because this leg degrades silently by design (see the note on
        # `_rerank_state`): a run must be checkable from outside before it is
        # submitted. `healthy` is null until the first rerank call happens.
        "rerank_leg": {
            "enabled": RERANK_ON,
            "model": RERANK_MODEL if RERANK_ON else None,
            "candidates": RERANK_CANDIDATES if RERANK_ON else None,
            "blended": RERANK_BLEND if RERANK_ON else None,
            "calls": _rerank_state["called"],
            "healthy": _rerank_state["ok"],
            "last_error": _rerank_state["error"],
        },
    }


# ------------------------------------------------- keep-alive（可选，默认关闭）
# 背景：Render 免费实例 15 分钟无入站流量即休眠；休眠会停容器，而免费档文件系统是
# 临时的，SQLite 里的记忆随之丢失。若评测在「写入」与「检索」之间跨越一次休眠，
# 结果会凭空消失。打开本开关即可用自流量把实例钉住（长驻 = 数据不丢）。
#
# 开启方式：给服务加一个环境变量 AML_KEEPALIVE_URL，值填本服务自己的公网健康检查地址，
# 例如 https://agent-memory-service.onrender.com/health（不带任何密钥）。
#   ⚠️ 必须指向 /health 之类的真实路由。**绝不能指向 /robots.txt** —— 免费实例休眠期间
#   robots.txt 由 Render 自己拦截返回，请求根本到不了应用，看着是 200 却永远唤不醒服务。
#   ⚠️ 代价要先算清：Render 每工作区每月只有 750 实例小时，常驻一个服务约吃 720 小时，
#   几乎占满整个配额；若工作区里还有别的免费服务，会被一起挤停。
# 不设置该变量时，本段代码完全不生效，无任何副作用。
KEEPALIVE_URL = (os.environ.get("AML_KEEPALIVE_URL") or "").strip()
KEEPALIVE_SECONDS = max(60, int(os.environ.get("AML_KEEPALIVE_SECONDS") or "600"))


def keepalive_loop(url: str, interval: float) -> None:
    while True:
        time.sleep(interval)
        try:
            r = httpx.get(url, timeout=90.0)
            print(f"[keepalive] GET {url} -> {r.status_code}")
        except Exception as e:  # 网络抖动不该拖垮服务
            print(f"[keepalive] GET {url} failed: {type(e).__name__}: {e}")


if KEEPALIVE_URL:
    threading.Thread(
        target=keepalive_loop, args=(KEEPALIVE_URL, KEEPALIVE_SECONDS), daemon=True
    ).start()
    print(f"[keepalive] enabled: every {KEEPALIVE_SECONDS}s -> {KEEPALIVE_URL}")


@app.post("/add")
async def add(request: Request):
    denied = guard(request)
    if denied is not None:
        return denied
    try:
        body = json.loads(await request.body())
    except Exception:
        return err(422, "invalid json")

    request_id = body.get("request_id")
    user_id = body.get("user_id")
    session_id = body.get("session_id")
    messages = body.get("messages")
    if not isinstance(request_id, str) or not request_id:
        return err(422, "request_id required")
    if not isinstance(user_id, str) or not user_id:
        return err(422, "user_id required")
    if not isinstance(session_id, str) or not session_id:
        return err(422, "session_id required")
    if not isinstance(messages, list) or not messages:
        return err(422, "messages required")
    if len(messages) > MAX_MESSAGES:
        return err(422, "too many messages")
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            return err(422, "each message needs role")
        c = m.get("content")
        if isinstance(c, str):
            continue
        if isinstance(c, list) and all(
            isinstance(x, dict) and ("text" in x or "image_url" in x) for x in c
        ):
            continue
        return err(422, "each message needs string or content-array 'content'")

    # idempotent retry with same request_id -> success echo
    with _db_lock, db() as c:
        row = c.execute(
            "SELECT user_id, session_id FROM seen_requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
    if row:
        return {
            "success": True,
            "request_id": request_id,
            "user_id": row[0],
            "session_id": row[1],
        }

    texts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            texts.append(c)
        else:  # multimodal content array -> keep text parts
            texts.append(" ".join(x.get("text", "") for x in c if isinstance(x, dict)))
    units = build_chunks([{"content": t} for t in texts])
    if not units:
        return err(422, "no usable content")

    # Off the event loop: the embedding call is blocking HTTP (up to
    # EMBED_TIMEOUT) and this handler is async, so calling it inline would
    # serialise every concurrent request behind one round-trip each. The
    # evaluator drives Add/Search at the concurrency declared at registration,
    # so inline would turn N parallel calls into N sequential ones.
    vecs = await asyncio.to_thread(embed_texts, units, "document")  # None -> BM25
    created = now_iso()
    rid_hash = hashlib.sha256(request_id.encode()).hexdigest()[:12]

    inserted = 0
    with _db_lock, db() as c:
        # Per-session ordinal for adjacency expansion: continue the session's
        # sequence so chunks from different requests in one session chain up.
        base = c.execute(
            "SELECT COALESCE(MAX(sess_seq), -1) FROM chunks "
            "WHERE user_id=? AND session_id=?",
            (user_id, session_id),
        ).fetchone()[0] + 1
        for i, text in enumerate(units):
            chash = hashlib.sha256(
                (user_id + "\x00" + text).encode()
            ).hexdigest()
            # Exact-duplicate suppression (MemOS stage-1): the same content for
            # the same user adds no retrieval value and only pollutes the
            # evidence list. The request itself stays recorded for idempotency.
            dup = c.execute(
                "SELECT 1 FROM chunks WHERE user_id=? AND content_hash=? LIMIT 1",
                (user_id, chash),
            ).fetchone()
            if dup:
                continue
            cid = f"mem_{rid_hash}_{i}"
            blob = None
            if vecs:
                blob = json.dumps(vecs[i]).encode()
            c.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (cid, user_id, session_id, request_id, text, created,
                 count_words(text), blob, base + i, parse_event_time(text), chash),
            )
            inserted += 1
        c.execute(
            "INSERT OR REPLACE INTO seen_requests VALUES (?,?,?)",
            (request_id, user_id, session_id),
        )

    return {"success": True, "request_id": request_id, "user_id": user_id,
            "session_id": session_id, "chunks_stored": inserted}


@app.post("/search")
async def search(request: Request):
    denied = guard(request)
    if denied is not None:
        return denied
    try:
        body = json.loads(await request.body())
    except Exception:
        return err(422, "invalid json")

    query = body.get("query")
    user_id = body.get("user_id")
    top_k = body.get("top_k")
    if not isinstance(query, str) or not query.strip():
        # multimodal: content array
        if isinstance(query, list):
            query = " ".join(
                x.get("text", "") for x in query if isinstance(x, dict)
            )
    if not isinstance(query, str) or not query.strip():
        return err(422, "query required")
    if not isinstance(user_id, str) or not user_id:
        return err(422, "user_id required")
    if not isinstance(top_k, int) or top_k <= 0:
        return err(422, "top_k required")
    top_k = min(top_k, MAX_TOP_K)

    with _db_lock, db() as c:
        rows = c.execute(
            "SELECT id, content, created_at, vec, sess_seq, session_id, event_time "
            "FROM chunks WHERE user_id=? ORDER BY rowid",
            (user_id,),
        ).fetchall()

    if not rows:
        return {"data": []}

    ids = [r[0] for r in rows]
    contents = [r[1] for r in rows]
    created = [r[2] for r in rows]
    vecs = [json.loads(r[3]) if r[3] else None for r in rows]
    sess_seq = [r[4] for r in rows]
    sessions = [r[5] for r in rows]
    event_times = [r[6] for r in rows]

    q_toks = tokenize(query)
    doc_toks = [tokenize(x) for x in contents]
    lex = bm25_scores(q_toks, doc_toks)
    lex_rank = _ranks(lex)

    dense_rank = None
    q_vec = None
    if any(v is not None for v in vecs):
        qv = await asyncio.to_thread(embed_texts, [query], "query")
        if qv:
            q_vec = qv[0]
    if q_vec is not None:
        dense = []
        for i in range(len(ids)):
            if vecs[i] is None:
                dense.append(0.0)
                continue
            dense.append(sum(a * b for a, b in zip(q_vec, vecs[i])))
        dense_rank = _ranks(dense)
        # Semantic relevance band: chunks whose raw cosine is within
        # COS_REL_GATE of the best. Only these may be reordered by the
        # temporal factors; everything else keeps factor 1.0.
        _max_cos = max(dense)
        relevant = {
            i for i in range(len(ids))
            if vecs[i] is not None and dense[i] >= COS_REL_GATE * _max_cos
        }

    # --- temporal signals ----------------------------------------------------
    q_years = set(query_years(query))
    current_state = bool(NOW_HINT_RE.search(query))
    n = len(ids)
    if n > 1:
        # Ingest recency must be TIME-based, not corpus-rank: rank-based
        # recency makes "newest of N" = 1.0, so in a small corpus an
        # irrelevant note written last rides the boost past genuinely
        # relevant older facts, and in a large one everything outside the
        # last few chunks gets ~0. Exponential decay with a day half-life
        # behaves the same at any corpus size.
        now_ts = time.time()
        recency = {}
        for i in range(n):
            try:
                age_days = max(
                    0.0,
                    (now_ts - time.mktime(
                        time.strptime(created[i][:19], "%Y-%m-%dT%H:%M:%S")
                    )) / 86400.0,
                )
            except Exception:
                age_days = 0.0
            recency[i] = 1.0 / (1.0 + age_days / RECENCY_HALF_LIFE_DAYS)
    else:
        recency = {0: 1.0}

    # Content-level date anchor: an absolute year in the chunk's own text.
    _this_year = time.gmtime().tm_year

    def _stale_anchor(i):
        et = event_times[i]
        if not et:
            return False
        try:
            return int(str(et)[:4]) <= _this_year - STALE_YEARS
        except Exception:
            return False

    # --- weighted RRF fusion + temporal factors ------------------------------
    fused = {}
    for i in range(n):
        s = 0.0
        if dense_rank is not None:
            s += W_RRF_DENSE / (RRF_K + dense_rank[i] + 1)
        s += W_RRF_LEX / (RRF_K + lex_rank[i] + 1)
        factor = 1.0
        if q_vec is not None and relevant and i in relevant:
            factor += W_REC * recency[i]
            if current_state:
                factor += W_NOW * recency[i]
            if current_state:
                factor += W_NOW * recency[i]
                # Update language beats surface similarity: the embedding can
                # rank an irrelevant "the user owns X" chunk top-1 on a shared
                # "the user" fragment (measured 0.438 vs 0.368), so chunks
                # announcing a state change get their own additive boost.
                if RECENT_CHANGE_RE.search(contents[i]):
                    factor += W_CHANGE
                # "Where do they live NOW": a chunk explicitly anchored to an
                # old year is the superseded side of an update, not current
                # state. Soft multiplier -- strongly relevant dated facts
                # (those that also pass the relevance gate) can still win.
                if _stale_anchor(i):
                    factor *= (1.0 - W_STALE)
        if q_years and any(y in contents[i] for y in q_years):
            factor += W_DATE
        fused[i] = s * factor

    # --- adjacency expansion (InvMem pattern) --------------------------------
    # Seeds = top ADJ_SEEDS fused chunks. Each seed's same-session neighbours
    # (sess_seq within +/- ADJ_WINDOW) re-enter the ranking at
    # seed_score * ADJ_FACTOR, so the rule premise, the matching question, or
    # the referent an otherwise-top seed points at travels with it. A chunk
    # reachable from several seeds keeps its best (max) score.
    id_pos = {cid: i for i, cid in enumerate(ids)}
    seeds = sorted(range(n), key=lambda i: fused[i], reverse=True)[:ADJ_SEEDS]
    merged = dict(fused)
    for s_idx in seeds:
        seq = sess_seq[s_idx]
        if seq is None:
            continue
        with _db_lock, db() as c:
            neigh = c.execute(
                "SELECT id, content, created_at FROM chunks "
                "WHERE user_id=? AND session_id=? AND sess_seq BETWEEN ? AND ?",
                (user_id, sessions[s_idx], seq - ADJ_WINDOW, seq + ADJ_WINDOW),
            ).fetchall()
        for nid, ncontent, ncreated in neigh:
            score = fused[s_idx] * ADJ_FACTOR
            if nid in id_pos:
                pos = id_pos[nid]
                if score > merged.get(pos, 0.0):
                    merged[pos] = score
            else:
                # chunk not in the corpus snapshot (inserted after the fetch)
                id_pos[nid] = len(ids)
                ids.append(nid)
                contents.append(ncontent)
                created.append(ncreated)
                sess_seq.append(None)
                sessions.append(sessions[s_idx])
                event_times.append(None)
                merged[id_pos[nid]] = score

    # --- v0.6.0 cross-encoder rerank leg -------------------------------------
    # Merged order (hybrid+temporal+adjacency) supplies candidates; the
    # cross-encoder reorders the top RERANK_CANDIDATES by true query-document
    # relevance. Falls back to the merged order on any failure.
    merged_sorted = sorted(merged.items(), key=lambda t: t[1], reverse=True)
    if RERANK_ON and merged_sorted:
        cand = merged_sorted[:RERANK_CANDIDATES]
        order = await asyncio.to_thread(
            _rerank_scores, query, [contents[i] for i, _ in cand])
        if order and all(0 <= x < len(cand) for x in order):
            seen = set(order)
            if RERANK_BLEND:
                bl = {}
                for pos, x in enumerate(order):            # 精排名次
                    bl[x] = bl.get(x, 0.0) + RERANK_BLEND_W / (RRF_K + pos + 1)
                for pos in range(len(cand)):               # 原融合名次
                    bl[pos] = bl.get(pos, 0.0) + (1.0 - RERANK_BLEND_W) / (RRF_K + pos + 1)
                new_order = sorted(bl, key=lambda x: -bl[x])
                merged_sorted = ([cand[x] for x in new_order]
                                 + merged_sorted[len(cand):])
            else:
                # reranked part first, then the rest in merged order (stable tail)
                tail = [x for x in range(len(cand)) if x not in seen]
                merged_sorted = [cand[x] for x in order + tail] + merged_sorted[len(cand):]

    results = []
    for i, s in merged_sorted[:top_k]:
        item = {"id": ids[i], "content": contents[i]}
        if s > 0:
            item["score"] = round(float(s), 6)
        item["created_at"] = created[i]
        results.append(item)
    return {"data": results}


if __name__ == "__main__":
    import uvicorn
    # Platform-injected PORT = managed hosting (bind all interfaces); local dev binds loopback only
    _port = int(os.environ.get("PORT") or os.environ.get("AML_PORT") or "8021")
    _host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run(app, host=_host, port=_port,
                log_level="warning", access_log=False, server_header=False)
