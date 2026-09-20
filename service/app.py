"""AML Add/Search memory adapter.

Contract (per AML API Guide, cycle 2):
- POST add:   {request_id, messages:[{role, timestamp?, content}], user_id, session_id}
              -> 200 {success:true, request_id, user_id, session_id}   (synchronous)
- POST search:{query, options?, user_id, top_k}
              -> 200 {data:[{id, content, score?, created_at?}]}       (sorted, <= top_k, per-user scope)
- GET health: unauthenticated, any 2xx.
- Auth: Token / Bearer / X-Api-Key (one chosen at registration).
"""

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

EMBED_MODEL = os.environ.get("AML_EMBED_MODEL", "bge-m3")
OLLAMA_URL = os.environ.get("AML_OLLAMA_URL", "http://127.0.0.1:11434")
MAX_TOP_K = 200
MAX_MESSAGES = 500
MAX_BODY_BYTES = 8 * 1024 * 1024

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

_db_lock = threading.Lock()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


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


init_db()


# ---------------------------------------------------------------- text utils

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str):
    toks = [t.lower() for t in WORD_RE.findall(text)]
    toks += CJK_RE.findall(text)  # single CJK chars act as weak tokens
    return toks


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text)) + len(CJK_RE.findall(text))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- embeddings

_embed_ok: Optional[bool] = None


def embed_texts(texts):
    """Return list of vectors (normalized) or None if Ollama unavailable."""
    global _embed_ok
    if _embed_ok is False:
        return None
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=60.0,
        )
        r.raise_for_status()
        embs = r.json().get("embeddings")
        if not embs or len(embs) != len(texts):
            raise ValueError("bad embeddings")
        out = []
        for v in embs:
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        _embed_ok = True
        return out
    except Exception:
        _embed_ok = False
        return None


# ---------------------------------------------------------------- chunking

def segment_messages(messages):
    """Deterministic segmentation: <=20 messages or <=2000 words per chunk."""
    chunks, cur, cur_words = [], [], 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        w = count_words(content)
        if cur and (len(cur) >= 20 or cur_words + w > 2000):
            chunks.append(cur)
            cur, cur_words = [], 0
        cur.append(content.strip())
        cur_words += w
    if cur:
        chunks.append(cur)
    return ["\n".join(c) for c in chunks]


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
    mx = max(scores) if scores else 0.0
    if mx > 0:
        scores = [s / mx for s in scores]
    return scores


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


@app.middleware("http")
async def harden(request: Request, call_next):
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
    if not auth_ok(request):
        return err(401, "invalid key")
    ip = request.client.host if request.client else "?"
    if not rate_ok(ip):
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "60"},
            content={"detail": {"reason": "rate limited"}},
        )
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
    seg = segment_messages([{"content": t} for t in texts])
    if not seg:
        return err(422, "no usable content")

    vecs = embed_texts(seg)  # None -> BM25-only fallback
    created = now_iso()
    rid_hash = hashlib.sha256(request_id.encode()).hexdigest()[:12]

    with _db_lock, db() as c:
        for i, text in enumerate(seg):
            cid = f"mem_{rid_hash}_{i}"
            blob = None
            if vecs:
                blob = json.dumps(vecs[i]).encode()
            c.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                (cid, user_id, session_id, request_id, text, created,
                 count_words(text), blob),
            )
        c.execute(
            "INSERT OR REPLACE INTO seen_requests VALUES (?,?,?)",
            (request_id, user_id, session_id),
        )

    return {"success": True, "request_id": request_id, "user_id": user_id,
            "session_id": session_id}


@app.post("/search")
async def search(request: Request):
    if not auth_ok(request):
        return err(401, "invalid key")
    ip = request.client.host if request.client else "?"
    if not rate_ok(ip):
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "60"},
            content={"detail": {"reason": "rate limited"}},
        )
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
            "SELECT id, content, created_at, vec FROM chunks WHERE user_id=?",
            (user_id,),
        ).fetchall()

    if not rows:
        return {"data": []}

    ids = [r[0] for r in rows]
    contents = [r[1] for r in rows]
    created = [r[2] for r in rows]
    vecs = [json.loads(r[3]) if r[3] else None for r in rows]

    q_toks = tokenize(query)
    doc_toks = [tokenize(x) for x in contents]
    lex = bm25_scores(q_toks, doc_toks)

    q_vec = None
    if any(v is not None for v in vecs):
        qv = embed_texts([query])
        if qv:
            q_vec = qv[0]

    results = []
    for i in range(len(ids)):
        s = lex[i]
        if q_vec is not None and vecs[i] is not None:
            dot = sum(a * b for a, b in zip(q_vec, vecs[i]))
            s = 0.65 * dot + 0.35 * lex[i]
        results.append((s, i))

    results.sort(key=lambda t: t[0], reverse=True)
    data = []
    for s, i in results[:top_k]:
        item = {"id": ids[i], "content": contents[i]}
        if s > 0:
            item["score"] = round(float(s), 6)
        item["created_at"] = created[i]
        data.append(item)

    return {"data": data}


if __name__ == "__main__":
    import uvicorn
    # Platform-injected PORT = managed hosting (bind all interfaces); local dev binds loopback only
    _port = int(os.environ.get("PORT") or os.environ.get("AML_PORT") or "8021")
    _host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run(app, host=_host, port=_port,
                log_level="warning", access_log=False, server_header=False)
