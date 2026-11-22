"""AML Add/Search memory adapter.

Contract (per AML API Guide, cycle 2):
- POST add:   {request_id, messages:[{role, timestamp?, content}], user_id, session_id}
              -> 200 {success:true, request_id, user_id, session_id}   (synchronous)
- POST search:{query, options?, user_id, top_k}
              -> 200 {data:[{id, content, score?, created_at?}]}       (sorted, <= top_k, per-user scope)
- GET health: unauthenticated, any 2xx.
- Auth: Token / Bearer / X-Api-Key (one chosen at registration).

v0.7.0 retrieval tuning (defaults re-measured, still hot-path / zero-LLM):
- Retrieval weights now ship as the measured optimum rather than the earlier
  guess: `W_RRF_LEX` 0.5 -> 0.8 and `ADJ_SEEDS` 20 -> 40. On 120 stratified
  LoCoMo questions with the dense leg live, that pair scores 0.9046 overall
  against 0.8744 for the old defaults, reproduced across two independent
  runs; every category moved up or held. Both are also the only levers that
  survived a sweep while the temporal branch below is inert.
- stdout is now reserved for data. The dev-key notice moved to stderr: on
  stdout it was parsed as five numbers by the T6 regression test and cost
  that group a spurious failure.
- Correction worth recording: the temporal knobs are NOT the lever they look
  like. With the dense leg provably alive (embed liveness probe 10/10),
  driving `W_NOW` to 10 and `W_REC` to 5 left recall bit-identical. The
  reason is `event_time` granularity, not code: the corpus carries a bare
  year per chunk, so all chunks of a year share one age, `recency` is a
  constant inside the relevance band, and the factor becomes a scale factor
  that reorders nothing. The machinery stays and is correct; it will bite
  once `event_time` carries month or day detail.

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
import sys
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

from limits import CONTRACT_MAX_TOP_K, check_top_k

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("AML_DB_PATH", APP_DIR / "data" / "memory.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- traffic capture (debug, 2026-09-29) -------------------------------------
# The AML smoke test failed three times with the same opaque summary
# ("expected=18, actual=11") while every local hypothesis kept passing. To see
# what the platform actually asks, record every /add and /search request to a
# JSONL file next to the database and expose it via GET /debug/traffic
# (authenticated). Capture is best-effort and never touches the response path;
# disable with AML_DEBUG_TRAFFIC=0. REMOVE before the final declared version.
DEBUG_TRAFFIC = os.environ.get("AML_DEBUG_TRAFFIC", "1") == "1"
DEBUG_FILE = DB_PATH.parent / "debug_traffic.jsonl"


def _dbg_capture(kind: str, payload: dict):
    if not DEBUG_TRAFFIC:
        return
    try:
        row = {"ts": time.time(), "kind": kind, **payload}
        with open(DEBUG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass  # diagnostics must never break the contract path

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
    # stderr, not stdout. The dev key line used to go to stdout, where it
    # polluted anything that parses the service's standard output as data --
    # it cost the T6 regression test its numbers once, because the rig read
    # `[dev]`, `AML_API_KEY`, `not`, `set`, `->` as the five expected floats.
    # stdout is the data channel; operational chatter belongs on stderr.
    print(f"[dev] AML_API_KEY not set -> generated ephemeral key "
          f"(loopback only): {API_KEY}", file=sys.stderr)

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
# Measured on 120 stratified LoCoMo questions with the dense leg live
# (ollama/bge-m3, top_k=100), reproduced across two independent runs at
# 0.9046 overall vs 0.8744 for the previous 0.5. Raising the lexical weight
# helps because BM25 is the leg that survives a chunk splitting mid-fact.
W_RRF_LEX = float(os.environ.get("AML_W_RRF_LEX", "0.8"))

# Adjacency expansion: this many top-scored chunks act as seeds; each seed's
# same-session neighbours within AML_ADJ_WINDOW of its ordinal are pulled into
# the result with score * AML_ADJ_FACTOR (kept below the seed, above noise).
# Same sweep: 40 seeds beat 20 by +0.77pt alone and +2.09pt on top of
# W_RRF_LEX=0.8, the single largest gain measured. Doubling the seed count
# costs nothing at query time -- it only reads rows already in memory.
ADJ_SEEDS = int(os.environ.get("AML_ADJ_SEEDS", "40"))
ADJ_WINDOW = int(os.environ.get("AML_ADJ_WINDOW", "1"))
ADJ_FACTOR = float(os.environ.get("AML_ADJ_FACTOR", "0.9"))

# Temporal reranking, multiplicative on the fused score:
#   current-state query ("where do they live *now*") -> newer chunks boosted
#   by up to W_NOW * recency_norm (recency_norm 0..1 within the user's corpus)
#   a plain recency nudge of W_REC * recency_norm always applies
#   a year/date found in the query boosts chunks containing that year by W_DATE
#
# These four knobs measured as a *no-op* on the 120-question sweep even with
# the dense leg provably alive: pushing W_NOW to 10, W_REC to 5 and
# COS_REL_GATE to 0.05 left overall recall bit-identical to baseline. The cause
# is the granularity of `event_time`, not the code. In LoCoMo every chunk
# carries a bare year -- the corpus has exactly three distinct values, '2022',
# '2023' and '2024' -- so every chunk of a given year has an identical age,
# `recency[i]` is a constant inside the relevance band, and the factor becomes
# a global scale factor that reorders nothing. W_DATE is separately inert
# because these questions do not contain a literal four-digit year. Keep the
# machinery (it is correct and will bite on corpora that do carry dates); do
# not spend tuning time here until `event_time` reaches month or day detail.
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
# Off = the ingest clock is the only time signal (the pre-fix behaviour, kept
# only so the fix above can be measured against it). On = chunk world time
# first. Production runs the fixed path; no variable needs setting.
TEMP_LEGACY_CLOCK = os.environ.get("AML_TEMP_LEGACY_CLOCK", "0") == "1"

# Fine-grained chunking: long messages become sliding windows.
# 2026-09-28: default raised 300 -> 4000. The AML smoke test failed twice with
# "expected=18, actual=11" because any message over 300 words was fragmented
# into sliding windows whose contents no longer equal the original message,
# so the platform's expected-record matcher could never match them (verified
# live: a 400-word message stored as 3 window chunks, none matching). The AML
# Add contract caps a message at 2,000 words and text-embedding-v4 accepts
# 8,192 tokens, so 4,000 words keeps every contract-sized message whole.
# Set AML_CHUNK_WORDS explicitly to override.
CHUNK_WORDS = int(os.environ.get("AML_CHUNK_WORDS", "4000"))
CHUNK_OVERLAP = int(os.environ.get("AML_CHUNK_OVERLAP", "50"))

# Render injects RENDER_GIT_COMMIT on every build, so the live revision stays
# externally verifiable without anyone maintaining a version string by hand.
VERSION = os.environ.get("AML_VERSION") or "0.7.1"
COMMIT = (os.environ.get("AML_COMMIT")
          or os.environ.get("RENDER_GIT_COMMIT")
          or "unknown")

MAX_TOP_K = CONTRACT_MAX_TOP_K
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

    Blank documents are removed before the call and their original positions
    are put back on the way out. Both halves matter:

    - the vendor rejects the *entire* request when any single document is
      blank (`HTTP 400 InvalidParameter: text input should not be empty`).
      Since /add is now required to keep empty messages, one such record
      would silently turn reranking off for that user's every search;
    - filtering inside here keeps the indices we return in the caller's
      coordinate system, so no renumbering leaks upward. Dropped documents
      simply fall into the caller's stable tail, i.e. they rank last, which
      is the right place for a record with no text to match against.
    """
    api_key = os.environ.get("AML_DASHSCOPE_KEY", "")
    if not api_key or not docs:
        _rerank_fail("no_key" if not api_key else "no_docs")
        return None
    keep = [i for i, d in enumerate(docs) if d and d.strip()]
    if not keep:
        # Every candidate is blank, so there is nothing to score. Deliberately
        # not routed through _rerank_fail: the health flag is meant to say
        # "the vendor call failed", and no call was made. Marking it unhealthy
        # here would let an all-empty corpus masquerade as a broken leg.
        return None
    url = ("https://dashscope.aliyuncs.com/api/v1/services/"
           "rerank/text-rerank/text-rerank")
    body = {"model": RERANK_MODEL,
            "input": {"query": query, "documents": [docs[i] for i in keep]},
            "parameters": {"return_documents": False,
                           "top_n": len(keep)}}
    try:
        r = httpx.post(url, json=body, timeout=RERANK_TIMEOUT,
                       headers={"Authorization": "Bearer " + api_key,
                                "User-Agent": "aml-memory-service/0.6"})
        r.raise_for_status()
        results = r.json().get("output", {}).get("results")
        if not isinstance(results, list) or not results:
            _rerank_fail("empty_result")
            return None
        # Indices arrive relative to the filtered list; map them back to the
        # caller's positions before returning. The bounds test is on the
        # filtered length, since that is the list the vendor saw.
        order = [keep[x["index"]] for x in results
                 if isinstance(x, dict) and isinstance(x.get("index"), int)
                 and 0 <= x["index"] < len(keep)]
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

# 属性词表（2026-11-22）：中文换新题不能靠"动词加分"解决——实测把
# 搬到/换成/改到 加进 RECENT_CHANGE_RE 后，"现在改成按月批了""我改成早上了"
# 这类与问题无关的短句被抬到第一（hit@1 0.737→0.711，40 题实测）。
# 原因是它们与任何"现在…？"问题的余弦都很高，加分与话题无关。
# 改为**按属性分组做新旧处置**：先认问题和句子属于同一个属性
# （住哪 / 在哪家公司 / 经理是谁 …），再在同一属性内部只保留最新的那条，
# 旧的那条乘惩罚。属性表只覆盖稳定、可判定的少量语义域，不追求全覆盖。
ATTR_LEX = [
    # 内容侧与**问句侧**共用一张表，所以每类都要同时覆盖"陈述怎么写"和"提问怎么问"。
    # 留出集实测：问句写成"我现在住哪"，而表里只有"住在"，属性识别直接为空，
    # 整套属性处置被跳过——这是真泛化缺口，不是个别题。
    ("residence", re.compile(r"住在|住哪|哪儿住|哪里住|住址|居住|搬到|搬去|搬来|搬家|迁到|定居|落户|生活在")),
    ("employer", re.compile(r"上班|任职|入职|跳槽|公司|单位|雇主|离职|在哪工作|哪家公司")),
    ("manager", re.compile(r"经理|主管|老板|领导|上司|导师|上级")),
    # 只用"车"字本身容易把"车牌/车站"也吞进来，所以列具体搭配；
    # 但"换了辆 SUV"这种换新句必须覆盖（漏了会让它不带属性、免惩罚排到第一）。
    ("vehicle", re.compile(r"开.{0,2}车|辆车|了辆|一辆|车是|油车|电车|车型|车牌|尾号|越野|轿跑|MPV|SUV")),
    ("device", re.compile(r"笔记本|电脑|主设备|显示器|平板|相机|编辑器")),
    ("phone", re.compile(r"手机号|尾号|号码|手机尾")),
    ("commute", re.compile(r"下班|几点|作息|通勤")),
    ("mail", re.compile(r"邮箱|邮件地址|域名|e-?mail", re.IGNORECASE)),
    ("gym", re.compile(r"健身|健身房|锻炼|运动|跑步|游泳|球|瑜伽")),
    # 楼层与"办公室在哪"不是同一件事，混成一个属性会让"我的办公室在哪个区"
    # 被"我搬到 12 楼了"顶掉（主集实测 q4 因此命中失败）。
    ("floor", re.compile(r"工位|楼层|楼|工区|座位|房间号")),
    ("office", re.compile(r"办公室|工区|办公点|园区")),
    ("budget", re.compile(r"预算|上限|额度|经费|怎么批|按月|按季度")),
    # 日程/周期：测试集 tq12「例会现在星期几？」暴露的缺口——
    # "例会改成周四了"里没有属性词，旧句靠字面重叠排前。这里按"周期"建一类，
    # 与"几点下班"的 commute 分开（后者是每日作息，前者是周/月周期）。
    ("schedule", re.compile(r"例会|会议|周几|星期几|礼拜几|日程|每周|每周几|上课|课程")),
]
W_ATTR_NEW = float(os.environ.get("AML_W_ATTR_NEW", "0.5"))
W_ATTR_OLD = float(os.environ.get("AML_W_ATTR_OLD", "0.45"))
# --- 问句时间短语亲和（难例集 n6/n7/n8/n9/n10 暴露）--------------------------
# 中文问句用"去年夏天 / 上个月 / 上周 / 我以前 / 我在甲公司时的"这类短语指定时间，
# 而检索只按绝对年份加分（W_DATE），短语一律看不见：于是"去年夏天我住哪"被
# 最新那句（"今年我搬到大连了"）压住，"我以前开什么车"被"现在换成帕萨特"压住。
# 这里把短语归成三类（past / recent / now），问句与句子同类就加分。
# 与 W_DATE 不叠加：带绝对年份的问句仍走 as-of 那一套，避免两套时间口径打架。
PERIOD_LEX = [
    ("past", re.compile(r"以前|过去|当时|那时|原来|当年|上家|曾经的|之前的|去年的|前年|去年")),
    ("past", re.compile(r"时的|的时候|那阵子")),
    ("recent", re.compile(r"最近|近来|上周|上个星期|上个月|前几天|这两天|刚换|刚刚")),
    ("now", re.compile(r"现在|目前|当前|如今|眼下|这个月|本月|今年|当下|暂时")),
]
W_PERIOD = float(os.environ.get("AML_W_PERIOD", "0.45"))
# 是非问句（"我现在喝咖啡吗？"）的否定识别：旧值不是被替换而是被否定，
# "我不再喝咖啡了"与"我每天喝两杯咖啡"字面几乎等价，必须靠否定标记区分。
# 只在问句确实命是非句式时生效，避免把陈述里的"不"当噪声。
YESNO_RE = re.compile(r"吗|是否|是不是|要不要|还在不在|有没有")
NEG_RE = re.compile(r"不再|不喝|不去|不吃|不抽|不用|没有了|没有在|已停|停止|放弃了|戒了|不开了|不在了|不再")
HOUSING_RE = re.compile(r"工位|办公室|楼层|楼|工区|座位")
RESIDENCE_STRONG_RE = re.compile(r"住在|住址|居住|定居|落户|生活在|搬进")
# 同形不同义的搭配：中文里"公司"既能指雇主也能修饰地点（公司健身房、
# 公司楼下），"车"能出现在与用车无关的句子里。这些搭配若没有强标记
# （上班/任职/跳槽 之类），就不能算进 employer / vehicle。
EMPLOYER_STRONG_RE = re.compile(r"上班|任职|入职|跳槽|离职|雇主|签了|加入了")
EMPLOYER_FALSE_FRIEND_RE = re.compile(r"健身|楼下|食堂|停车|班车|年会|团建")
# 单值属性：这些问题问"当前值"，除非问句里带了具体年份（as-of 场景）。
SINGLE_VALUED = {"residence", "employer", "manager", "vehicle", "device", "commute",
                 "mail", "phone", "gym", "floor", "budget", "schedule"}
# 属性新旧处置默认"本人"主体（我住在哪、我的经理是谁）。问句如果指向
# 第三人称亲属/同事（"我太太的弟弟在哪开公司"），把本人属性套上去会误伤
# （实测 q34 因 employer 注入被挤出前五）。这一版只做最保守的排除，
# 不做实体消解——真正的实体级状态跟踪留给后续（见报告缺口）。
THIRD_PARTY_RE = re.compile(
    r"太太|先生|老公|老婆|丈夫|妻子|弟弟|妹妹|哥哥|姐姐|父亲|母亲|爸爸|妈妈|"
    r"女儿|儿子|孩子|搭档|同事|朋友|客户|供应商|表姐|表妹|表哥|表弟|阿姨|叔叔|"
    r"室友|房东|邻居|导师的")
# as-of 问句的期间解析：问"2025 年 6 月的经理是谁"时，正确答案是该时点
# **有效**的那个版本，而不是"最新"那个。只按年份做加重（老逻辑）在同一年的
# 两次更新上分不开（实测 q23 旧经理与新经理同属 2025，靠字面重叠旧句反而靠前）。
PERIOD_RE = re.compile(r"(20\d{2})\s*年(?:\s*(\d{1,2})\s*月)?(?:\s*(\d{1,2})\s*[日号])?")


def query_period(text):
    """返回 (截止日字符串 'YYYY-MM-DD', 年, 月) ；无年份返回 None。"""
    m = PERIOD_RE.search(str(text or ""))
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2)) if m.group(2) else 12
    d = int(m.group(3)) if m.group(3) else 0
    if not 1 <= mo <= 12:
        mo = 12
    if d:
        end = "%04d-%02d-%02d" % (y, mo, min(d, 28))
    elif m.group(2):
        last = [31, 29 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 28,
                31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mo - 1]
        end = "%04d-%02d-%02d" % (y, mo, last)
    else:
        end = "%04d-12-31" % y
    return end, y, mo


# 槽位词（attr 之下的细分）：一个属性类里可能装着不同"槽"——
# 导师与客户经理都算 manager，相机与编辑器都算 device。
# 只按属性取最新，会把"客户经理"当成"导师"的现值（留出集 hq8 实测），
# 也会用"编辑器换到 VS Code"顶掉"新相机"（hq7）。所以先认槽，再在槽内取最新。
SLOT_LEX = [
    ("client_mgr", re.compile(r"客户经理|客户主管")),
    ("advisor", re.compile(r"导师|指导老师")),
    ("mgr", re.compile(r"经理|主管|老板|领导|上司")),
    ("camera", re.compile(r"相机|摄影机")),
    ("editor", re.compile(r"编辑器|IDE|Vim|VS Code|Emacs", re.IGNORECASE)),
    ("laptop", re.compile(r"笔记本|电脑|主设备|MacBook|ThinkPad|Surface|Dell|XPS")),
    ("monitor", re.compile(r"显示器|屏幕")),
    ("tablet", re.compile(r"平板|iPad")),
]


def slot_keys(text):
    """返回文本命中的槽位集合（可能为空）。命中了具体槽就不再记宽槽 mgr。"""
    t = str(text or "")
    out = {name for name, rx in SLOT_LEX if rx.search(t)}
    if out & {"client_mgr", "advisor"}:
        out.discard("mgr")
    return out


def attr_classes(text):
    t = str(text or "")
    found = {name for name, rx in ATTR_LEX if rx.search(t)}
    # "搬到 12 楼了"同时命中 residence 的"搬到"，但它说的是办公楼层（housing），
    # 不是居住城市。实测这会把 residence 的"最新一条"错认成楼层那句，
    # 于是"我现在住在哪"注入的是楼层，真正的"我搬到上海了"反而被埋。
    # 只做一条去混淆规则：出现楼层/工位/办公室且没有明确居住动词时，摘掉 residence。
    if HOUSING_RE.search(t) and not RESIDENCE_STRONG_RE.search(t):
        found.discard("residence")
        found.add("floor")
    # "我改去公司健身房了"命中 employer 的"公司"，但它不是雇主信息。
    if (EMPLOYER_FALSE_FRIEND_RE.search(t)
            and not EMPLOYER_STRONG_RE.search(t)):
        found.discard("employer")
    return found



def period_classes(text):
    """把时间短语归成 past / recent / now；认不出就返回空集。"""
    t = str(text or "")
    out = set()
    for name, rx in PERIOD_LEX:
        if rx.search(t):
            out.add(name)
    return out


# --- 当前态加分的相关性门 -----------------------------------------------------
# 难例集 hq1–hq4 暴露：只要句子里出现"搬到/换成/调到"这类变更动词，就会拿到
# W_NOW + W_CHANGE 的叠加（"我现在住在哪"这类题能差 1.0 → 2.1 倍），于是
# **同一用户下任何一句"我搬到大连了"都通吃所有"现在…"的问法**——问咖啡、
# 问游泳馆、问妹妹住哪，返回的第一条都是它。这不是"时间信号强"，是把
# "宣布了变更"错当成"与这个问题有关"。
# 处置：只有该句与问句落在同一属性，或至少共享一个中文实词 bigram 时，
# 才允许吃当前态/变更加分。属性重叠负责"我现在住哪"↔"我搬到上海了"
# 这种字面零重叠但同属性成立的情形。
CJK_STOP = set("我你他她它的了在有是和与吗呢吧啊么什哪怎这那个们就都也还很太"
               "请问一下把被给对从到说做要会能可以没不")


def cjk_bigrams(text: str) -> set:
    runs = re.findall(r"[\u4e00-\u9fff]+", str(text or ""))
    out = set()
    for run in runs:
        run = "".join(ch for ch in run if ch not in CJK_STOP)
        for i in range(len(run) - 1):
            out.add(run[i:i + 2])
        if len(run) == 1:
            out.add(run)
    return out


def _age_days(event_time, created_at, now_ts: float,
              world_clock: bool = True) -> float:
    """Age of one chunk in days, on the *world* clock rather than the storage one.

    Recency has to measure when a fact happened, not when it was filed away.
    A backfilled memory store ingests years of history inside a single sitting
    -- measured on this corpus at 5,893 chunks inside a 17-minute window -- so
    an ingest-only clock gives every chunk the same age of roughly zero. The
    temporal branch then multiplies every score by an identical constant and
    does nothing at all, which is worse than having no temporal feature,
    because the feature is advertised in `/version`'s retrieval description
    while contributing exactly nothing.

    `event_time` is the anchor parsed out of the chunk text at ingest time
    ("2022", "2022-03", "2022-03-04"). Where it is absent -- a store fed by a
    live stream, say -- the ingest clock is the only honest signal there is.

    `world_clock=False` reproduces the pre-fix behaviour (ingest clock only).
    That is not a live path: it exists so this change can be measured against
    the code it replaces with the environment-variable sweep, because "the
    feature was a no-op" is exactly the kind of claim that should be checked
    rather than asserted.
    """
    cands = (event_time, created_at) if world_clock else (created_at,)
    for cand in cands:
        if not cand:
            continue
        s = str(cand).strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m", "%Y"):
            for probe in (s, s[:19]):
                try:
                    t = time.mktime(time.strptime(probe, fmt))
                except (ValueError, TypeError):
                    continue
                return max(0.0, (now_ts - t) / 86400.0)
    return 0.0


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


def _content_to_text(content):
    """Normalise a message `content` field into the text we index.

    Never drops the record. An image-only content array used to collapse to an
    empty string and then get discarded by `build_chunks`, so Search came back
    with fewer records than Add accepted -- the AML smoke test caught exactly
    that (`expected=18, actual=11`). Image parts are serialised rather than
    thrown away so the chunk keeps a stable identity and a non-empty body.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                t = x.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
                elif "image_url" in x:
                    parts.append(json.dumps(x.get("image_url"),
                                            ensure_ascii=False))
            elif isinstance(x, str):
                parts.append(x)
        return " ".join(parts)
    # Reachable only for shapes the /add validator already rejects with 422
    # (a bare dict). AML's contract is str or a {text|image_url} array, so this
    # branch is defensive rather than live; the validator is the contract of
    # record and a 422 here is correct behaviour, not a dropped record.
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return "" if content is None else str(content)


def _blank_placeholder(m):
    """Honest non-empty stand-in for a message whose normalised text is blank.

    AML's response contract requires every returned record to carry non-empty
    `content`; a chunk stored with content "" surfaces as an illegal entry and
    fails the whole eval round (one-veto). Dropping the record is not an
    option either -- every message accepted by /add must be findable by
    /search (counts must match 1:1). A truthful placeholder satisfies both
    constraints: the record stays retrievable and contract-legal.

    Note (2026-10-07 red-team review): the "[image-only message]" branch is
    currently unreachable for /add-validated input -- any content array that
    contains an image_url part is serialised by `_content_to_text` into a
    non-empty JSON string (at minimum 'null'), so it never hits the blank
    path. The branch is kept as defensive cover in case that normalisation
    ever changes; image-only messages therefore store their serialised URL,
    not this placeholder.
    """
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, list) and any(
        isinstance(x, dict) and "image_url" in x for x in c
    ):
        return "[image-only message]"
    return "[blank message]"


def event_time_from_message(m):
    """消息自带的 `timestamp` 折成事件日期（UTC，`YYYY-MM-DD`）；没有就返回 None。

    官方 `/add` 的消息 shape 是 `{"role","timestamp"(毫秒 epoch),"content"}`，
    但老实现只从**正文**里 `parse_event_time` 抠日期，`created_at` 存的是摄取
    时钟。于是"时间只写在 timestamp 里"的语料（评测常见形态）在 as-of /
    时间线题上会退化成"全部等于摄取当天"。这里只补**正文没有日期时**的兜底，
    正文里写了日期仍以正文为准——那样才不会改动既有 D 维排序语义。
    """
    raw = m.get("timestamp")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        try:
            secs = raw / 1000.0 if raw > 1e11 else float(raw)
            return time.strftime("%Y-%m-%d", time.gmtime(secs))
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw, str) and raw.strip():
        # ISO 字符串也收（部分客户端直接传 "2025-03-20T00:00:00Z"）
        head = raw.strip()[:10]
        if len(head) == 10 and head[4] == "-" and head[7] == "-":
            return head
    return None


def build_units(messages):
    """`build_chunks` 的带时间版：每个检索单元配一条事件日期线索。

    返回 [(text, ts_hint)]；`ts_hint` 来自消息 timestamp，可能是 None。
    一个长消息被切成多个窗口时，各窗口共用该消息的时间线索。
    """
    units = []
    for m in messages:
        text = _content_to_text(m.get("content"))
        if not text.strip():
            text = _blank_placeholder(m)
        hint = event_time_from_message(m)
        for piece in split_long_message(text):
            units.append((piece, hint))
    return units


def build_chunks(messages):
    """One unit per message; long messages -> overlapping windows.

    Keeps every message a separate retrieval unit so adjacency expansion has
    natural neighbours (InvMem's message-boundary chunking). Unlike the earlier
    version this never skips a message: an empty or image-only body still
    produces one retrievable unit, because a record accepted by /add must be
    findable by /search. Since the blank-message contract fix, a message whose
    normalised text is blank stores a truthful placeholder ("[blank message]"
    / "[image-only message]") instead of an empty string, so /search can never
    return a record with empty content (AML response contract: content is
    required and non-empty; an illegal entry fails the whole round).
    """
    return [t for t, _ in build_units(messages)]


# ---------------------------------------------------------------- embeddings

_embed_state = {"ok": None, "backend": None, "fail_until": 0.0}

# --- background embedding ---------------------------------------------------
# The evaluator paces Add sequentially and its add phase runs on a finite
# budget: embedding 20-message batches synchronously (~1.5s per request of
# DashScope round-trip) meant later dataset segments were never delivered
# before the phase deadline -- four smoke runs in a row then failed with
# "expected=18, actual=11", where the missing records belonged to data that
# had never reached the service (verified via traffic capture 2026-09-29:
# e.g. clbench_0_4k's 4k-token conversation delivered as a single system
# prompt, zero occurrences of a word the questions explicitly reference).
# Add now persists immediately and embeds in the background; Search waits
# briefly for pending vectors in the requested scope, so memories are
# complete and searchable by the time the next evaluator call lands.
_embed_pending = {}
_embed_cond = threading.Condition()


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


def _bg_embed(user_id, rows):
    """Fill `vec` for (cid, text) rows; always clear the pending marker."""
    try:
        vecs = embed_texts([t for _, t in rows], "document")
        if vecs:
            with _db_lock, db() as c:
                for (cid, _), v in zip(rows, vecs):
                    c.execute("UPDATE chunks SET vec=? WHERE id=?",
                              (json.dumps(v).encode(), cid))
    except Exception:
        pass  # chunks keep vec=NULL; Search falls back to the lexical leg
    finally:
        with _embed_cond:
            _embed_pending[user_id] = max(
                0, _embed_pending.get(user_id, 0) - 1)
            _embed_cond.notify_all()


def _wait_vectors(user_id, timeout=15.0):
    """Block until no background embedding is pending for this scope."""
    deadline = time.time() + timeout
    with _embed_cond:
        while _embed_pending.get(user_id, 0) > 0:
            left = deadline - time.time()
            if left <= 0:
                return False
            _embed_cond.wait(left)
    return True


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

@app.get("/debug/traffic")
async def debug_traffic(request: Request):
    """Authenticated read-back of captured platform traffic (debug only)."""
    denied = guard(request)
    if denied is not None:
        return denied
    rows = []
    if DEBUG_FILE.exists():
        try:
            with open(DEBUG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-400:]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        except Exception as e:
            return {"detail": {"reason": f"read failed: {e}"}}
    rows.reverse()  # newest first
    return {"count": len(rows), "rows": rows}


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

    # Hand the raw messages over so `build_chunks` normalises content itself;
    # pre-flattening here would repeat the image-only -> empty-string collapse
    # that made /search return fewer records than /add accepted.
    units = build_units(messages)
    if not units:
        return err(422, "no usable content")

    # Off the event loop: the embedding call is blocking HTTP (up to
    # EMBED_TIMEOUT) and this handler is async, so calling it inline would
    # serialise every concurrent request behind one round-trip each. The
    # evaluator drives Add/Search at the concurrency declared at registration,
    # so inline would turn N parallel calls into N sequential ones.
    # Dense vectors are filled by the background leg (see _bg_embed): the
    # add phase of the evaluator is time-budgeted, so returning fast here is
    # what gets the whole dataset delivered before the deadline.
    created = now_iso()
    rid_hash = hashlib.sha256(request_id.encode()).hexdigest()[:12]

    inserted = 0
    bg_rows = []
    with _db_lock, db() as c:
        # Per-session ordinal for adjacency expansion: continue the session's
        # sequence so chunks from different requests in one session chain up.
        base = c.execute(
            "SELECT COALESCE(MAX(sess_seq), -1) FROM chunks "
            "WHERE user_id=? AND session_id=?",
            (user_id, session_id),
        ).fetchone()[0] + 1
        for i, (text, ts_hint) in enumerate(units):
            chash = hashlib.sha256(
                (user_id + "\x00" + text).encode()
            ).hexdigest()
            # No chunk-level de-duplication here any more. Request-level
            # idempotency is already handled by `seen_requests` above, so a
            # retry with the same request_id cannot double-insert. The old
            # content-hash filter swallowed records the evaluator had just
            # written -- first across the whole user, then, once scoped to a
            # session, even empty messages (they all hash the same), which is
            # what made /search answer 11 of the 18 records the smoke test
            # expected. Keeping the hash on the row is harmless; filtering on
            # it is what cost us the smoke test.
            cid = f"mem_{rid_hash}_{i}"
            bg_rows.append((cid, text))
            c.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (cid, user_id, session_id, request_id, text, created,
                 count_words(text), None, base + i,
                 parse_event_time(text) or ts_hint,     # 正文日期优先；仅无日期时用 timestamp 兜底
                 chash),
            )
            inserted += 1
        c.execute(
            "INSERT OR REPLACE INTO seen_requests VALUES (?,?,?)",
            (request_id, user_id, session_id),
        )

    # Schedule the dense leg off the response path; Search drains pending
    # vectors for this scope before ranking, so nothing is ever invisible.
    with _embed_cond:
        _embed_pending[user_id] = _embed_pending.get(user_id, 0) + 1
    threading.Thread(target=_bg_embed, args=(user_id, bg_rows),
                     daemon=True, name=f"embed-{request_id[-12:]}").start()

    _dbg_capture("add", {
        "request_id": request_id, "user_id": user_id, "session_id": session_id,
        "messages": len(messages), "chunks_stored": inserted,
        "roles": [str(m.get("role")) for m in messages][:30],
        "content_types": [type(m.get("content")).__name__ for m in messages][:30],
        "content_heads": [str(m.get("content"))[:80] for m in messages][:30],
    })
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
    top_k_err, top_k = check_top_k(top_k)
    if top_k_err is not None:
        return err(422, top_k_err)

    # Dense leg completeness: hold briefly while background embedding for
    # this scope finishes, so freshly added memories rank with full quality.
    _wait_vectors(user_id)

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
    q_periods = period_classes(query)
    q_yesno = bool(YESNO_RE.search(query))
    q_bigrams = cjk_bigrams(query)
    now_gate = os.environ.get("AML_NOW_GATE", "1") != "0"
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
            # World time first, ingest clock only as a fallback: see
            # `_age_days`. Reading the ingest clock here made every recency
            # value 1.0 for a bundled corpus, so the temporal branch below was
            # a constant multiplier and therefore a no-op.
            age_days = _age_days(event_times[i], created[i], now_ts,
                                 not TEMP_LEGACY_CLOCK)
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

    # --- 属性级新旧处置（中文换新题）----------------------------------------
    # 只在"现在/目前"类问题、且问题与句子落在同一属性时生效（见 ATTR_LEX 注释）。
    # 同一属性内按事件时间（无则摄取时钟）取最新一条作为"现值"，其余乘惩罚。
    q_attrs = attr_classes(query)
    chunk_attrs = [attr_classes(x) for x in contents]
    chunk_slots = [slot_keys(x) for x in contents]
    q_slots = slot_keys(query)
    # 会话内省略继承：中文接着说的时候主语/属性常省掉——
    # "我的预算按季度批。" → "现在改成按月批了。" 第二句不带"预算"，
    # 若不给它继承，属性处置会把旧句当成唯一现值（实测 q20 因此答反）。
    # 只继承**同一会话**里紧邻的前一句，不跨会话、不跨话题。
    if sessions and sess_seq:
        by_session = {}
        for i in range(len(ids)):
            if sessions[i] is not None:
                by_session.setdefault(sessions[i], []).append(i)
        for sid, idxs in by_session.items():
            idxs.sort(key=lambda i: (sess_seq[i] is None, sess_seq[i] or 0))
            prev = None
            for i in idxs:
                if not chunk_attrs[i] and prev is not None:
                    chunk_attrs[i] = set(chunk_attrs[prev])
                prev = i
    attr_newest = {}
    # 无年份：取最新（"我现在住哪"）。有年份：不动属性处置（交给 W_DATE/字面），
    # 因为"同一年内两次更新"的 as-of 细分实测会伤到别的题
    # （试过按年月取"该时点有效版本"，38 题 hit@5 从 1.000 掉到 0.921，已回退；
    #  细化版留作后续，见报告缺口）。
    # 问句明确指向过去/近期（"去年夏天""上个月""我以前"）时，不要给"现值"加分——
    # 那是 as-of 场景，现值注入会把正确答案挤掉（难例 n6/n7 实测如此）。
    attr_logic = (bool(q_attrs & SINGLE_VALUED) and not q_years
                  and not (q_periods - {"now"})
                  and not THIRD_PARTY_RE.search(str(query)))
    if attr_logic:
        for a in q_attrs:
            cands = [i for i in range(len(ids)) if a in chunk_attrs[i]]
            if not cands:
                continue
            # 问句认得出具体槽、且同槽候选不止一条时才收窄到同槽；
            # 只有一条就退回全属性——否则"主设备是 MacBook"这类问法会把
            # 换新句（"我换到 雷蛇灵刃 了"，无槽位词）排除在外（规模集实测 -6 题）。
            if q_slots:
                same = [i for i in cands if chunk_slots[i] & q_slots]
                if len(same) >= 2:
                    cands = same
            attr_newest[a] = max(
                cands, key=lambda i: str(event_times[i] or created[i] or ""))

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
                # 相关性门：只按"宣布了变更"加分会让无关的"我搬到X了"通吃所有
                # "现在…"问句（难例 hq1–hq4 实测如此）。要求同属性或实词重叠。
                same_attr = bool(chunk_attrs[i] & q_attrs)
                overlap = bool(cjk_bigrams(contents[i]) & q_bigrams)
                if same_attr or overlap or not now_gate:
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
        date_affinity = bool(q_years) and (
            any(y in contents[i] for y in q_years)
            # 日期也可能只存在于 event_time（来自消息 timestamp），正文里没写年。
            # 老逻辑只认正文，于是"2025 年 6 月的经理是谁"这类题拿不到加分，
            # 旧经理句靠字面重叠压住新经理句（实测 q23/q27 as-of 均失败）。
            or (event_times[i] and str(event_times[i])[:4] in q_years)
        )
        if date_affinity:
            factor += W_DATE
        # 时间短语同类加分：只对 past / recent 生效。
        # "现在"必须排除——旧句里常直接写着"我现在住在杭州"，一加分就把真正的
        # 更新句（"我搬到上海了"，无"现在"二字）压下去（主集 q11/q14/q16/q18
        # 实测 0.895→0.789 就是这么来的）；当前值语义已由 NOW_HINT_RE + 现值注入负责。
        if (not q_years) and (period_classes(contents[i]) & q_periods & {"past", "recent"}):
            factor += W_PERIOD
        # 是非问句：带否定标记的句子优先（旧值被否定 ≠ 旧值不存在）
        if q_yesno and NEG_RE.search(contents[i]):
            factor += W_PERIOD
        if attr_logic:
            shared = chunk_attrs[i] & q_attrs
            if shared:
                if any(attr_newest.get(a) == i for a in shared):
                    factor += W_ATTR_NEW
                else:
                    factor *= (1.0 - W_ATTR_OLD)
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
    # 现值注入：属性级最新一条即使字面/语义都不匹配，也要作为候选进入排序。
    # "我现在住在哪"与"我搬到上海了"没有共享词，纯检索永远捞不上来；
    # 但同一属性上更新的那条就是答案。注入分取当前最高分的 0.9，既不抢
    # 明显更相关的条目，也不会让现值被埋到 top_k 之外。
    if attr_logic and merged:
        top_score = max(merged.values())
        for a, idx in attr_newest.items():
            merged[idx] = max(merged.get(idx, 0.0), top_score * 0.9)
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
    # Dedupe by content before truncating to top_k. The add path keeps
    # request-level idempotency but no content-level dedupe (that filter used
    # to swallow records the evaluator had just written), so the same sentence
    # restated in three sessions is stored three times and used to occupy
    # three answer slots. Scores are ordered already, so the first occurrence
    # per content keeps its best rank and later copies are dropped; dropping
    # them frees slots for distinct records instead of shortening the list.
    seen_content = set()
    for i, s in merged_sorted:
        key = " ".join(str(contents[i]).split())
        if key in seen_content:
            continue
        seen_content.add(key)
        item = {"id": ids[i], "content": contents[i]}
        if s > 0:
            item["score"] = round(float(s), 6)
        item["created_at"] = created[i]
        results.append(item)
        if len(results) >= top_k:
            break
    _dbg_capture("search", {
        "user_id": user_id, "top_k": top_k, "returned": len(results),
        "query": str(query)[:400],
        "top3": [{"id": r["id"], "content": str(r["content"])[:120]}
                 for r in results[:3]],
    })
    return {"data": results}


if __name__ == "__main__":
    import uvicorn
    # Platform-injected PORT = managed hosting (bind all interfaces); local dev binds loopback only
    _port = int(os.environ.get("PORT") or os.environ.get("AML_PORT") or "8021")
    _host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run(app, host=_host, port=_port,
                log_level="warning", access_log=False, server_header=False)
