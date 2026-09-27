"""Proof: a blank document no longer kills the whole rerank batch.

Established by probe (`_probe_rerank_detail.py` case B): the vendor answers
HTTP 400 `InvalidParameter: text input should not be empty` for the *entire*
request when any single document is blank. Since the nodrop fix requires /add
to keep empty messages, one such record was enough to disable reranking for
every search by that user -- silently, because the leg degrades to the fusion
order by design. That is the worst shape of failure: the service looks fine.

This also pins the index mapping. The caller feeds `_rerank_scores` a list and
reorders its own candidates with the returned positions, so returning indices
relative to the *filtered* list would silently shuffle the wrong chunks.
"""
import os
import subprocess
import sys
from pathlib import Path

PY = r"C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe"
GET = r"C:/Users/Administrator/.workbuddy/skills/dpapi-local-vault/get_secret.py"
SVC = Path(r"D:\Workbuddy\2026-09-20-19-28-08\aml-handoff\service")

p = subprocess.run([PY, GET, "dashscope_api_key", "--raw"],
                   capture_output=True, text=True, timeout=90)
os.environ["AML_DASHSCOPE_KEY"] = (p.stdout or "").strip()
os.environ["AML_DB_PATH"] = str(Path(os.environ.get("TEMP", ".")) / "aml_rerank_blank.db")
os.environ["AML_EMBED_TIMEOUT"] = "1.0"
sys.path.insert(0, str(SVC))

import app as A  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# Positions 1 and 3 are the blanks. Position 2 is the one a sane reranker
# must put first for this query.
docs = ["the widget is 1mm wide", "", "I prefer coffee without sugar",
        "   ", '{"url": "http://img/0.png"}']

print("用例：候选里掺 2 条空文档（位置 1、3）")
order = A._rerank_scores("coffee sugar", docs)
print(f"       返回 order = {order}")
check("返回非 None（不再整批 400）", order is not None, str(order))
if order is not None:
    check("空文档位置 1、3 未出现在结果里", 1 not in order and 3 not in order, str(order))
    check("下标仍按原坐标系（0..4，不是过滤后的 0..2）",
          all(0 <= x < len(docs) for x in order) and max(order) > 2,
          f"order={order} max={max(order)}")
    check("最相关那条（位置 2）排首位", order[0] == 2, f"order[0]={order[0]}")
check("rerank 腿状态 ok=True", A._rerank_state["ok"] is True, str(A._rerank_state))

print()
print("对照组：候选全空")
before = A._rerank_state["ok"]
order2 = A._rerank_scores("anything", ["", "   "])
check("全空时返回 None（无内容可排）", order2 is None, str(order2))
check("全空不污染 healthy（没发过请求，就不算腿故障）",
      A._rerank_state["ok"] == before, f"before={before} after={A._rerank_state['ok']}")

print()
print("RESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAIL -> {FAILS}")
sys.exit(0 if not FAILS else 1)
