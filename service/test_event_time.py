"""本地可复跑：事件时间该从哪儿取。

合同消息 shape 是 `{"role","timestamp"(ms epoch),"content"}`。老实现只从正文
里 parse_event_time 抠日期，结果"时间只写在 timestamp 里"的语料在 as-of /
时间线题上退化为"全部等于摄取日"。

两条断言：
1. 正文**没有**日期 + timestamp=2025-03-20 → 落库 event_time 必须是 2025-03-20；
2. 正文**写了**日期 2025-03-20 + timestamp 指向别的日子 → event_time 仍以正文为准
   （不许改动既有 D 维排序语义）。
"""
import json
import os
import sqlite3
import sys
import tempfile

DB = os.path.join(tempfile.mkdtemp(), "evt.db")
os.environ["AML_DB_PATH"] = DB
os.environ.setdefault("AML_API_KEY", "test-key")
os.environ.pop("DASHSCOPE_API_KEY", None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi.testclient import TestClient  # noqa: E402
import app as svc  # noqa: E402

H = {"Authorization": "Bearer test-key"}
c = TestClient(svc.app)
fails = []


def add(rid, sid, text, ts_ms):
    r = c.post("/add", headers=H, json={
        "request_id": rid, "user_id": "u_evt", "session_id": sid,
        "messages": [{"role": "user", "timestamp": ts_ms, "content": text}]})
    assert r.status_code == 200, (r.status_code, r.text[:200])
    return r.json()


def evt_of(rid):
    con = sqlite3.connect(DB)
    try:
        row = con.execute(
            "SELECT event_time FROM chunks WHERE request_id=?", (rid,)
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else "<missing>"


# 1) 只有 timestamp：2025-03-20 00:00:00 UTC = 1742428800000
add("evt-ts-only", "s1", "我搬到了杭州。", 1742428800000)
got = evt_of("evt-ts-only")
ok1 = got == "2025-03-20"
print("仅 timestamp 供时 -> event_time=%r 期望 '2025-03-20'：%s" % (got, "PASS" if ok1 else "FAIL"))
if not ok1:
    fails.append("timestamp 未参与事件时间")

# 2) 正文带日期 + timestamp 指向另一天：正文优先
add("evt-text-wins", "s2", "2025-03-20 我搬到了杭州。", 1743465600000)  # 2025-04-01
got = evt_of("evt-text-wins")
ok2 = got == "2025-03-20"
print("正文有日期优先 -> event_time=%r 期望 '2025-03-20'：%s" % (got, "PASS" if ok2 else "FAIL"))
if not ok2:
    fails.append("正文日期被 timestamp 覆盖（回归）")

# 3) 都没有：不崩，event_time 为空不外溢
add("evt-none", "s3", "随便说一句没有日期的话。", None)
got = evt_of("evt-none")
ok3 = got is None or got == "" or got == "<missing>"
print("无时间线索 -> event_time=%r 允许空：%s" % (got, "PASS" if ok3 else "FAIL"))
if not ok3:
    fails.append("无时间线索时写出了假 event_time")

print("\n结果：%d 项失败 %s" % (len(fails), fails or "（全通过）"))
sys.exit(1 if fails else 0)

