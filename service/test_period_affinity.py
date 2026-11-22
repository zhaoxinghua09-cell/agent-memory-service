"""本地可复跑：问句时间短语（去年夏天/上个月/以前）必须把对应那句排到第一。

背景：检索只按**绝对年份**加分（W_DATE），中文短语一律看不见。于是
"去年夏天我住哪"被最新那句（"今年我搬到大连了"）压住，"我以前开什么车"
被"现在换成帕萨特"压住——难例集这四类 hit@1 只有 0.100。

本测试守住两件事：
A. 短语同类匹配能生效（past / recent 各自命中对应句子）；
B. **"现在"不得参与该加分**。这是本轮踩到的坑：旧句常直接写着
   "我现在住在杭州"，一旦给"现在"加分，真正的更新句（"我搬到上海了"，
   不带"现在"二字）就被压下去——主集 q11/q14/q16/q18 实测 0.895→0.789。
   当前值语义由 NOW_HINT_RE + 现值注入负责，短语加分必须让路。
"""
import os
import sys
import tempfile

os.environ.setdefault("AML_DB", os.path.join(tempfile.mkdtemp(), "period.db"))
os.environ.setdefault("AML_API_KEY", "test-key")
os.environ.pop("DASHSCOPE_API_KEY", None)   # 无 key 走词法腿 + 时间信号，结果与稠密腿无关

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi.testclient import TestClient  # noqa: E402
import app as svc  # noqa: E402

H = {"Authorization": "Bearer test-key"}
c = TestClient(svc.app)
fails = []

BASE = 1735689600000          # 2025-01-01
DAY = 86400000


def add(uid, sid, text, days):
    r = c.post("/add", headers=H, json={
        "request_id": "p-%s-%s" % (uid, days), "user_id": uid, "session_id": sid,
        "messages": [{"role": "user", "timestamp": BASE + days * DAY, "content": text}]})
    assert r.status_code == 200, (r.status_code, r.text[:200])


def top1(uid, q):
    r = c.post("/search", headers=H, json={"user_id": uid, "query": q, "top_k": 10})
    assert r.status_code == 200, (r.status_code, r.text[:200])
    data = r.json()["data"]
    return (data[0]["content"] if data else "")


def check(name, uid, q, want, gap):
    got = top1(uid, q)
    ok = want in got
    print("%-28s 期望含「%s」实际「%s」：%s" % (name, want, got[:26], "PASS" if ok else "FAIL"))
    if not ok:
        fails.append("%s（期望 %s，实得 %s）" % (name, want, got[:40]))


# A1 过去短语
add("u_a1", "s1", "去年夏天我住在青岛。", 300)
add("u_a1", "s1", "今年我搬到大连了。", 800)
check("过去-去年夏天", "u_a1", "去年夏天我住哪？", "青岛", "大连")

# A2 近期短语
add("u_a2", "s1", "上个月我的预算还是 20 万。", 420)
add("u_a2", "s1", "这个月提到 40 万了。", 520)
check("近期-上个月预算", "u_a2", "上个月我的预算多少？", "20 万", "40 万")

# A3 过去短语 vs 现在
add("u_a3", "s1", "我以前开的是一辆老捷达。", 0)
add("u_a3", "s1", "现在换成帕萨特了。", 260)
check("过去-以前开什么车", "u_a3", "我以前开什么车？", "捷达", "帕萨特")

# B 反例：问"现在"必须仍取更新句，不得被旧句里的"现在"二字抬上来
add("u_b", "s1", "我现在住在杭州。", 0)
add("u_b", "s1", "我搬到上海了。", 200)
check("现在-不得被旧句压住", "u_b", "我现在住在哪？", "上海", "杭州")

print("\n结果：%d 项失败 %s" % (len(fails), fails or "（全通过）"))
sys.exit(1 if fails else 0)
