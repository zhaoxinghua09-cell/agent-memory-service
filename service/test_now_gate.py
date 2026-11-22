"""本地可复跑：当前态加分必须过相关性门。

背景（难例集 hq1–hq4 暴露的真 bug）：只要句子里出现"搬到/换成/调到"这类
变更动词，就会吃到 W_NOW + W_CHANGE 的叠加，**不检查是否与问句有关**。
后果：同一用户下任何一句"我搬到大连了"通吃所有"现在…"的问法——
问咖啡、问游泳馆、问妹妹住哪，第一条都是它。这不是时间信号强，是把
"宣布了变更"错当成"与这个问题有关"。

处置：只有该句与问句同属性（"我现在住哪" ↔ "我搬到上海了"，字面零重叠）
或至少共享一个中文实词 bigram 时，才允许吃当前态/变更加分。

本测试守两件事（缺一不可）：
A. 反例：无关的"我搬到上海了"不得压过真正回答问题的"我不再喝咖啡了"；
B. 正例：同属性的"我搬到上海了"仍必须能把旧的"我现在住在杭州"压下去
   ——门不能把有效的现值注入一起关掉（那是主集 0.737→0.921 的来源）。
"""
import os
import sys
import tempfile

os.environ.setdefault("AML_DB", os.path.join(tempfile.mkdtemp(), "gate.db"))
os.environ.setdefault("AML_API_KEY", "test-key")
os.environ.pop("DASHSCOPE_API_KEY", None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi.testclient import TestClient  # noqa: E402
import app as svc  # noqa: E402

H = {"Authorization": "Bearer test-key"}
c = TestClient(svc.app)
fails = []
BASE = 1735689600000
DAY = 86400000


def add(uid, sid, text, days):
    r = c.post("/add", headers=H, json={
        "request_id": "g-%s-%s" % (uid, days), "user_id": uid, "session_id": sid,
        "messages": [{"role": "user", "timestamp": BASE + days * DAY, "content": text}]})
    assert r.status_code == 200, (r.status_code, r.text[:200])


def top1(uid, q):
    r = c.post("/search", headers=H, json={"user_id": uid, "query": q, "top_k": 10})
    assert r.status_code == 200, (r.status_code, r.text[:200])
    d = r.json()["data"]
    return d[0]["content"] if d else ""


def check(name, uid, q, want):
    got = top1(uid, q)
    ok = want in got
    print("%-34s 期望含「%s」实际「%s」：%s" % (name, want, got[:24], "PASS" if ok else "FAIL"))
    if not ok:
        fails.append("%s（期望 %s 实得 %s）" % (name, want, got[:40]))


# A 反例：无关变更句不得抢答
add("u_gate", "s1", "我每天喝两杯咖啡。", 0)
add("u_gate", "s1", "我不再喝咖啡了。", 120)
add("u_gate", "s1", "我搬到上海了。", 200)          # 更新、且含变更动词，但与"咖啡"无关
check("反例-无关变更句不得抢答", "u_gate", "我现在喝咖啡吗？", "不再喝咖啡")

# B 正例：同属性更新必须仍然赢（门不能误杀现值注入）
add("u_gate2", "s1", "我现在住在杭州。", 0)
add("u_gate2", "s1", "我搬到上海了。", 200)
check("正例-同属性更新仍须胜出", "u_gate2", "我现在住在哪？", "上海")

print("\n结果：%d 项失败 %s" % (len(fails), fails or "（全通过）"))
sys.exit(1 if fails else 0)
