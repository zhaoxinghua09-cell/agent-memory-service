"""本地可复跑：第三人称问题的代词跨句指代（主体消解）。

背景（难例集 hq3/hq4 暴露）：问"我妹妹现在住哪"，答案在"她后来搬去珠海了"里；
问"我的经理现在在哪办公"，答案是"他调去成都了"。这两句只带代词，主体靠上文承接。
属性处置原来是"本人单值"模型，本人的"今年我搬到大连了"被认成同属性最新，
把她/他的更新句压下去（hq3/hq4 全败）。

守三件事：
A. 正例（亲属）：问句指向妹妹时，代词句"她后来搬去珠海了"必须排第一；
B. 正例（职位）：问句指向经理时，代词句"他调去成都了"必须排第一；
C. 反例（不能过冲）：问本人的现住地时，第三人的住址不得抢答——
   门只对第三人称问题收紧，不能把本人的现值注入一起掐掉。
"""
import os
import sys
import tempfile

os.environ.setdefault("AML_DB", os.path.join(tempfile.mkdtemp(), "person.db"))
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
        "request_id": "p-%s-%s-%s" % (uid, sid, days),
        "user_id": uid, "session_id": sid,
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
    print("%-36s 期望含「%s」实际「%s」：%s" % (name, want, got[:24], "PASS" if ok else "FAIL"))
    if not ok:
        fails.append("%s（期望 %s 实得 %s）" % (name, want, got[:40]))


# A 亲属代词
add("u_sis", "s1", "我妹妹在珠海工作。", 0)
add("u_sis", "s1", "她后来搬去珠海了。", 200)
add("u_sis", "s1", "今年我搬到大连了。", 300)      # 本人的更新，且更新更近
check("A-亲属代词句须胜出", "u_sis", "我妹妹现在住哪？", "珠海")

# B 职位代词
add("u_mgr", "s1", "我的经理是王工，在上海办公室。", 0)
add("u_mgr", "s1", "他调去成都了。", 200)
add("u_mgr", "s1", "今年我搬到大连了。", 300)
check("B-职位代词句须胜出", "u_mgr", "我的经理现在在哪办公？", "成都")

# C 反例：问本人时，第三人的住址不得抢答
add("u_self", "s1", "我妹妹在珠海工作。", 0)
add("u_self", "s1", "她后来搬去珠海了。", 100)
add("u_self", "s1", "我搬到上海了。", 300)
check("C-问本人时第三人住址不得抢答", "u_self", "我现在住在哪？", "上海")

# D 反例：问第三人时，与这个人无关的新句不得抢答（hq4 实测现场）
add("u_mgr2", "s1", "我的经理是葛亮，他在深圳办公。", 0)
add("u_mgr2", "s1", "他调去成都了。", 200)
add("u_mgr2", "s1", "这个月预算是 40 万。", 300)   # 更新、更近，但与此人无关
check("D-无关新句不得抢答第三人问题", "u_mgr2", "我的经理现在在哪办公？", "成都")

# E 反例：反过来的过冲——问本人时，本人的现值注入不能被第三人逻辑掐掉
add("u_mix", "s1", "我妹妹住在佛山。", 0)
add("u_mix", "s1", "我搬到上海了。", 200)
check("E-本文主体现值注入仍生效", "u_mix", "我现在住在哪？", "上海")

print("\n结果：%d 项失败 %s" % (len(fails), fails or "（全通过）"))
sys.exit(1 if fails else 0)
