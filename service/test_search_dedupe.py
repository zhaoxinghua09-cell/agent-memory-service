"""本地可复跑：同文多 session 不应挤占 top_k 槽位。

背景：`/add` 只做 request 级幂等，不做内容级去重（当年跨 session content_hash
过滤吞掉了评测刚写进去的记录，才改成现在这样）。后果是同句分 3 个 session
投喂后 `/search` 返回 3 条同文，占掉 3 个 answer 前缀槽位。

本测试在本地起服务（TestClient，不需要外网、不需要 embedding key）：
1. 同一 user 下把同一句话写进 3 个 session；
2. `/search` 必须只返回 1 条该内容；
3. 再补 4 条不同内容，`top_k=3` 必须返回 3 条内容互不相同的记录（去重后仍填满槽位）。
"""
import os
import sys
import tempfile

os.environ.setdefault("AML_DB", os.path.join(tempfile.mkdtemp(), "dedupe.db"))
os.environ.setdefault("AML_API_KEY", "test-key")
os.environ.pop("DASHSCOPE_API_KEY", None)   # 无 key 时走词法腿，结果与稠密腿无关

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi.testclient import TestClient  # noqa: E402
import app as svc  # noqa: E402

H = {"Authorization": "Bearer test-key"}
c = TestClient(svc.app)
fails = []


def add(rid, uid, sid, text):
    r = c.post("/add", headers=H, json={
        "request_id": rid, "user_id": uid, "session_id": sid,
        "messages": [{"role": "user", "timestamp": 1735689600000, "content": text}]})
    assert r.status_code == 200, (r.status_code, r.text[:200])
    return r.json()


def search(uid, q, k):
    r = c.post("/search", headers=H, json={"user_id": uid, "query": q, "top_k": k})
    assert r.status_code == 200, (r.status_code, r.text[:200])
    return r.json()["data"]


SAME = "我现在住在杭州。"

# 1) 同文进 3 个 session
for i in range(3):
    add("dup-%d" % i, "u_dup", "s%d" % i, SAME)
d = search("u_dup", "我住在哪？", 10)
uniq = {" ".join(x["content"].split()) for x in d}
ok1 = len(d) == 1 and len(uniq) == 1
print("同文 3 session -> 返回条数 %d，去重后 %d 条：%s" % (len(d), len(uniq), "PASS" if ok1 else "FAIL"))
if not ok1:
    fails.append("同文 3 session 未去重")

# 2) 去重后仍能填满 top_k（5 条不同内容，top_k=3 应给 3 条互不相同）
for i, t in enumerate(["我的猫叫年糕。", "我在微软上班。", "我妹妹住在苏州。",
                       "我每天早上跑步五公里。", "经理的办公室在 12 楼。"]):
    add("dist-%d" % i, "u_dedup", "s1", t)
d = search("u_dedup", "我的猫叫什么？", 3)
uniq = {" ".join(x["content"].split()) for x in d}
ok2 = len(d) == 3 and len(uniq) == 3
print("top_k=3 -> 返回 %d 条、去重后 %d 条：%s" % (len(d), len(uniq), "PASS" if ok2 else "FAIL"))
if not ok2:
    fails.append("去重后未填满 top_k")

print("\n结果：%d 项失败 %s" % (len(fails), fails or "（全通过）"))
sys.exit(1 if fails else 0)
