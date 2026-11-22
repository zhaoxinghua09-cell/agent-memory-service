"""契约边界：top_k 上限。

平台合同（api-guide，2026-11-21 复核）规定评测固定 top_k=100，且服务端**不得超量返回**。
线上 0.7.1 的行为是 `min(top_k, 200)` 静默夹取：
- 请求 500 时返回 200 且照常检索（本次外人实跑探针实测）；
- 上限写成 200，比合同允许的 100 宽一倍，等于默许客户端越界。

本模块把"越界必须显式拒绝"做成纯函数，便于在没有 fastapi 的环境里直接单测。
"""
CONTRACT_MAX_TOP_K = 100


def check_top_k(top_k, max_top_k=CONTRACT_MAX_TOP_K):
    """返回 (error_message, value)。

    - 非 int / <=0  → ("top_k required", None)
    - > 上限        → ("top_k exceeds contract maximum %d" % max_top_k, None)
    - 合法          → (None, top_k)
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        return "top_k required", None
    if top_k > max_top_k:
        return "top_k exceeds contract maximum %d" % max_top_k, None
    return None, top_k
