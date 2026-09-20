FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AML_DB_PATH=/data/memory.db

WORKDIR /app

COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY service/ ./

EXPOSE 8000

# 单进程即可：SQLite + 每请求一次托管嵌入调用，评测期流量很低。
# （嵌入走 text-embedding-v4，服务内部有 90 秒冷却与 BM25 降级；不装 torch/onnx。）
# 尊重平台注入的 PORT（Render / 多数 PaaS 会注入），未注入时回落 8000。
# 注意：PORT 存在即代表托管环境，app.py 会要求必须显式配置 AML_API_KEY 才启动。
#
# --forwarded-allow-ips=* ：必须。Render 的代理从内网地址（10.x）连进来，
# 而 uvicorn 的 forwarded_allow_ips 默认只信 127.0.0.1 ⇒ 不信任 X-Forwarded-For
# ⇒ request.client.host 只会是那个内网代理地址。后果有二：
#   ① 按 IP 限流退化成「全网共用一个桶」，任一调用方的突发会让**别人**吃 429；
#   ② 日志里所有访客 IP 都变成同一个 10.x，出问题无法定位来源。
# 服务只经平台代理暴露，故信任其转发头是安全的。
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
