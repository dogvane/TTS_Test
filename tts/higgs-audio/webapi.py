"""Higgs Audio v3 TTS webapi — Docker 模式下的健康检查代理。

Higgs Audio 通过 Docker 容器运行 SGLang-Omni 服务，网关不需要启动后端进程。
此 webapi 仅提供 health 端点供网关检测服务状态。
实际的 TTS 推理由 Docker 容器内的 SGLang-Omni 服务完成。
"""

import sys
import os
import argparse

import requests
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = FastAPI(title="Higgs Audio v3 TTS (Docker)")

# 加载配置
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DOCKER_URL = cfg["backend_url"].rstrip("/")
SAMPLE_RATE = cfg.get("sample_rate", 24000)


@app.get("/health")
def health():
    """检查 Docker 容器中 SGLang-Omni 服务的健康状态。"""
    try:
        resp = requests.get(f"{DOCKER_URL}/health", timeout=5)
        if resp.status_code == 200:
            return JSONResponse({
                "status": "ok",
                "model": "higgs-audio",
                "sample_rate": SAMPLE_RATE,
                "backend": "docker",
            })
    except requests.RequestException:
        pass
    try:
        resp = requests.get(f"{DOCKER_URL}/v1/models", timeout=5)
        if resp.status_code == 200:
            return JSONResponse({
                "status": "ok",
                "model": "higgs-audio",
                "sample_rate": SAMPLE_RATE,
                "backend": "docker",
            })
    except requests.RequestException:
        pass
    return JSONResponse({"status": "not_ready", "model": "higgs-audio"}, status_code=503)


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=cfg.get("webapi_port", 8001))
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
