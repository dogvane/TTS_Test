"""TTS 统一评测网关 — 兼容 OpenAI /v1/audio/speech 接口。

按需启动：收到请求时才拉起对应引擎的后端，网关本身启动不加载任何模型。

启动:
    cd /mnt/o/ai/TTS/TTS_Test
    python -m server.main
    python -m server.main --port 9000
"""

import importlib
import io
import base64
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np
import requests
import soundfile as sf
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from server.base_adapter import BaseAdapter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 后端进程管理（懒加载）
# ---------------------------------------------------------------------------
_engine_configs: dict[str, dict] = {}     # {engine_dir: config_dict}
_engine_conda_envs: dict[str, str] = {}   # {engine_dir: conda_env_name}
_backend_procs: dict[str, subprocess.Popen] = {}
_backend_ready: dict[str, bool] = {}
_backend_locks: dict[str, threading.Lock] = {}
_adapter_to_dir: dict[str, str] = {}      # {adapter.name: engine_dir} 反向映射
_engine_adapter_names: dict[str, str] = {} # {engine_dir: adapter.name} 引擎 -> 适配器名


def _find_conda_python(conda_env: str) -> str | None:
    """通过 conda info --json 查找指定环境的 python 路径。"""
    try:
        result = subprocess.run(
            ["conda", "info", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(result.stdout)
        for env in info.get("envs", []):
            # env 是完整路径，如 /home/user/miniconda3/envs/voxcpm
            if os.path.basename(env) == conda_env:
                if sys.platform == "win32":
                    python_path = os.path.join(env, "python.exe")
                else:
                    python_path = os.path.join(env, "bin", "python")
                if os.path.isfile(python_path):
                    return python_path
    except Exception as e:
        logger.error("[conda] Failed to query conda info: %s", e)
    return None


def _load_engine_configs(engine_defs: list[dict]):
    """预读各引擎配置，但不启动。engine_defs 是 config.yaml 中的 engines 列表。"""
    for eng_def in engine_defs:
        name = eng_def["name"]
        conda_env = eng_def.get("conda_env", "")
        config_path = os.path.join(PROJECT_ROOT, "tts", name, "config.yaml")
        if not os.path.isfile(config_path):
            logger.error("Config not found: %s", config_path)
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _engine_configs[name] = cfg
        _engine_conda_envs[name] = conda_env
        _backend_locks[name] = threading.Lock()
        logger.info("[config] Loaded %s (port %s, conda_env=%s)", name, cfg.get("webapi_port", "?"), conda_env)


def _get_adapter_name(engine_dir: str) -> str | None:
    """从引擎目录的 adapter 中读取 name 字段。"""
    config_path = os.path.join(PROJECT_ROOT, "tts", engine_dir, "config.yaml")
    try:
        mod = importlib.import_module(f"tts.{engine_dir}.adapter")
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseAdapter) and attr is not BaseAdapter:
                instance = attr(config_path=config_path)
                return instance.name
    except Exception:
        pass
    return None


def _check_health(port: int, expected_model: str | None = None) -> dict | None:
    """检查后端 health 端点。返回 health dict 或 None。"""
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if expected_model and data.get("model") != expected_model:
                return None  # 模型身份不匹配
            return data
    except requests.RequestException:
        pass
    return None


def _kill_port_process(port: int):
    """杀死占用指定端口的进程（通过 lsof/fuser 或 netstat）。"""
    import signal as sig
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                f"netstat -ano | findstr :{port} | findstr LISTENING",
                capture_output=True, text=True, timeout=5, shell=True,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split()
                if parts:
                    pid = int(parts[-1])
                    logger.warning("[backend] Killing process %d on port %d (wrong model)", pid, port)
                    os.kill(pid, sig.SIGTERM)
        else:
            # Linux/WSL: use fuser or lsof
            result = subprocess.run(
                ["fuser", f"{port}/tcp"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    logger.warning("[backend] Killing process %d on port %d (wrong model)", pid, port)
                    os.kill(pid, sig.SIGTERM)
                except (ValueError, ProcessLookupError):
                    pass
        time.sleep(1)
    except Exception as e:
        logger.warning("[backend] Failed to kill process on port %d: %s", port, e)


def ensure_backend(engine_dir: str) -> bool:
    """确保后端已启动并就绪，未启动则启动。返回是否就绪。"""
    if _backend_ready.get(engine_dir):
        return True

    lock = _backend_locks.get(engine_dir)
    if lock is None:
        return False

    with lock:
        # double-check
        if _backend_ready.get(engine_dir):
            return True

        cfg = _engine_configs.get(engine_dir)
        if not cfg:
            logger.error("[backend] No config for %s", engine_dir)
            return False

        port = cfg.get("webapi_port")
        if not port:
            logger.error("[backend] No webapi_port for %s", engine_dir)
            return False

        expected_model = _engine_adapter_names.get(engine_dir, engine_dir)

        # 检查端口上是否已有正确的后端在运行
        health = _check_health(port, expected_model)
        if health:
            logger.info("[backend] %s already running on port %d", engine_dir, port)
            _backend_ready[engine_dir] = True
            return True

        # 端口上有进程但模型身份不对 → 必须杀掉后重启
        wrong_health = _check_health(port)
        if wrong_health:
            logger.warning("[backend] Port %d occupied by wrong model '%s' (expected '%s'), killing...",
                           port, wrong_health.get("model"), expected_model)
            _kill_port_process(port)

        # 如果本引擎之前启动的子进程还活着，先停掉
        old_proc = _backend_procs.get(engine_dir)
        if old_proc and old_proc.poll() is None:
            old_proc.terminate()
            try:
                old_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                old_proc.kill()
            _backend_ready[engine_dir] = False

        # 确定 python 路径：优先使用 conda 环境，回退到 sys.executable
        conda_env = _engine_conda_envs.get(engine_dir, "")
        python_path = None
        if conda_env:
            python_path = _find_conda_python(conda_env)
            if python_path:
                logger.info("[backend] Using conda env '%s': %s", conda_env, python_path)
            else:
                logger.warning("[backend] Conda env '%s' not found, falling back to sys.executable", conda_env)

        if not python_path:
            python_path = sys.executable

        # 启动后端
        webapi_script = os.path.join(PROJECT_ROOT, "tts", engine_dir, "webapi.py")
        if not os.path.isfile(webapi_script):
            logger.error("[backend] webapi.py not found: %s", webapi_script)
            return False

        logger.info("[backend] Starting %s on port %d ...", engine_dir, port)
        proc = subprocess.Popen(
            [python_path, webapi_script, "--port", str(port)],
            cwd=PROJECT_ROOT,
            stdout=None,
            stderr=None,
        )
        _backend_procs[engine_dir] = proc

        # 等待就绪
        deadline = time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                logger.error("[backend] %s exited with code %d", engine_dir, proc.returncode)
                return False
            health = _check_health(port, expected_model)
            if health:
                logger.info("[backend] %s is ready (model=%s, sr=%s)",
                            engine_dir, health.get("model"), health.get("sample_rate"))
                _backend_ready[engine_dir] = True
                return True
            time.sleep(5)

        logger.error("[backend] %s failed to start within 600s", engine_dir)
        return False


def stop_backends():
    """关闭所有后端子进程。"""
    for name, proc in _backend_procs.items():
        if proc.poll() is None:
            logger.info("[backend] Stopping %s ...", name)
            proc.terminate()
    for proc in _backend_procs.values():
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    _backend_procs.clear()
    _backend_ready.clear()


# ---------------------------------------------------------------------------
# 加载适配器
# ---------------------------------------------------------------------------
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_adapters(engine_defs: list[dict]) -> dict[str, BaseAdapter]:
    """动态导入 tts/{engine}/adapter.py 并实例化。"""
    adapters: dict[str, BaseAdapter] = {}
    for eng_def in engine_defs:
        name = eng_def["name"]
        config_path = os.path.join(PROJECT_ROOT, "tts", name, "config.yaml")
        try:
            mod = importlib.import_module(f"tts.{name}.adapter")
        except ImportError as e:
            logger.error("Failed to import adapter '%s': %s", name, e)
            continue

        adapter_cls = None
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseAdapter) and attr is not BaseAdapter:
                adapter_cls = attr
                break

        if adapter_cls is None:
            logger.error("No BaseAdapter subclass found in tts.%s.adapter", name)
            continue

        adapter = adapter_cls(config_path=config_path)
        adapters[adapter.name] = adapter
        _adapter_to_dir[adapter.name] = name
        _engine_adapter_names[name] = adapter.name
        logger.info("[adapter] %s -> %s", name, adapter.name)
    return adapters


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TTS Test Gateway", description="统一评测网关，兼容 OpenAI 接口")

config = load_config()
engine_defs = config.get("engines", [])
_load_engine_configs(engine_defs)
adapters = load_adapters(engine_defs)

signal.signal(signal.SIGTERM, lambda s, f: (stop_backends(), sys.exit(0)))
signal.signal(signal.SIGINT, lambda s, f: (stop_backends(), sys.exit(0)))


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    # ── 标准 OpenAI 字段 ──
    model: str = Field(..., description="引擎名称，如 VoxCPM2")
    input: str = Field(..., description="待合成文本")
    voice: str = Field(default="default", description="音色名称、音色描述或情绪标签")
    response_format: str = Field(default="wav", description="输出格式: wav / mp3 / opus / pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="语速倍率")
    # ── 扩展测试字段 ──
    reference_audio: Optional[str] = Field(default=None, description="Base64 编码的参考音频（voice clone）")
    prompt_text: Optional[str] = Field(default=None, description="参考音频对应的文本转录")
    reference_wav_path: Optional[str] = Field(default=None, description="参考音频本地文件路径（仅供 webapi 本地使用）")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="采样温度")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Nucleus sampling 阈值")
    repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0, description="重复惩罚系数")
    # ── session 自举克隆字段 ──
    session_id: Optional[str] = Field(default=None, description="Session ID for bootstrap cloning across segments")
    session_action: Optional[str] = Field(default=None, description="start|continue|end")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def _resolve_engine_dir(model_name: str) -> str | None:
    """根据 adapter name 反查引擎目录名。

    adapters 字典的 key 是 adapter.name（即请求中的 model 字段），
    _adapter_to_dir 维护 adapter.name -> engine_dir 的映射。
    """
    return _adapter_to_dir.get(model_name)


@app.post("/v1/audio/speech")
def create_speech(request: TTSRequest):
    logger.info("[gateway] POST /v1/audio/speech model=%s input_len=%d", request.model, len(request.input))

    adapter = adapters.get(request.model)
    if adapter is None:
        logger.error("[gateway] Unknown model '%s', available: %s", request.model, list(adapters.keys()))
        raise HTTPException(400, f"Unknown model: {request.model}. Available: {list(adapters.keys())}")

    if not request.input.strip():
        raise HTTPException(400, "input text is empty")

    # 按需启动后端
    engine_dir = _resolve_engine_dir(request.model)
    if engine_dir and not ensure_backend(engine_dir):
        raise HTTPException(503, f"Backend for '{request.model}' is not available. Check logs/{engine_dir}.log")

    # 解码 base64 音频
    ref_bytes = None
    if request.reference_audio:
        try:
            ref_bytes = base64.b64decode(request.reference_audio)
        except Exception:
            raise HTTPException(400, "reference_audio is not valid base64")

    try:
        wav = adapter.synthesize(
            text=request.input,
            voice=request.voice,
            reference_wav_bytes=ref_bytes,
            prompt_text=request.prompt_text,
            speed=request.speed,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            session_id=request.session_id,
            session_action=request.session_action,
        )
    except Exception as e:
        logger.error("synthesize failed for '%s': %s", request.model, e, exc_info=True)
        raise HTTPException(500, f"Generation failed: {e}")

    buffer = io.BytesIO()
    sf.write(buffer, wav, adapter.sample_rate, format="WAV")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")


@app.post("/v1/audio/speech/stream")
def create_speech_stream(request: TTSRequest):
    adapter = adapters.get(request.model)
    if adapter is None:
        raise HTTPException(400, f"Unknown model: {request.model}. Available: {list(adapters.keys())}")

    if not request.input.strip():
        raise HTTPException(400, "input text is empty")

    engine_dir = _resolve_engine_dir(request.model)
    if engine_dir and not ensure_backend(engine_dir):
        raise HTTPException(503, f"Backend for '{request.model}' is not available")

    ref_bytes = None
    if request.reference_audio:
        try:
            ref_bytes = base64.b64decode(request.reference_audio)
        except Exception:
            raise HTTPException(400, "reference_audio is not valid base64")

    sr = adapter.sample_rate

    def _stream():
        for chunk in adapter.synthesize_streaming(
            text=request.input,
            voice=request.voice,
            reference_wav_bytes=ref_bytes,
            prompt_text=request.prompt_text,
            speed=request.speed,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        ):
            yield chunk.astype(np.float32).tobytes()

    return StreamingResponse(
        _stream(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(sr), "X-Sample-Format": "float32"},
    )


@app.get("/health")
def health():
    status = {}
    for name, adapter in adapters.items():
        status[name] = "ok" if adapter.health_check() else "not_started"
    return {"status": "ok", "adapters": status}


@app.get("/v1/models")
def list_models():
    """返回所有注册的可用模型及其能力。"""
    models = []
    for name, adapter in adapters.items():
        info = adapter.to_dict()
        info["available"] = adapter.health_check()
        models.append(info)
    return {"models": models}


class InitModelRequest(BaseModel):
    model: str = Field(..., description="引擎名称，如 VoxCPM2 或 MOSS-TTSD")


class InitModelResponse(BaseModel):
    model: str
    status: str                   # "started" | "already_running" | "restarted" | "error"
    is_restarted: bool            # 是否重新启动（杀掉旧进程后重启）
    elapsed_seconds: float        # 启动耗时（秒）
    health: Optional[dict] = None # health 端点返回的信息
    error: Optional[str] = None


@app.post("/v1/init_model", response_model=InitModelResponse)
def init_model(request: InitModelRequest):
    """显式启动指定引擎的后端进程。

    runner.py 在评测开始时调用一次，用于提前加载模型、
    记录启动耗时，以及判断是否需要重启。
    """
    model_name = request.model
    logger.info("[gateway] POST /v1/init_model model=%s", model_name)

    adapter = adapters.get(model_name)
    if adapter is None:
        raise HTTPException(400, f"Unknown model: {model_name}. Available: {list(adapters.keys())}")

    engine_dir = _resolve_engine_dir(model_name)
    if not engine_dir:
        raise HTTPException(400, f"No engine directory found for model: {model_name}")

    cfg = _engine_configs.get(engine_dir, {})
    port = cfg.get("webapi_port")
    expected_model = _engine_adapter_names.get(engine_dir, engine_dir)

    # 检查当前状态
    health = _check_health(port, expected_model)
    if health:
        # 已经是正确的模型在运行
        return InitModelResponse(
            model=model_name,
            status="already_running",
            is_restarted=False,
            elapsed_seconds=0,
            health=health,
        )

    # 判断是否需要杀掉错误进程后重启
    wrong_health = _check_health(port)
    is_restarted = wrong_health is not None

    # 清除 ready 标记，强制 ensure_backend 走启动流程
    _backend_ready[engine_dir] = False

    t0 = time.perf_counter()
    ok = ensure_backend(engine_dir)
    elapsed = round(time.perf_counter() - t0, 2)

    if not ok:
        return InitModelResponse(
            model=model_name,
            status="error",
            is_restarted=is_restarted,
            elapsed_seconds=elapsed,
            error=f"Backend failed to start within 600s",
        )

    health = _check_health(port, expected_model)
    status = "restarted" if is_restarted else "started"

    return InitModelResponse(
        model=model_name,
        status=status,
        is_restarted=is_restarted,
        elapsed_seconds=elapsed,
        health=health,
    )


@app.get("/v1/models/{model_id}/capabilities")
def model_capabilities(model_id: str):
    """返回指定模型的能力详情。"""
    adapter = adapters.get(model_id)
    if adapter is None:
        raise HTTPException(404, f"Unknown model: {model_id}. Available: {list(adapters.keys())}")
    return adapter.to_dict()


@app.on_event("shutdown")
def on_shutdown():
    stop_backends()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TTS Test Gateway")
    parser.add_argument("--host", default=config.get("host", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=config.get("port", 9000))
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
