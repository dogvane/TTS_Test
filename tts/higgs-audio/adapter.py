"""Higgs Audio v3 TTS 适配器 — 通过 HTTP 调用 Docker 中的 SGLang-Omni 服务。

Higgs Audio 通过 Docker 启动（sglang-omni），提供 OpenAI 兼容的
/v1/audio/speech 接口，无需本地 conda 环境。
"""

import base64
import io
import logging
import os
import shutil
import tempfile
from typing import Generator

import numpy as np
import requests
import soundfile as sf
import yaml

from server.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# Docker 挂载配置：宿主机 models 目录挂载到容器内 /model
DOCKER_MODEL_HOST_DIR = r"G:\ai\TTS\higgs-audio-v3-tts-4b\models"
DOCKER_MODEL_CONTAINER_DIR = "/model"
REF_AUDIO_SUBDIR = "ref_audio"


class HiggsAudioAdapter(BaseAdapter):
    name = "higgs-audio"
    sample_rate = 24000

    # Higgs Audio 支持 TTS 和语音克隆
    capabilities = ["tts", "voice_clone"]

    # 预设音色
    voices = ["default"]

    # 克隆音色（对应 voices/ 目录下的 wav 文件）
    clone_voices = []

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = __file__.replace("adapter.py", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.base_url = cfg["backend_url"].rstrip("/")
        self.timeout = cfg.get("timeout", 300)
        self.sample_rate = cfg.get("sample_rate", 24000)
        logger.info("higgs-audio adapter -> %s", self.base_url)

    def _build_request(self, text: str, voice: str = "default",
                       reference_wav_bytes: bytes | None = None,
                       prompt_text: str | None = None,
                       speed: float = 1.0,
                       temperature: float = 1.0,
                       top_p: float = 0.9,
                       repetition_penalty: float = 1.1,
                       session_id: str | None = None,
                       session_action: str | None = None) -> dict:
        payload = {
            "input": text,
        }
        if reference_wav_bytes and prompt_text:
            # 将参考音频保存到 Docker 挂载目录，使用容器内路径
            host_ref_dir = os.path.join(DOCKER_MODEL_HOST_DIR, REF_AUDIO_SUBDIR)
            os.makedirs(host_ref_dir, exist_ok=True)
            # 用 session_id 或临时文件名作为文件名
            ref_name = (session_id or "ref") + ".wav"
            host_ref_path = os.path.join(host_ref_dir, ref_name)
            with open(host_ref_path, "wb") as f:
                f.write(reference_wav_bytes)
            container_path = f"{DOCKER_MODEL_CONTAINER_DIR}/{REF_AUDIO_SUBDIR}/{ref_name}"
            payload["references"] = [{
                "audio_path": container_path,
                "text": prompt_text,
            }]
        if temperature != 1.0:
            payload["temperature"] = temperature
        if top_p != 0.9:
            payload["top_p"] = top_p
        return payload

    def synthesize(self, text: str, voice: str = "default",
                   reference_wav_bytes: bytes | None = None,
                   prompt_text: str | None = None,
                   speed: float = 1.0,
                   temperature: float = 1.0,
                   top_p: float = 0.9,
                   repetition_penalty: float = 1.1,
                   session_id: str | None = None,
                   session_action: str | None = None,
                   **kwargs) -> np.ndarray:
        payload = self._build_request(text, voice, reference_wav_bytes,
                                      prompt_text, speed,
                                      temperature, top_p, repetition_penalty,
                                      session_id, session_action)
        resp = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        wav, sr = sf.read(io.BytesIO(resp.content))
        self.sample_rate = sr
        return wav.astype(np.float32)

    def synthesize_streaming(self, text: str, voice: str = "default",
                             reference_wav_bytes: bytes | None = None,
                             prompt_text: str | None = None,
                             speed: float = 1.0,
                             temperature: float = 1.0,
                             top_p: float = 0.9,
                             repetition_penalty: float = 1.1,
                             session_id: str | None = None,
                             session_action: str | None = None,
                             **kwargs) -> Generator[np.ndarray, None, None]:
        payload = self._build_request(text, voice, reference_wav_bytes,
                                      prompt_text, speed,
                                      temperature, top_p, repetition_penalty,
                                      session_id, session_action)
        payload["stream"] = True
        resp = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()

        import json as json_mod
        chunk_buf = b""
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json_mod.loads(line[6:])
            if event.get("finish_reason") == "stop":
                break
            audio = event.get("audio") or {}
            if audio.get("data"):
                chunk_buf += base64.b64decode(audio["data"])
                # 按 float32 对齐返回
                float_count = len(chunk_buf) // 4
                if float_count > 0:
                    data = np.frombuffer(chunk_buf[:float_count * 4], dtype=np.float32)
                    chunk_buf = chunk_buf[float_count * 4:]
                    yield data

        if chunk_buf:
            yield np.frombuffer(chunk_buf, dtype=np.float32)

    def health_check(self) -> bool:
        """检查 Docker 容器中的服务是否可用。"""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            pass
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
