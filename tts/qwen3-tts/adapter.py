"""Qwen3-TTS 适配器 — 通过 HTTP 调用 Qwen3-TTS 的 OpenAI 兼容 webapi。"""

import base64
import io
import logging
from typing import Generator

import numpy as np
import requests
import soundfile as sf
import yaml

from server.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class Qwen3TTSAdapter(BaseAdapter):
    name = "qwen3-tts"
    sample_rate = 24000

    # Qwen3-TTS 支持三种能力
    capabilities = ["tts", "voice_clone", "voice_design"]

    # 预设音色（Qwen3-TTS CustomVoice 内置）
    voices = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
              "Ryan", "Aiden", "Ono_Anna", "Sohee"]

    # 克隆音色（对应 voices/ 目录下的 wav 文件）
    clone_voices = []

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = __file__.replace("adapter.py", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.base_url = cfg["backend_url"].rstrip("/")
        self.timeout = cfg.get("timeout", 120)
        self.sample_rate = cfg.get("sample_rate", 24000)
        logger.info("qwen3-tts adapter -> %s", self.base_url)

    def _build_request(self, text: str, voice: str = "default",
                       reference_wav_bytes: bytes | None = None,
                       prompt_text: str | None = None,
                       instruction: str | None = None,
                       speed: float = 1.0,
                       temperature: float = 0.9,
                       top_p: float = 1.0,
                       repetition_penalty: float = 1.05,
                       session_id: str | None = None,
                       session_action: str | None = None) -> dict:
        payload = {
            "model": "qwen3-tts",
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "wav",
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        if reference_wav_bytes:
            payload["reference_audio"] = base64.b64encode(reference_wav_bytes).decode("ascii")
        if prompt_text:
            payload["prompt_text"] = prompt_text
        if instruction:
            payload["instruction"] = instruction
        if session_id:
            payload["session_id"] = session_id
            if session_action:
                payload["session_action"] = session_action
        return payload

    def synthesize(self, text: str, voice: str = "default",
                   reference_wav_bytes: bytes | None = None,
                   prompt_text: str | None = None,
                   instruction: str | None = None,
                   speed: float = 1.0,
                   temperature: float = 0.9,
                   top_p: float = 1.0,
                   repetition_penalty: float = 1.05,
                   session_id: str | None = None,
                   session_action: str | None = None,
                   **kwargs) -> np.ndarray:
        payload = self._build_request(text, voice, reference_wav_bytes,
                                      prompt_text, instruction, speed,
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
                             instruction: str | None = None,
                             speed: float = 1.0,
                             temperature: float = 0.9,
                             top_p: float = 1.0,
                             repetition_penalty: float = 1.05,
                             session_id: str | None = None,
                             session_action: str | None = None,
                             **kwargs) -> Generator[np.ndarray, None, None]:
        payload = self._build_request(text, voice, reference_wav_bytes,
                                      prompt_text, instruction, speed,
                                      temperature, top_p, repetition_penalty,
                                      session_id, session_action)
        resp = requests.post(
            f"{self.base_url}/v1/audio/speech/stream",
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()
        sr = int(resp.headers.get("X-Sample-Rate", self.sample_rate))
        self.sample_rate = sr

        chunk_size = sr * 4  # float32 = 4 bytes
        buffer = b""
        for chunk in resp.iter_content(chunk_size=chunk_size):
            buffer += chunk
            if len(buffer) >= chunk_size:
                data = np.frombuffer(buffer[:chunk_size], dtype=np.float32)
                buffer = buffer[chunk_size:]
                yield data

        if buffer:
            yield np.frombuffer(buffer, dtype=np.float32)

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
