"""TTS 适配器基类，所有引擎适配器需继承此类。"""

from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


class BaseAdapter(ABC):
    """统一的 TTS 适配器接口。

    每个引擎实现此类，供 server/main.py 调用。
    """

    name: str = ""
    sample_rate: int = 0

    # 能力声明，子类按实际情况覆盖
    capabilities: list[str] = ["tts"]  # 可选: voice_design, voice_clone, emotion, ssml, streaming

    # 预设音色列表
    voices: list[str] = ["default"]

    # 克隆音色列表（对应 voices/ 目录下的文件名，不含扩展名）
    clone_voices: list[str] = []

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str = "default",
        reference_wav_bytes: bytes | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        **kwargs,
    ) -> np.ndarray:
        """合成语音，返回 1D float32 numpy 数组。"""
        ...

    @abstractmethod
    def synthesize_streaming(
        self,
        text: str,
        voice: str = "default",
        reference_wav_bytes: bytes | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        **kwargs,
    ) -> Generator[np.ndarray, None, None]:
        """流式合成语音，逐步 yield 音频 chunk。"""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """检查后端是否可用。"""
        ...

    def get_clone_voice_path(self, voice_name: str) -> str | None:
        """返回克隆音色的参考音频路径。"""
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "voices", f"{voice_name}.wav")
        return path if os.path.isfile(path) else None

    def to_dict(self) -> dict:
        """序列化为 API 响应格式。"""
        return {
            "id": self.name,
            "capabilities": self.capabilities,
            "voices": self.voices,
            "clone_voices": self.clone_voices,
            "sample_rate": self.sample_rate,
        }
