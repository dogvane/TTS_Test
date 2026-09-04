"""Qwen3-TTS Web API — OpenAI 兼容 /v1/audio/speech 端点。

使用 qwen_tts 包提供三种模式：
  - tts (CustomVoice): 预设音色合成
  - voice_design (VoiceDesign): 根据文字描述设计音色
  - voice_clone (Base): 根据参考音频克隆音色

三种模型按需懒加载，不同时占用显存。

启动:
    conda activate qwen3-tts
    cd /mnt/o/ai/TTS/TTS_Test
    python -m tts.qwen3-tts.webapi --port 8002

注意: 本文件在 WSL2 conda 环境中运行，路径使用 /mnt/ 前缀。
"""

import io
import os
import sys
import argparse
import base64
import logging
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 加载配置
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径 & 设备
# ---------------------------------------------------------------------------
CUSTOM_VOICE_MODEL = _cfg["custom_voice_model"]
VOICE_DESIGN_MODEL = _cfg["voice_design_model"]
VOICE_CLONE_MODEL = _cfg["voice_clone_model"]
DEVICE = _cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32

# 预设音色列表
PRESET_VOICES = _cfg.get("voices", ["Vivian"])

logger.info("Qwen3-TTS CustomVoice model: %s", CUSTOM_VOICE_MODEL)
logger.info("Qwen3-TTS VoiceDesign model: %s", VOICE_DESIGN_MODEL)
logger.info("Qwen3-TTS VoiceClone model: %s", VOICE_CLONE_MODEL)
logger.info("Device: %s, dtype: %s", DEVICE, DTYPE)

# ---------------------------------------------------------------------------
# 模型加载（按需懒加载，不同时占用显存）
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3-TTS API", description="OpenAI-compatible TTS API")

_custom_voice_model = None   # Qwen3-TTS-12Hz-1.7B-CustomVoice
_voice_design_model = None   # Qwen3-TTS-12Hz-1.7B-VoiceDesign
_voice_clone_model = None    # Qwen3-TTS-12Hz-1.7B-Base

SAMPLE_RATE = 24000


def _gpu_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_custom_voice():
    global _custom_voice_model
    if _custom_voice_model is None:
        return
    logger.info("[CustomVoice] Unloading model...")
    del _custom_voice_model
    _custom_voice_model = None
    _gpu_gc()
    logger.info("[CustomVoice] Model unloaded")


def _unload_voice_design():
    global _voice_design_model
    if _voice_design_model is None:
        return
    logger.info("[VoiceDesign] Unloading model...")
    del _voice_design_model
    _voice_design_model = None
    _gpu_gc()
    logger.info("[VoiceDesign] Model unloaded")


def _unload_voice_clone():
    global _voice_clone_model
    if _voice_clone_model is None:
        return
    logger.info("[VoiceClone] Unloading model...")
    del _voice_clone_model
    _voice_clone_model = None
    _gpu_gc()
    logger.info("[VoiceClone] Model unloaded")


def _load_custom_voice():
    """加载 CustomVoice 模型（用于基础 TTS）。先释放其他模型。"""
    global _custom_voice_model
    if _custom_voice_model is not None:
        return _custom_voice_model

    _unload_voice_design()
    _unload_voice_clone()

    from qwen_tts import Qwen3TTSModel

    logger.info("[CustomVoice] Loading model from '%s' ...", CUSTOM_VOICE_MODEL)
    model = Qwen3TTSModel.from_pretrained(
        CUSTOM_VOICE_MODEL,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation="flash_attention_2",
    )
    _custom_voice_model = model
    logger.info("[CustomVoice] Model loaded.")
    return model


def _load_voice_design():
    """加载 VoiceDesign 模型（用于音色设计）。先释放其他模型。"""
    global _voice_design_model
    if _voice_design_model is not None:
        return _voice_design_model

    _unload_custom_voice()
    _unload_voice_clone()

    from qwen_tts import Qwen3TTSModel

    logger.info("[VoiceDesign] Loading model from '%s' ...", VOICE_DESIGN_MODEL)
    model = Qwen3TTSModel.from_pretrained(
        VOICE_DESIGN_MODEL,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation="flash_attention_2",
    )
    _voice_design_model = model
    logger.info("[VoiceDesign] Model loaded.")
    return model


def _load_voice_clone():
    """加载 VoiceClone (Base) 模型（用于语音克隆）。先释放其他模型。"""
    global _voice_clone_model
    if _voice_clone_model is not None:
        return _voice_clone_model

    _unload_custom_voice()
    _unload_voice_design()

    from qwen_tts import Qwen3TTSModel

    logger.info("[VoiceClone] Loading model from '%s' ...", VOICE_CLONE_MODEL)
    model = Qwen3TTSModel.from_pretrained(
        VOICE_CLONE_MODEL,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation="flash_attention_2",
    )
    _voice_clone_model = model
    logger.info("[VoiceClone] Model loaded.")
    return model


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    # 标准 OpenAI 字段
    model: str = Field(default="qwen3-tts", description="Model name")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Voice name or description")
    response_format: str = Field(default="wav", description="Output format: wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed multiplier")
    # 扩展测试字段
    reference_audio: Optional[str] = Field(default=None, description="Base64 encoded reference audio for cloning")
    prompt_text: Optional[str] = Field(default=None, description="Transcript of reference audio")
    reference_wav_path: Optional[str] = Field(default=None, description="Path to reference audio file")
    instruction: Optional[str] = Field(default=None, description="Voice design instruction (text description)")
    temperature: float = Field(default=0.9, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.1, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: float = Field(default=1.05, ge=0.9, le=2.0, description="Repetition penalty")
    # session
    session_id: Optional[str] = Field(default=None, description="Session ID for bootstrap cloning")
    session_action: Optional[str] = Field(default=None, description="start|continue|end")


# ---------------------------------------------------------------------------
# Session 缓存（自举克隆）
# ---------------------------------------------------------------------------
_session_cache: dict = {}


def _resolve_session_action(request: TTSRequest) -> str:
    if not request.session_id:
        return "none"
    if request.session_action in ("start", "continue", "end"):
        return request.session_action
    return "continue" if request.session_id in _session_cache else "none"


# ---------------------------------------------------------------------------
# 推理路由
# ---------------------------------------------------------------------------
def _detect_language(text: str) -> str:
    """简单语言检测：根据文本中的中文字符比例判断。"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars > len(text) * 0.3:
        return "Chinese"
    return "Auto"


def _route_request(request: TTSRequest) -> str:
    """判断请求应该走哪条推理路径: tts / voice_design / voice_clone。"""
    # 有参考音频 → voice_clone
    if request.reference_audio or request.reference_wav_path:
        return "voice_clone"
    # session continue 走 voice_clone
    if request.session_id and request.session_id in _session_cache:
        return "voice_clone"
    # 有 instruction 字段 → voice_design
    if request.instruction and request.instruction.strip():
        return "voice_design"
    # voice 不是预设音色也不是 default → 当作 voice_design 的描述
    if request.voice and request.voice not in PRESET_VOICES and request.voice != "default":
        return "voice_design"
    return "tts"


# ---------------------------------------------------------------------------
# TTS (CustomVoice)
# ---------------------------------------------------------------------------
def _run_tts(request: TTSRequest) -> np.ndarray:
    model = _load_custom_voice()
    text = request.input.strip()

    speaker = request.voice if request.voice in PRESET_VOICES else "Vivian"
    language = _detect_language(text)

    logger.info("[TTS] speaker=%s language=%s text='%s...'", speaker, language, text[:50])

    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )

    wav = wavs[0]
    if wav.ndim > 1:
        wav = wav.squeeze(0)
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Voice Design
# ---------------------------------------------------------------------------
def _run_voice_design(request: TTSRequest) -> np.ndarray:
    model = _load_voice_design()
    text = request.input.strip()
    # instruction 优先；如果 voice 不是 default 也作为 instruction
    instruct = request.instruction.strip() if request.instruction else ""
    if not instruct and request.voice and request.voice != "default":
        instruct = request.voice
    if not instruct:
        raise HTTPException(400, "instruction (or voice description) is required for voice_design")

    language = _detect_language(text)

    logger.info("[VoiceDesign] instruct='%s...' language=%s text='%s...'",
                instruct[:30], language, text[:50])

    wavs, sr = model.generate_voice_design(
        text=text,
        instruct=instruct,
        language=language,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )

    wav = wavs[0]
    if wav.ndim > 1:
        wav = wav.squeeze(0)
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Voice Clone (Base)
# ---------------------------------------------------------------------------
def _get_reference_audio(request: TTSRequest) -> tuple[np.ndarray, int] | None:
    """从请求中提取参考音频，返回 (wav_array, sample_rate) 或 None。"""
    ref_path = request.reference_wav_path

    if request.reference_audio:
        try:
            audio_bytes = base64.b64decode(request.reference_audio)
        except Exception:
            raise HTTPException(400, "reference_audio is not valid base64")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        ref_path = tmp.name

    if not ref_path:
        return None

    if not os.path.isfile(ref_path):
        raise HTTPException(400, f"Reference audio not found: {ref_path}")

    wav, sr = sf.read(ref_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # 清理临时文件
    if request.reference_audio and ref_path:
        try:
            os.unlink(ref_path)
        except OSError:
            pass

    return (wav, sr)


def _run_voice_clone(request: TTSRequest) -> np.ndarray:
    model = _load_voice_clone()
    text = request.input.strip()
    language = _detect_language(text)

    # session 缓存的自举克隆
    session_action = _resolve_session_action(request)
    if session_action == "continue" and request.session_id in _session_cache:
        session = _session_cache[request.session_id]
        logger.info("[VoiceClone] session=%s using cached prompt", request.session_id)
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=session["prompt"],
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        wav = wavs[0]
        if wav.ndim > 1:
            wav = wav.squeeze(0)
        return wav.astype(np.float32)

    # 普通克隆：需要参考音频
    ref_result = _get_reference_audio(request)
    if ref_result is None:
        raise HTTPException(400, "Reference audio is required for voice_clone")

    ref_wav, ref_sr = ref_result
    prompt_text = request.prompt_text or ""

    logger.info("[VoiceClone] ref_sr=%d prompt='%s...' text='%s...'",
                ref_sr, prompt_text[:30], text[:50])

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=(ref_wav, ref_sr),
        ref_text=prompt_text if prompt_text.strip() else None,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )

    wav = wavs[0]
    if wav.ndim > 1:
        wav = wav.squeeze(0)
    wav = wav.astype(np.float32)

    # session: 缓存 clone prompt 供后续分段使用
    if request.session_id and session_action in ("start", "none"):
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=(ref_wav, ref_sr),
                ref_text=prompt_text if prompt_text.strip() else None,
            )
            _session_cache[request.session_id] = {"prompt": prompt}
            logger.info("[SESSION %s] Cached clone prompt", request.session_id)
        except Exception as e:
            logger.warning("[SESSION %s] Failed to cache prompt: %s", request.session_id, e)

    return wav


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/audio/speech")
def create_speech(request: TTSRequest):
    if not request.input.strip():
        raise HTTPException(400, "input text is empty")

    mode = _route_request(request)

    try:
        if mode == "voice_design":
            wav = _run_voice_design(request)
        elif mode == "voice_clone":
            wav = _run_voice_clone(request)
        else:
            wav = _run_tts(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("%s failed: %s", mode, e, exc_info=True)
        raise HTTPException(500, f"Generation failed: {e}")

    # session 清理
    if request.session_id and request.session_action == "end":
        _session_cache.pop(request.session_id, None)
        logger.info("[SESSION %s] Cleaned up", request.session_id)

    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")


@app.post("/v1/audio/speech/stream")
def create_speech_stream(request: TTSRequest):
    # Qwen3-TTS 不支持真正的逐 chunk 流式，先整体推理再模拟流式返回
    if not request.input.strip():
        raise HTTPException(400, "input text is empty")

    mode = _route_request(request)

    def _stream():
        try:
            if mode == "voice_design":
                wav = _run_voice_design(request)
            elif mode == "voice_clone":
                wav = _run_voice_clone(request)
            else:
                wav = _run_tts(request)

            if wav.ndim > 1:
                wav = wav.squeeze(0)
            data = wav.astype(np.float32)
            chunk_samples = SAMPLE_RATE  # 每秒一个 chunk
            for i in range(0, len(data), chunk_samples):
                yield data[i:i + chunk_samples].tobytes()
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Streaming failed: %s", e, exc_info=True)
            raise

    return StreamingResponse(
        _stream(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Sample-Format": "float32"},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "qwen3-tts",
        "capabilities": ["tts", "voice_clone", "voice_design"],
        "sample_rate": SAMPLE_RATE,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run Qwen3-TTS Web API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args, _ = parser.parse_known_args()
    uvicorn.run(app, host=args.host, port=args.port)
