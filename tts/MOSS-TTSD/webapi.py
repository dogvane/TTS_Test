"""MOSS-TTSD + VoiceGenerator Web API — OpenAI 兼容 /v1/audio/speech 端点。

本文件位于 TTS_Test 项目内，通过 sys.path 引用外部 MOSS-TTSD 源码。

启动:
    conda activate moss-tts
    cd /mnt/o/ai/TTS/TTS_Test
    python -m tts.MOSS-TTSD.webapi --port 8004

注意: 本文件在 WSL2 conda 环境中运行，路径使用 /mnt/ 前缀。
"""

import io
import os
import re
import sys
import argparse
import base64
import logging
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ---------------------------------------------------------------------------
# 加载配置
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

_source_path = _cfg["source_path"]
if _source_path not in sys.path:
    sys.path.insert(0, _source_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径 & 设备
# ---------------------------------------------------------------------------
MODEL_PATH = _cfg["model_path"]
CODEC_PATH = _cfg.get("codec_path", "OpenMOSS-Team/MOSS-Audio-Tokenizer")
VOICEGEN_PATH = _cfg["voicegen_path"]
DEVICE = _cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32

logger.info("MOSS-TTSD  model: %s", MODEL_PATH)
logger.info("VoiceGenerator model: %s", VOICEGEN_PATH)
logger.info("Codec: %s", CODEC_PATH)
logger.info("Device: %s, dtype: %s", DEVICE, DTYPE)

# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
app = FastAPI(title="MOSS-TTSD TTS API", description="OpenAI-compatible TTS API")

# 全局模型引用：按需加载，避免同时占用两份显存
_ttsd_model = None       # MOSS-TTSD-v1.0
_ttsd_processor = None
_ttsd_sr: int = 0

_voicegen_model = None   # MOSS-VoiceGenerator
_voicegen_processor = None
_voicegen_sr: int = 0

# session 缓存：{session_id: {"wav": numpy_array, "sr": int, "prompt_text": str}}
_session_cache: dict = {}


def _gpu_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_audio_tokenizer_device(processor, device: str):
    if getattr(processor, "audio_tokenizer", None) is None:
        return
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    processor.audio_tokenizer.eval()


def _unload_ttsd():
    """释放 MOSS-TTSD 模型显存。"""
    global _ttsd_model, _ttsd_processor, _ttsd_sr
    if _ttsd_model is None:
        return
    logger.info("[TTSD] Unloading model, freeing GPU memory...")
    del _ttsd_model, _ttsd_processor
    _ttsd_model = None
    _ttsd_processor = None
    _ttsd_sr = 0
    torch.cuda.empty_cache()
    logger.info("[TTSD] Model unloaded")


def _unload_voicegen():
    """释放 MOSS-VoiceGenerator 模型显存。"""
    global _voicegen_model, _voicegen_processor, _voicegen_sr
    if _voicegen_model is None:
        return
    logger.info("[VoiceGen] Unloading model, freeing GPU memory...")
    del _voicegen_model, _voicegen_processor
    _voicegen_model = None
    _voicegen_processor = None
    _voicegen_sr = 0
    torch.cuda.empty_cache()
    logger.info("[VoiceGen] Model unloaded")


def _load_ttsd():
    """加载 MOSS-TTSD 模型（用于 tts / voice_clone），先释放 VoiceGenerator。"""
    global _ttsd_model, _ttsd_processor, _ttsd_sr
    if _ttsd_model is not None:
        return _ttsd_model, _ttsd_processor, _ttsd_sr

    _unload_voicegen()

    from transformers import AutoModel, AutoProcessor

    logger.info("[TTSD] Loading model from '%s' ...", MODEL_PATH)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        codec_path=CODEC_PATH,
    )
    if getattr(processor, "audio_tokenizer", None) is not None:
        tokenizer_device = "cpu" if DEVICE.startswith("cuda") else DEVICE
        _set_audio_tokenizer_device(processor, tokenizer_device)

    def _load_with_attn(attn_impl: str):
        return AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=DTYPE,
        ).to(DEVICE)

    if DEVICE.startswith("cuda"):
        try:
            model = _load_with_attn("flash_attention_2")
        except Exception as e:
            logger.warning("[TTSD] flash_attention_2 unavailable, fallback to sdpa: %s", e)
            model = _load_with_attn("sdpa")
    else:
        model = _load_with_attn("sdpa")

    model.eval()
    _ttsd_model = model
    _ttsd_processor = processor
    _ttsd_sr = int(processor.model_config.sampling_rate)
    logger.info("[TTSD] Model loaded. Sample rate: %d Hz", _ttsd_sr)
    return model, processor, _ttsd_sr


def _load_voicegen():
    """加载 MOSS-VoiceGenerator 模型（用于 voice_design），先释放 TTSD。"""
    global _voicegen_model, _voicegen_processor, _voicegen_sr
    if _voicegen_model is not None:
        return _voicegen_model, _voicegen_processor, _voicegen_sr

    _unload_ttsd()

    from transformers import AutoModel, AutoProcessor

    logger.info("[VoiceGen] Loading model from '%s' ...", VOICEGEN_PATH)
    processor = AutoProcessor.from_pretrained(
        VOICEGEN_PATH,
        trust_remote_code=True,
        codec_path=CODEC_PATH,
        normalize_inputs=True,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)

    model = AutoModel.from_pretrained(
        VOICEGEN_PATH,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    _voicegen_model = model
    _voicegen_processor = processor
    _voicegen_sr = int(processor.model_config.sampling_rate)
    logger.info("[VoiceGen] Model loaded. Sample rate: %d Hz", _voicegen_sr)
    return model, processor, _voicegen_sr


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    # ── 标准 OpenAI 字段 ──
    model: str = Field(default="MOSS-TTSD", description="Model name")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Voice name, description, or emotion tag")
    response_format: str = Field(default="wav", description="Output format: wav / mp3 / opus / pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed multiplier")
    # ── 扩展测试字段 ──
    reference_audio: Optional[str] = Field(default=None, description="Base64 encoded reference audio for cloning")
    prompt_text: Optional[str] = Field(default=None, description="Transcript of reference/prompt audio")
    reference_wav_path: Optional[str] = Field(default=None, description="Path to reference audio for cloning")
    instruction: Optional[str] = Field(default=None, description="Voice design instruction (free-form text)")
    temperature: float = Field(default=1.1, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Nucleus sampling threshold")
    repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0, description="Repetition penalty")
    # ── session 自举克隆字段 ──
    session_id: Optional[str] = Field(default=None, description="Session ID for bootstrap cloning across segments")
    session_action: Optional[str] = Field(default=None, description="start|continue|end (auto-detected if omitted)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_mono_wav(wav_path: str) -> tuple:
    """加载音频并转为单声道 tensor。"""
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio).transpose(0, 1).contiguous()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.contiguous(), int(sr)


def _maybe_resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_sr, target_sr)


def _ensure_speaker_tag(text: str, speaker_id: int = 1) -> str:
    expected_tag = f"[S{speaker_id}]"
    if not text.lstrip().startswith(expected_tag):
        return f"{expected_tag} {text}"
    return text


def _merge_consecutive_speaker_tags(text: str) -> str:
    segments = re.split(r"(?=\[S\d+\])", text)
    if not segments:
        return text
    merged_parts = []
    current_tag = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        matched = re.match(r"^(\[S\d+\])\s*(.*)", seg, re.DOTALL)
        if not matched:
            merged_parts.append(seg)
            continue
        tag, content = matched.groups()
        if tag == current_tag:
            merged_parts.append(content)
        else:
            current_tag = tag
            merged_parts.append(f"{tag}{content}")
    return "".join(merged_parts)


# ---------------------------------------------------------------------------
# voice_design — MOSS-VoiceGenerator
# ---------------------------------------------------------------------------
def _build_voice_design(request: TTSRequest) -> tuple:
    """构建 voice_design 推理参数。"""
    if not request.input.strip():
        raise HTTPException(400, "input text is empty")
    if not request.instruction or not request.instruction.strip():
        raise HTTPException(400, "instruction is required for voice_design mode")

    model, processor, sr = _load_voicegen()

    conversations = [
        [processor.build_user_message(
            text=request.input.strip(),
            instruction=request.instruction.strip(),
        )],
    ]

    # VoiceGenerator 官方推荐参数
    kwargs = dict(
        conversations=conversations,
        mode="generation",
        temperature=request.temperature if request.temperature != 1.1 else 1.5,
        top_p=request.top_p if request.top_p != 0.9 else 0.6,
        top_k=50,
        repetition_penalty=request.repetition_penalty,
    )
    return kwargs, sr, "voice_design", []


def _run_voice_design(kwargs: dict, sample_rate: int) -> np.ndarray:
    """执行 voice_design 推理。"""
    model, processor, _ = _load_voicegen()
    batch = processor(kwargs["conversations"], mode="generation")
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logger.info("[INFER] mode=voice_design temperature=%.2f top_p=%.2f",
                    kwargs["temperature"], kwargs["top_p"])
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_temperature=kwargs["temperature"],
            audio_top_p=kwargs["top_p"],
            audio_top_k=kwargs["top_k"],
            audio_repetition_penalty=kwargs["repetition_penalty"],
        )

    audio_segments = []
    for message in processor.decode(outputs):
        for audio in message.audio_codes_list:
            audio_segments.append(audio.detach().cpu().to(torch.float32).numpy())

    if not audio_segments:
        raise HTTPException(500, "VoiceGenerator produced no audio output")

    result = np.concatenate(audio_segments, axis=-1)
    if result.ndim > 1:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# Session 自举克隆
# ---------------------------------------------------------------------------
def _resolve_session_action(request: TTSRequest) -> str:
    """推断 session action：start / continue / end / none。"""
    if not request.session_id:
        return "none"
    if request.session_action in ("start", "continue", "end"):
        return request.session_action
    # 自动推断：有缓存则 continue，否则 start
    return "continue" if request.session_id in _session_cache else "start"


def _save_session_ref(session_id: str, wav: np.ndarray, sr: int, prompt_text: str):
    """将生成的音频保存为 session 参考音频。"""
    _session_cache[session_id] = {
        "wav": wav,
        "sr": sr,
        "prompt_text": prompt_text,
    }
    logger.info("[SESSION %s] Cached reference audio (%.2fs)", session_id, len(wav) / sr)


def _build_ttsd_with_session(request: TTSRequest) -> tuple:
    """构建带 session 自举克隆的 TTS 推理参数。

    首次请求（session start）走 generation 模式，缓存输出。
    后续请求（session continue）用缓存的音频走 voice_clone 模式。
    """
    action = _resolve_session_action(request)

    if action == "continue" and request.session_id in _session_cache:
        session = _session_cache[request.session_id]
        ref_wav = session["wav"]
        ref_sr = session["sr"]
        ref_prompt = session["prompt_text"]
        logger.info("[SESSION %s] Using cached voice clone (ref %.2fs)",
                     request.session_id, len(ref_wav) / ref_sr)

        # 直接走 voice_clone 路径，不走 _build_ttsd
        model, processor, sr = _load_ttsd()
        text = request.input.strip()
        speaker_id = 1

        dialogue_text = _ensure_speaker_tag(text, speaker_id)

        wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0) if ref_wav.ndim == 1 else torch.from_numpy(ref_wav)
        wav_tensor = _maybe_resample(wav_tensor, ref_sr, sr)

        _set_audio_tokenizer_device(processor, DEVICE)
        encoded_wavs = processor.encode_audios_from_wav([wav_tensor], sampling_rate=sr)
        reference_codes: list = [None, None, None, None, None]
        reference_codes[speaker_id - 1] = encoded_wavs[0]
        _set_audio_tokenizer_device(processor, "cpu")
        _gpu_gc()

        conversations = [
            [processor.build_user_message(text=dialogue_text, reference=reference_codes)],
        ]

        kwargs = dict(
            conversations=conversations,
            mode="voice_clone",
            max_new_tokens=3000,
            temperature=1.1,
            top_p=0.9,
            top_k=50,
            repetition_penalty=request.repetition_penalty,
        )
        return kwargs, sr, "voice_clone", [], action

    # start 或 none：走正常的 _build_ttsd
    kwargs, sr, mode, temp_files = _build_ttsd(request)
    return kwargs, sr, mode, temp_files, action


def _build_ttsd(request: TTSRequest) -> tuple:
    """构建 tts / voice_clone 推理参数。Returns (kwargs, sample_rate, mode, temp_files)。"""
    if not request.input.strip():
        raise HTTPException(400, "input text is empty")

    model, processor, sr = _load_ttsd()
    temp_files = []

    # 解析参考音频
    ref_path = request.reference_wav_path
    if request.reference_audio:
        try:
            audio_bytes = base64.b64decode(request.reference_audio)
        except Exception:
            raise HTTPException(400, "reference_audio is not valid base64")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        temp_files.append(tmp.name)
        ref_path = tmp.name
        logger.info("[CLONE MODE] Wrote reference audio to temp file: %s", ref_path)
    elif request.reference_wav_path and not os.path.isfile(request.reference_wav_path):
        raise HTTPException(400, f"reference_wav_path not found: {request.reference_wav_path}")

    has_ref_audio = ref_path is not None
    has_prompt_text = request.prompt_text is not None and request.prompt_text.strip()
    text = request.input.strip()

    if has_ref_audio and has_prompt_text:
        # ── pure voice_clone ──
        speaker_id = 1
        dialogue_text = _ensure_speaker_tag(text, speaker_id)
        mode = "voice_clone"
        logger.info("[VOICE CLONE] text: '%s'", dialogue_text[:100])

        wav, orig_sr = _load_mono_wav(ref_path)
        wav = _maybe_resample(wav, orig_sr, sr)

        _set_audio_tokenizer_device(processor, DEVICE)
        encoded_wavs = processor.encode_audios_from_wav([wav], sampling_rate=sr)
        reference_codes: list = [None, None, None, None, None]
        reference_codes[speaker_id - 1] = encoded_wavs[0]
        _set_audio_tokenizer_device(processor, "cpu")
        _gpu_gc()

        conversations = [
            [processor.build_user_message(text=dialogue_text, reference=reference_codes)],
        ]
    else:
        # ── 基础 generation ──
        text = _ensure_speaker_tag(text)
        full_text = text
        mode = "tts"
        logger.info("[GENERATION] text: '%s...'", full_text[:80])

        conversations = [
            [processor.build_user_message(text=full_text)],
        ]

    # max_new_tokens 控制音频 token 数量，1 秒音频 ≈ 12.5 tokens
    # 根据文本长度动态估算：中文语速约 4 字/秒，加 padding
    is_clone = mode == "voice_clone"
    default_temp = 1.1 if is_clone else request.temperature
    default_top_p = 0.9 if is_clone else request.top_p
    padding = 2.0 if is_clone else 5.0
    estimated_sec = len(text) / 4.0 + padding
    default_max_tokens = max(int(estimated_sec * 12.5), 200)

    kwargs = dict(
        conversations=conversations,
        mode=mode,
        max_new_tokens=default_max_tokens,
        temperature=default_temp,
        top_p=default_top_p,
        top_k=50,
        repetition_penalty=request.repetition_penalty,
    )
    return kwargs, sr, mode, temp_files


def _run_ttsd(kwargs: dict, sample_rate: int) -> np.ndarray:
    """执行 MOSS-TTSD 推理（tts / voice_clone）。"""
    model, processor, _ = _load_ttsd()
    proc_mode = "generation"

    max_new_tokens = kwargs["max_new_tokens"]
    # voice_clone 模式可能生成极短音频，需要重试（与 demo_clone_v2.py 一致）
    is_clone = kwargs["mode"] == "voice_clone"
    max_attempts = 3 if is_clone else 1

    for attempt in range(max_attempts):
        batch = processor(kwargs["conversations"], mode=proc_mode)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        _set_audio_tokenizer_device(processor, "cpu")
        _gpu_gc()

        with torch.inference_mode():
            logger.info("[INFER] mode=%s proc_mode=%s temp=%.2f top_p=%.2f rp=%.2f attempt=%d/%d",
                        kwargs["mode"], proc_mode,
                        kwargs["temperature"], kwargs["top_p"], kwargs["repetition_penalty"],
                        attempt + 1, max_attempts)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=kwargs["temperature"],
                audio_top_p=kwargs["top_p"],
                audio_top_k=kwargs["top_k"],
                audio_repetition_penalty=kwargs["repetition_penalty"],
            )

        del input_ids, attention_mask, batch
        _set_audio_tokenizer_device(processor, DEVICE)

        audio_segments = []
        for message in processor.decode(outputs):
            for audio in message.audio_codes_list:
                seg = audio.detach().cpu().to(torch.float32).numpy()
                logger.info("[SEGMENT] shape=%s samples=%d max_abs=%.4f",
                            seg.shape, seg.shape[-1], float(np.abs(seg).max()))
                # 跳过极短片段（< 100 采样 ≈ 4ms @24kHz）
                if seg.shape[-1] > 100:
                    audio_segments.append(seg)

        _set_audio_tokenizer_device(processor, "cpu")
        del outputs
        _gpu_gc()

        if not audio_segments:
            if is_clone and attempt < max_attempts - 1:
                logger.warning("[INFER] No valid audio segments, retrying with +1000 tokens")
                max_new_tokens += 1000
                continue
            raise HTTPException(500, "Model produced no audio output")

        total_samples = sum(s.shape[-1] for s in audio_segments)
        logger.info("[INFER] %d valid segments, total %d samples (%.2fs @%dHz)",
                    len(audio_segments), total_samples,
                    total_samples / sample_rate, sample_rate)

        # 如果 clone 模式音频过短（< 0.5 秒），尝试重试
        if is_clone and total_samples < sample_rate * 0.5 and attempt < max_attempts - 1:
            logger.warning("[INFER] Clone audio too short (%.2fs), retrying", total_samples / sample_rate)
            max_new_tokens += 1000
            continue

        break

    result = np.concatenate(audio_segments, axis=-1)
    if result.ndim > 1:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/audio/speech")
def create_speech(request: TTSRequest):
    # 路由到不同模式
    if request.instruction:
        # voice_design: 有 instruction 字段 -> VoiceGenerator
        kwargs, sr, mode, temp_files = _build_voice_design(request)
        try:
            wav = _run_voice_design(kwargs, sr)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("voice_design failed: %s", e, exc_info=True)
            raise HTTPException(500, f"Generation failed: {e}")
    else:
        # tts / voice_clone: MOSS-TTSD（支持 session 自举克隆）
        kwargs, sr, mode, temp_files, session_action = _build_ttsd_with_session(request)
        try:
            wav = _run_ttsd(kwargs, sr)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Inference failed: %s", e, exc_info=True)
            raise HTTPException(500, f"Generation failed: {e}")
        finally:
            for p in temp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        # session: 首次生成后缓存参考音频
        if request.session_id and session_action in ("start", "continue"):
            _save_session_ref(request.session_id, wav.copy(), sr, request.input.strip())

        # session: 结束时清理缓存
        if request.session_id and session_action == "end":
            _session_cache.pop(request.session_id, None)
            logger.info("[SESSION %s] Cleaned up", request.session_id)

    buffer = io.BytesIO()
    sf.write(buffer, wav.T if wav.ndim > 1 else wav, sr, format="WAV")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")


@app.post("/v1/audio/speech/stream")
def create_speech_stream(request: TTSRequest):
    # MOSS-TTSD 不支持真正的逐 chunk 流式，先整体推理再模拟流式返回
    if request.instruction:
        kwargs, sr, mode, temp_files = _build_voice_design(request)
    else:
        kwargs, sr, mode, temp_files = _build_ttsd(request)

    def _stream():
        try:
            if mode == "voice_design":
                wav = _run_voice_design(kwargs, sr)
            else:
                wav = _run_ttsd(kwargs, sr)

            if wav.ndim > 1:
                wav = wav.squeeze(0)
            data = wav.astype(np.float32)
            chunk_samples = sr  # 每秒一个 chunk
            for i in range(0, len(data), chunk_samples):
                yield data[i:i + chunk_samples].tobytes()
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Streaming inference failed: %s", e, exc_info=True)
            raise
        finally:
            for p in temp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return StreamingResponse(
        _stream(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(sr), "X-Sample-Format": "float32"},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "MOSS-TTSD",
        "capabilities": ["tts", "voice_clone", "voice_design"],
        "sample_rate": _ttsd_sr or _voicegen_sr or 24000,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run MOSS-TTSD TTS Web API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8004)
    args, _ = parser.parse_known_args()
    uvicorn.run(app, host=args.host, port=args.port)
