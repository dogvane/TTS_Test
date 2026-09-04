"""IndexTTS 2.5 Web API — OpenAI 兼容 /v1/audio/speech 端点。

本文件位于 TTS_Test 项目内，在 Windows 原生 venv（IndexTTS 的 .venv）下运行，
通过 sys.path 引用 IndexTTS 源码（indextts 包）。

启动（Windows，使用 IndexTTS 的 .venv）：
    cd O:\\ai\\TTS\\TTS_Test
    set PYTHONPATH=O:\\ai\\TTS\\TTS_Test;G:\\ai\\TTS\\index-tts\\IndexTTS-2.5\\index-tts
    python -m tts.IndexTTS25.webapi --port 8006

能力：
    - tts          基础合成（使用默认参考音频）
    - voice_clone  零样本克隆（reference_audio 提供参考音色）
    - emotion      情绪控制（emotion 字段 → Qwen 情绪模型转向量）
    - streaming    流式合成（stream_return=True）
"""

import argparse
import base64
import io
import logging
import os
import sys
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

# ---------------------------------------------------------------------------
# 加载配置，将 IndexTTS 源码加入 sys.path
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

_source_path = _cfg["source_path"]
if _source_path not in sys.path:
    sys.path.insert(0, _source_path)

from indextts.infer_v2 import IndexTTS2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 加载模型
# ---------------------------------------------------------------------------
app = FastAPI(title="IndexTTS 2.5 TTS API", description="OpenAI-compatible TTS API")

MODEL_DIR = _cfg["model_dir"]
CFG_PATH = _cfg["cfg_path"]
DEVICE = _cfg.get("device", "cuda")
USE_FP16 = _cfg.get("use_fp16", True)

logger.info("Loading IndexTTS2 from '%s' (cfg: %s, device: %s, fp16: %s)",
            MODEL_DIR, CFG_PATH, DEVICE, USE_FP16)
model = IndexTTS2(
    cfg_path=CFG_PATH,
    model_dir=MODEL_DIR,
    use_fp16=USE_FP16,
    device=DEVICE,
)
SAMPLE_RATE = 22050  # IndexTTS2 固定输出 22050Hz
logger.info("Model loaded. Sample rate: %d Hz", SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    # ── 标准 OpenAI 字段 ──
    model: str = Field(default="IndexTTS2", description="Model name")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="保留字段，IndexTTS 通过参考音频定音色")
    response_format: str = Field(default="wav", description="Output format: wav / pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="语速倍率（通过文本预处理实现）")
    # ── 克隆字段 ──
    reference_audio: Optional[str] = Field(default=None, description="Base64 编码的参考音频（voice clone）")
    prompt_text: Optional[str] = Field(default=None, description="参考音频对应的文本转录（IndexTTS 不强制需要）")
    # ── 情绪字段（IndexTTS 2.5 特色）──
    emotion: Optional[str] = Field(default=None, description="情绪标签，如 兴奋/严肃/轻松/中性/悲伤/愤怒")
    # ── 采样参数 ──
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="采样温度")
    top_p: float = Field(default=0.8, ge=0.1, le=1.0, description="Nucleus sampling 阈值")
    # ── 流式 ──
    stream: bool = Field(default=False, description="是否流式返回 PCM")


# ---------------------------------------------------------------------------
# 参考音频：base64 -> 临时 wav 文件
# ---------------------------------------------------------------------------
# 默认参考音频（无克隆输入时使用 IndexTTS 自带样本，保证基础 TTS 可跑）
DEFAULT_PROMPT_AUDIO = os.path.join(_source_path, "examples", "sample_prompt.wav")
if not os.path.isfile(DEFAULT_PROMPT_AUDIO):
    # 退化为任意一个 examples 下的 wav
    _ex_dir = os.path.join(_source_path, "examples")
    if os.path.isdir(_ex_dir):
        for _f in os.listdir(_ex_dir):
            if _f.endswith(".wav"):
                DEFAULT_PROMPT_AUDIO = os.path.join(_ex_dir, _f)
                break
    if not os.path.isfile(DEFAULT_PROMPT_AUDIO):
        logger.warning("No default prompt audio found under %s/examples; base TTS will fail without reference_audio", _source_path)


def _decode_ref_to_tempfile(b64: str) -> str:
    """把 base64 参考音频写成临时 wav，返回路径。调用方负责删除。"""
    try:
        audio_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "reference_audio is not valid base64")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# 推理调用
# ---------------------------------------------------------------------------
def _build_infer_kwargs(request: TTSRequest, ref_path: str, output_path: str) -> dict:
    """构造 IndexTTS2.infer() 的参数。"""
    kwargs = dict(
        spk_audio_prompt=ref_path,
        text=request.input,
        output_path=output_path,
        # 采样参数透传到 generation_kwargs
        temperature=request.temperature,
        top_p=request.top_p,
    )

    # 情绪控制：emotion 字段非空 → 用 Qwen 情绪模型把情绪词转向量
    if request.emotion and request.emotion.strip():
        kwargs["use_emo_text"] = True
        kwargs["emo_text"] = request.emotion.strip()
        logger.info("[EMOTION] use_emo_text=True emo_text='%s'", request.emotion)

    return kwargs


def _read_wav_as_numpy(path: str) -> np.ndarray:
    """读取 wav 文件为 1D float32 numpy。"""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # 多声道取均值
    return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/audio/speech")
def create_speech(request: TTSRequest):
    if not request.input.strip():
        raise HTTPException(400, "input text is empty")

    # 选定参考音频：克隆模式用上传的，否则用默认
    temp_files = []
    if request.reference_audio:
        ref_path = _decode_ref_to_tempfile(request.reference_audio)
        temp_files.append(ref_path)
        logger.info("[CLONE MODE] using uploaded reference audio")
    else:
        ref_path = DEFAULT_PROMPT_AUDIO
        logger.info("[BASE  MODE] using default prompt audio: %s", ref_path)

    # 输出临时文件
    out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_tmp.close()
    temp_files.append(out_tmp.name)

    kwargs = _build_infer_kwargs(request, ref_path, out_tmp.name)
    try:
        result = model.infer(**kwargs)
    except Exception as e:
        logger.error("infer() failed: %s", e, exc_info=True)
        _cleanup(temp_files)
        raise HTTPException(500, f"Generation failed: {e}")

    # infer 非 stream 模式返回 output_path（字符串）
    if isinstance(result, str) and os.path.isfile(result):
        wav = _read_wav_as_numpy(result)
    elif isinstance(result, list) and result and isinstance(result[-1], tuple):
        # (sr, wav_data) 形式
        wav = np.asarray(result[-1][1], dtype=np.float32)
    else:
        # 兜底：直接读 output_path
        if not os.path.isfile(out_tmp.name):
            _cleanup(temp_files)
            raise HTTPException(500, "Generation produced no audio")
        wav = _read_wav_as_numpy(out_tmp.name)

    _cleanup(temp_files)

    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")


@app.post("/v1/audio/speech/stream")
def create_speech_stream(request: TTSRequest):
    """流式合成，逐块 yield float32 PCM。"""
    if not request.input.strip():
        raise HTTPException(400, "input text is empty")

    temp_files = []
    if request.reference_audio:
        ref_path = _decode_ref_to_tempfile(request.reference_audio)
        temp_files.append(ref_path)
    else:
        ref_path = DEFAULT_PROMPT_AUDIO

    out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_tmp.close()
    temp_files.append(out_tmp.name)

    kwargs = _build_infer_kwargs(request, ref_path, out_tmp.name)
    kwargs["stream_return"] = True

    def _stream():
        try:
            for chunk in model.infer(**kwargs):
                # infer_generator 流式 yield 的是 torch tensor (1, T) 或 None
                if chunk is None:
                    continue
                if isinstance(chunk, torch.Tensor):
                    arr = chunk.detach().cpu().float().numpy().reshape(-1)
                    yield arr.astype(np.float32).tobytes()
        finally:
            _cleanup(temp_files)

    return StreamingResponse(
        _stream(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Sample-Format": "float32"},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": "IndexTTS2", "sample_rate": SAMPLE_RATE}


def _cleanup(paths: list[str]):
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IndexTTS 2.5 Web API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8006)
    args, _ = parser.parse_known_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
