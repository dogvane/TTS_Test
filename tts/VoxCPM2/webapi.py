"""VoxCPM2 TTS Web API — OpenAI 兼容 /v1/audio/speech 端点。

本文件位于 TTS_Test 项目内，通过 sys.path 引用外部 VoxCPM 源码。

启动:
    cd O:\ai\TTS\TTS_Test
    python -m tts.VoxCPM2.webapi --port 8000
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
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 加载配置，将 VoxCPM 源码加入 sys.path
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

_source_path = _cfg["source_path"]
if _source_path not in sys.path:
    sys.path.insert(0, _source_path)

from voxcpm import VoxCPM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 加载模型
# ---------------------------------------------------------------------------
app = FastAPI(title="VoxCPM2 TTS API", description="OpenAI-compatible TTS API")

MODEL_PATH = _cfg["model_path"]
logger.info("Loading VoxCPM2 from '%s' (source: %s)", MODEL_PATH, _source_path)
model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False)
SAMPLE_RATE = model.tts_model.sample_rate
logger.info("Model loaded. Sample rate: %d Hz", SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    model: str = Field(default="VoxCPM2", description="Model name")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(
        default="default",
        description="Voice description for Voice Design mode. Use 'default' for basic TTS.",
    )
    reference_wav_path: Optional[str] = Field(default=None, description="Path to reference audio for cloning.")
    reference_audio: Optional[str] = Field(default=None, description="Base64 encoded reference audio for cloning.")
    prompt_wav_path: Optional[str] = Field(default=None, description="Path to prompt audio for ultimate cloning.")
    prompt_text: Optional[str] = Field(default=None, description="Transcript of prompt audio.")
    response_format: str = Field(default="wav", description="Output format: wav")
    cfg_value: float = Field(default=2.0, ge=0.5, le=10.0)
    inference_timesteps: int = Field(default=10, ge=1, le=100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_and_build(request: TTSRequest) -> tuple[dict, list[str]]:
    """Validate request and build kwargs. Returns (kwargs, temp_files_to_cleanup)."""
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="input text is empty")

    text = request.input
    if request.voice and request.voice != "default":
        text = f"({request.voice}){text}"

    temp_files = []

    # 处理 base64 上传的参考音频
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
    elif request.reference_wav_path and not os.path.isfile(request.reference_wav_path):
        raise HTTPException(400, f"reference_wav_path not found: {request.reference_wav_path}")

    if request.prompt_wav_path:
        if not os.path.isfile(request.prompt_wav_path):
            raise HTTPException(400, f"prompt_wav_path not found: {request.prompt_wav_path}")
        if not request.prompt_text:
            raise HTTPException(400, "prompt_text is required when prompt_wav_path is provided")

    kwargs = dict(
        text=text,
        reference_wav_path=ref_path,
        prompt_wav_path=request.prompt_wav_path,
        prompt_text=request.prompt_text,
        cfg_value=request.cfg_value,
        inference_timesteps=request.inference_timesteps,
        normalize=False,
        denoise=False,
    )
    return kwargs, temp_files


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/audio/speech")
def create_speech(request: TTSRequest):
    kwargs, temp_files = _validate_and_build(request)
    try:
        wav: np.ndarray = model.generate(**kwargs)
    except Exception as e:
        logger.error("generate() failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Generation failed: {e}")
    finally:
        for p in temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass

    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")


@app.post("/v1/audio/speech/stream")
def create_speech_stream(request: TTSRequest):
    kwargs, temp_files = _validate_and_build(request)

    def _stream():
        try:
            for chunk in model.generate_streaming(**kwargs):
                yield chunk.astype(np.float32).tobytes()
        finally:
            for p in temp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return StreamingResponse(
        _stream(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Sample-Format": "float32"},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": "VoxCPM2", "sample_rate": SAMPLE_RATE}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run VoxCPM2 TTS Web API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()
    uvicorn.run(app, host=args.host, port=args.port)
