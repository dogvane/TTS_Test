"""Breeze-TTS 2 Web API — 评测统一协议 → Breeze OpenAI 接口的转换代理。

本进程不加载模型。请求翻译后转发给 Breeze TTS 2 的 OpenAI 兼容服务
（G:\\ai\\TTS\\Breeze-TTS-2，由 src/test_tts/start_server.bat 启动，默认 :8000）。

字段映射（评测统一协议 → Breeze）：
  - input                        → input
  - voice == "default"/空        → 删除（Breeze 服务端默认用第一个预设角色）
  - voice = 预设角色 id / S0… / 描述文本 → 原样透传（服务端分别按 预设克隆 /
    内置说话人 / 音色设计 处理；GET /v1/voices 获取角色列表）
  - reference_audio (b64)        → ref_audio_b64
  - prompt_text / 内置文字稿表    → ref_text（Breeze 要求参考音频的精确文字稿）
  - emotion                      → instruction="用{emotion}的情感演绎"（声音导演），cfg_scale=4
  - session_id / session_action  → 忽略（长文本各段独立合成；基础 TTS 传预设
    角色 id 即可跨段锁定音色）
  - response_format              → 原样传递（wav）

启动:
    cd /d O:\\ai\\TTS\\TTS_Test
    <python> -m tts.breeze-tts.webapi --port 8010
前提: G:\\ai\\TTS\\Breeze-TTS-2\\src\\test_tts\\start_server.bat 已启动。
"""

import argparse
import logging
import os

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("breeze-tts.webapi")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
BREEZE_URL = os.environ.get("BREEZE_URL", "http://127.0.0.1:8000")

# 参考音频精确文字稿表（key = 文件名去扩展名）。
# indextts/voice_XX.wav 来自 IndexTTS-2.5 官方 examples/cases.jsonl；
# 让子弹飞两段的文件名即台词，由 prompt_text 兜底。
REFERENCE_TRANSCRIPTS = {
    "voice_01": "Translate for me, what is a surprise!",
    "voice_02": "The palace is strict, no false rumors, Lady Qi!",
    "voice_03": "这个呀，就是我们精心制作准备的纪念品，大家可以看到这个色泽和这个材质啊，哎呀多么的光彩照人。",
    "voice_04": "你就需要我这种专业人士的帮助，就像手无缚鸡之力的人进入雪山狩猎，一定需要最老练的猎人指导。",
    "voice_05": "在真正的日本剑道中，格斗过程极其短暂，常常短至半秒，最长也不超过两秒，利剑相击的转瞬间，已有一方倒在血泊中。但在这电光石火的对决之前，双方都要以一个石雕般凝固的姿势站定，长时间的逼视对方，这一过程可能长达十分钟！",
    "voice_06": "今天呢，咱们开一部新书，叫《赛博朋克二零七七》。这词儿我听着都新鲜。这赛博朋克啊，简单理解就是\u201c高科技，低生活\u201d。这一听，我就明白了，于老师就爱用那高科技的东西，手机都得拿脚纹开，大冬天为了解锁脱得一丝不挂，冻得跟王八蛋似的。",
    "voice_07": "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。",
    "voice_08": "你看看你，对我还有没有一点父子之间的信任了。",
    "voice_09": "对不起嘛！我的记性真的不太好，但是和你在一起的事情，我都会努力记住的~",
    "voice_11": "这些年的时光终究是错付了...",
    "voice_12": "快躲起来！是他要来了！他要来抓我们了！",
}

# T06 情绪词 → Breeze 声音导演指令短语
EMOTION_PHRASES = {
    "兴奋": "兴奋饱满",
    "严肃": "严肃克制",
    "轻松": "轻松自然",
    "中性": "平静自然",
    "悲伤": "悲伤低沉",
    "愤怒": "愤怒有力",
}


class SpeechRequest(BaseModel):
    """评测统一协议字段 + Breeze 原生字段直通。"""

    model: str = "breeze-tts-2"
    input: str
    voice: str | None = None
    response_format: str = "wav"
    reference_audio: str | None = None   # base64
    prompt_text: str | None = None
    emotion: str | None = None
    session_id: str | None = None
    session_action: str | None = None
    # Breeze 原生（可选直通）
    instruction: str | None = None
    ref_audio_b64: str | None = None
    ref_text: str | None = None
    cfg_scale: float | None = None
    seed: int = 42


app = FastAPI(title="Breeze TTS 2 adapter for TTS_Test")


@app.get("/health")
def health():
    try:
        resp = requests.get(f"{BREEZE_URL}/health", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Breeze 服务不可达（{BREEZE_URL}）：{exc}。"
                   f"请先运行 G:\\ai\\TTS\\Breeze-TTS-2\\src\\test_tts\\start_server.bat",
        )
    info = resp.json()
    info["engine"] = "breeze-tts-2"
    info["backend"] = BREEZE_URL
    return info


@app.get("/v1/voices")
def voices():
    """转发 Breeze 的预设角色列表（含 default 角色）。"""
    try:
        resp = requests.get(f"{BREEZE_URL}/v1/voices", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"获取角色列表失败: {exc}")
    return resp.json()


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    payload: dict = {
        "model": "breeze-tts-2",
        "input": req.input,
        "response_format": req.response_format,
        "seed": req.seed,
    }

    # 参考音频（音色克隆 / 声音导演）
    ref_b64 = req.ref_audio_b64 or req.reference_audio
    if ref_b64:
        payload["ref_audio_b64"] = ref_b64
        # 文字稿优先级：显式 ref_text > 内置文字稿表 > prompt_text > 文件名兜底
        ref_text = req.ref_text
        if not ref_text and req.prompt_text:
            stem = req.prompt_text.strip()
            ref_text = REFERENCE_TRANSCRIPTS.get(stem, stem)
        payload["ref_text"] = ref_text or req.prompt_text or ""

    # voice 透传（"default"/空 删除），由 Breeze 服务端自行分类：
    # 预设角色 id → 预设克隆（音色锁定）；S0/S1… → 内置说话人；其他描述文本 → 音色设计
    voice = (req.voice or "").strip()
    if voice and voice.lower() != "default":
        payload["voice"] = voice

    # 指令：显式 instruction > emotion（导演）
    instruction = req.instruction
    if not instruction and req.emotion:
        phrase = EMOTION_PHRASES.get(req.emotion.strip(), req.emotion.strip())
        instruction = f"用{phrase}的情感演绎"
    if instruction:
        payload["instruction"] = instruction
        # 官方建议：指令场景 cfg=4 增强指令遵循
        payload["cfg_scale"] = req.cfg_scale if req.cfg_scale is not None else 4.0
    elif req.cfg_scale is not None:
        payload["cfg_scale"] = req.cfg_scale

    if req.session_id:
        logger.info("session %s action=%s（Breeze 无状态，独立合成该段）",
                    req.session_id, req.session_action or "-")

    logger.info("→ Breeze: %d chars, ref=%s, instruction=%s",
                len(req.input), bool(ref_b64),
                (payload.get("instruction") or "-")[:30])

    try:
        resp = requests.post(f"{BREEZE_URL}/v1/audio/speech", json=payload, timeout=1500)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Breeze 请求失败: {exc}")

    if resp.status_code != 200:
        detail = resp.text[:300]
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        logger.error("Breeze HTTP %d: %s", resp.status_code, detail)
        raise HTTPException(status_code=resp.status_code, detail=detail)

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "audio/wav"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("BREEZE_WEBAPI_PORT", "8010")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend", default=None, help="Breeze 服务地址（默认 http://127.0.0.1:8000）")
    args = parser.parse_args()

    global BREEZE_URL
    if args.backend:
        BREEZE_URL = args.backend

    logger.info("Breeze backend: %s", BREEZE_URL)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
