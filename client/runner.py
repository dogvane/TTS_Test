"""评测客户端 — 遍历测试文本，以 OpenAI 兼容方式直连各引擎 webapi 生成音频并输出 HTML 报告。

各引擎需先按 tts/<引擎>/readme.md 手动启动（见 client/config.yaml 中的 base_url）。
根据模型能力自动执行：基础 TTS、音色设计、语音克隆等测试。

用法:
    python -m client.runner --engine voxcpm
    python -m client.runner --engine voxcpm --test T02
"""

import csv
import html
import io
import logging
import numpy as np
import os
import re
import sys
import time
from datetime import datetime

import requests
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENGINES_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_engine_config(engine: str) -> dict:
    """从 client/config.yaml 读取引擎配置（base_url / model / capabilities）。"""
    import yaml
    with open(ENGINES_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    engines = cfg.get("engines", {})
    if engine not in engines:
        available = ", ".join(sorted(engines))
        logger.error("Unknown engine '%s'. Available engines: %s (see client/config.yaml)", engine, available)
        sys.exit(1)
    return engines[engine]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def read_test_text(test_id: str) -> str:
    texts_dir = os.path.join(PROJECT_ROOT, "test_texts")
    for f in os.listdir(texts_dir):
        if f.startswith(test_id) and f.endswith(".txt"):
            with open(os.path.join(texts_dir, f), "r", encoding="utf-8") as fh:
                return fh.read().strip()
    raise FileNotFoundError(f"No test text found for {test_id}")


def read_voice_design_csv() -> list[dict]:
    """读取音色设计 CSV，返回 [{voice: "描述", text: "文本"}, ...]。"""
    texts_dir = os.path.join(PROJECT_ROOT, "test_texts")
    for f in os.listdir(texts_dir):
        if f.startswith("T07") and f.endswith(".csv"):
            rows = []
            with open(os.path.join(texts_dir, f), "r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                for line in reader:
                    if len(line) >= 2 and line[0].strip() and line[1].strip():
                        rows.append({"voice": line[0].strip(), "text": line[1].strip()})
            return rows
    return []


def read_voice_clone_csv(prefix: str = "T08") -> list[dict]:
    """读取语音克隆 CSV，返回 [{ref_path: "路径", text: "文本"}, ...]。

    T08（基础克隆）和 T09（多音色克隆文本）结构相同（2 字段：参考音频,文本），
    通过 prefix 参数区分。
    """
    texts_dir = os.path.join(PROJECT_ROOT, "test_texts")
    for f in os.listdir(texts_dir):
        if f.startswith(prefix) and f.endswith(".csv"):
            rows = []
            with open(os.path.join(texts_dir, f), "r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                for line in reader:
                    if len(line) >= 2 and line[0].strip() and line[1].strip():
                        ref_path = line[0].strip()
                        if not os.path.isabs(ref_path):
                            ref_path = os.path.join(PROJECT_ROOT, ref_path)
                        rows.append({"ref_path": ref_path, "text": line[1].strip()})
            return rows
    return []


def read_emotion_clone_csv() -> list[dict]:
    """读取情绪克隆 CSV（T06），返回 [{ref_path, emotion, text}, ...]。

    CSV 格式为 3 字段：参考音频路径, 情绪, 说话内容。
    """
    texts_dir = os.path.join(PROJECT_ROOT, "test_texts")
    for f in os.listdir(texts_dir):
        if f.startswith("T06") and f.endswith(".csv"):
            rows = []
            with open(os.path.join(texts_dir, f), "r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                for line in reader:
                    if len(line) >= 3 and line[0].strip() and line[1].strip() and line[2].strip():
                        ref_path = line[0].strip()
                        if not os.path.isabs(ref_path):
                            ref_path = os.path.join(PROJECT_ROOT, ref_path)
                        rows.append({
                            "ref_path": ref_path,
                            "emotion": line[1].strip(),
                            "text": line[2].strip(),
                        })
            return rows
    return []


def list_test_texts() -> list[str]:
    texts_dir = os.path.join(PROJECT_ROOT, "test_texts")
    ids = set()
    for f in os.listdir(texts_dir):
        if f.endswith(".txt"):
            ids.add(f.split("_")[0])
    return sorted(ids)


def check_health(engine_cfg: dict) -> dict:
    """请求引擎自身 /health，返回健康信息；不可达时直接退出并给出启动提示。"""
    base_url = engine_cfg["base_url"].rstrip("/")
    engine_dir = engine_cfg.get("engine_dir", "tts/<engine>")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error("Engine webapi not available at %s: %s", base_url, e)
        logger.error("请先按 %s/readme.md 启动该引擎的 webapi，再运行测试。", engine_dir)
        sys.exit(1)


def synthesize(engine_cfg: dict, text: str, voice: str = "default",
               reference_wav_bytes: bytes | None = None,
               prompt_text: str | None = None,
               emotion: str | None = None,
               session_id: str | None = None,
               session_action: str | None = None) -> tuple[bytes, dict]:
    import base64
    payload = {
        "model": engine_cfg["model"],
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }
    if reference_wav_bytes:
        payload["reference_audio"] = base64.b64encode(reference_wav_bytes).decode("ascii")
        # 如果有 reference audio 且有 prompt_text，也传递 prompt_text
        if prompt_text:
            payload["prompt_text"] = prompt_text
    if emotion:
        payload["emotion"] = emotion
    if session_id:
        payload["session_id"] = session_id
        if session_action:
            payload["session_action"] = session_action

    t0 = time.perf_counter()
    base_url = engine_cfg["base_url"].rstrip("/")
    resp = requests.post(f"{base_url}/v1/audio/speech", json=payload, timeout=300)
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        logger.error("HTTP %d: %s", resp.status_code, resp.text)
        resp.raise_for_status()
    audio_bytes = resp.content

    wav, sr = sf.read(io.BytesIO(audio_bytes))
    duration = len(wav) / sr

    metrics = {
        "total_time": round(elapsed, 3),
        "audio_duration": round(duration, 3),
        "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
        "sample_rate": sr,
        "text_length": len(text),
    }
    return audio_bytes, metrics


def split_text_to_segments(text: str, max_chars: int = 100) -> list[str]:
    """将文本按行分割，超长行再按断句符号分割。"""
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            segments.append(line)
        else:
            # 按断句符号分割，保留标点
            parts = re.split(r'(?<=[。！？；!?])\s*', line)
            parts = [p.strip() for p in parts if p.strip()]
            segments.extend(parts)
    return segments


def synthesize_segments(engine_cfg: dict, text: str,
                        max_chars: int = 100) -> tuple[bytes, dict]:
    """分段合成文本，合并所有音频片段为一个 WAV。使用 session 自举克隆保持音色一致。"""
    segments = split_text_to_segments(text, max_chars)
    if not segments:
        raise ValueError("No segments to synthesize")

    import uuid
    session_id = f"seg_{uuid.uuid4().hex[:8]}"
    logger.info("  session_id=%s (%d segments)", session_id, len(segments))

    total_tts_time = 0.0
    wav_arrays = []
    sr = None

    for i, seg in enumerate(segments):
        if len(segments) == 1:
            action = "end"  # 单段不需要 session
        elif i == 0:
            action = "start"
        elif i == len(segments) - 1:
            action = "end"
        else:
            action = "continue"
        logger.info("  segment %d/%d (%d chars) action=%s...",
                     i + 1, len(segments), len(seg), action)
        audio_bytes, metrics = synthesize(engine_cfg, seg,
                                          session_id=session_id,
                                          session_action=action)
        total_tts_time += metrics["total_time"]

        wav, seg_sr = sf.read(io.BytesIO(audio_bytes))
        if sr is None:
            sr = seg_sr
        wav_arrays.append(wav)

    # 合并所有音频片段
    combined = np.concatenate(wav_arrays)
    duration = len(combined) / sr

    # 写回 WAV bytes
    buf = io.BytesIO()
    sf.write(buf, combined, sr, format='WAV')
    buf.seek(0)
    merged_bytes = buf.read()

    merged_metrics = {
        "total_time": round(total_tts_time, 3),
        "audio_duration": round(duration, 3),
        "rtf": round(total_tts_time / duration, 3) if duration > 0 else 0,
        "sample_rate": sr,
        "text_length": len(text),
    }
    return merged_bytes, merged_metrics


def generate_srt(text: str, audio_duration: float) -> str:
    sentences = re.split(r'(?<=[。！？!?])\s*|\n', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    total_chars = sum(len(s) for s in sentences)
    srt_lines = []
    current_time = 0.0

    for i, sentence in enumerate(sentences):
        ratio = len(sentence) / total_chars if total_chars > 0 else 1 / len(sentences)
        segment_duration = audio_duration * ratio
        start = _format_srt_time(current_time)
        end = _format_srt_time(current_time + segment_duration)
        srt_lines.extend([f"{i + 1}", f"{start} --> {end}", sentence, ""])
        current_time += segment_duration

    return "\n".join(srt_lines)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# 保存文件
# ---------------------------------------------------------------------------
def save_wav(run_dir: str, filename: str, audio_bytes: bytes) -> str:
    path = os.path.join(run_dir, f"{filename}.wav")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return f"{filename}.wav"


def save_srt(run_dir: str, filename: str, text: str, duration: float) -> str:
    path = os.path.join(run_dir, f"{filename}.srt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(generate_srt(text, duration))
    return f"{filename}.srt"


# ---------------------------------------------------------------------------
# HTML 报告生成
# ---------------------------------------------------------------------------
def generate_html_report(engine: str, model_info: dict, run_dir: str,
                         sections: list[dict], start_time: datetime,
                         init_info: dict | None = None) -> str:
    end_time = datetime.now()
    duration_str = f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_time.strftime('%H:%M:%S')}"

    capabilities = model_info.get("capabilities", [])
    voices = model_info.get("voices", [])
    clone_voices = model_info.get("clone_voices", [])

    # 统计
    total_tests = sum(len(s.get("rows", [])) for s in sections)
    all_metrics = []
    for s in sections:
        for r in s.get("rows", []):
            if "rtf" in r:
                all_metrics.append(r)

    avg_rtf = sum(r["rtf"] for r in all_metrics) / len(all_metrics) if all_metrics else 0
    avg_time = sum(r["total_time"] for r in all_metrics) / len(all_metrics) if all_metrics else 0
    total_audio = sum(r["audio_duration"] for r in all_metrics if "audio_duration" in r)

    # 构建 HTML
    h = [
        '<!DOCTYPE html>',
        '<html lang="zh-CN">',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>TTS 评测报告 - ' + engine + '</title>',
        '<style>',
        'body { font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #f5f5f5; }',
        'h1 { color: #333; border-bottom: 2px solid #4a9eff; padding-bottom: 8px; }',
        'h2 { color: #555; margin-top: 32px; border-left: 4px solid #4a9eff; padding-left: 10px; }',
        'h3 { color: #666; }',
        '.meta { color: #888; font-size: 14px; margin-bottom: 20px; }',
        '.summary { display: flex; gap: 16px; margin: 16px 0; }',
        '.summary-card { background: white; border-radius: 8px; padding: 16px; flex: 1; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
        '.summary-card .label { color: #888; font-size: 13px; }',
        '.summary-card .value { font-size: 24px; font-weight: bold; color: #333; }',
        '.cap-list span { display: inline-block; background: #e8f4ff; color: #1a73e8; padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 13px; }',
        '.test-item { background: white; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
        '.test-item .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }',
        '.test-item .id { font-weight: bold; color: #333; }',
        '.test-item .metrics { font-size: 13px; color: #888; }',
        '.test-item .text { background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 8px 0; font-size: 14px; line-height: 1.6; white-space: pre-wrap; }',
        'audio { width: 100%; margin-top: 8px; }',
        '.voice-tag { display: inline-block; background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 8px; }',
        '.clone-ref { font-size: 13px; color: #666; margin: 4px 0; }',
        'table { width: 100%; border-collapse: collapse; margin: 12px 0; background: white; }',
        'th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }',
        'th { background: #fafafa; color: #666; font-weight: 600; }',
        '</style>',
        '</head>',
        '<body>',
        '<h1>TTS 评测报告</h1>',
        '<div class="meta">引擎: <strong>' + engine + '</strong> &nbsp;|&nbsp; ' + duration_str + '</div>',
        # 能力标签
        f'<div style="margin-bottom:12px">能力: ',
        ' '.join(f'<span>{c}</span>' for c in capabilities),
        '</div>',
        f'<div style="margin-bottom:12px; font-size:13px; color:#888">预设音色: {", ".join(voices)} &nbsp;|&nbsp; 克隆音色: {", ".join(clone_voices)}</div>',
    ]

    # 引擎连接信息
    if init_info:
        init_sr = init_info.get("sample_rate", "?")
        h.append('<div style="background:#fff; border-radius:8px; padding:12px 16px; margin:12px 0; box-shadow:0 1px 3px rgba(0,0,0,0.1)">')
        h.append('<span style="font-weight:600; color:#555">引擎</span> &nbsp; ')
        h.append('<span style="display:inline-block; background:#4caf50; color:#fff; padding:1px 8px; border-radius:4px; font-size:12px">在线</span>')
        if init_sr != "?":
            h.append(f' &nbsp; <span style="color:#888; font-size:13px">采样率 {init_sr}Hz</span>')
        h.append('</div>')

    h += [
        # 汇总卡片
        '<div class="summary">',
        f'<div class="summary-card"><div class="label">总测试数</div><div class="value">{total_tests}</div></div>',
        f'<div class="summary-card"><div class="label">平均 RTF</div><div class="value">{avg_rtf:.3f}</div></div>',
        f'<div class="summary-card"><div class="label">平均耗时</div><div class="value">{avg_time:.1f}s</div></div>',
        f'<div class="summary-card"><div class="label">总音频时长</div><div class="value">{total_audio:.1f}s</div></div>',
        '</div>',
    ]

    # 各测试段落
    for section in sections:
        h.append(f'<h2>{html.escape(section["title"])}</h2>')
        if section.get("description"):
            h.append(f'<p style="color:#888; font-size:14px">{html.escape(section["description"])}</p>')

        for row in section.get("rows", []):
            if row.get("error"):
                h.append(f'<div class="test-item"><div class="header"><span class="id">{html.escape(row["id"])}</span></div>'
                         f'<div style="color:red">失败: {html.escape(row["error"])}</div></div>')
                continue

            voice_html = f'<span class="voice-tag">{html.escape(row.get("voice", ""))}</span>' if row.get("voice") and row["voice"] != "default" else ""
            ref_html = f'<div class="clone-ref">参考音频: {html.escape(row.get("ref_audio", ""))}</div>' if row.get("ref_audio") else ""

            h.append(f'<div class="test-item">')
            h.append(f'<div class="header"><span class="id">{html.escape(row["id"])}{voice_html}</span>'
                     f'<span class="metrics">耗时 {row.get("total_time", "?")}s &nbsp;|&nbsp; 音频 {row.get("audio_duration", "?")}s &nbsp;|&nbsp; RTF {row.get("rtf", "?")}</span></div>')
            h.append(f'<div class="text">{html.escape(row.get("text", ""))}</div>')
            h.append(ref_html)
            if row.get("audio_file"):
                h.append(f'<audio controls src="{row["audio_file"]}"></audio>')
            h.append('</div>')

    h.append('</body></html>')

    report_path = os.path.join(run_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(h))
    return report_path


# ---------------------------------------------------------------------------
# 运行评测
# ---------------------------------------------------------------------------
def run(engine: str, test_ids: list[str] | None = None,
        base_url_override: str | None = None):
    logger.info("Engine: %s", engine)
    engine_cfg = load_engine_config(engine)
    if base_url_override:
        engine_cfg = {**engine_cfg, "base_url": base_url_override}

    # 检查引擎 webapi 是否在线
    health = check_health(engine_cfg)
    logger.info("[health] status=%s sample_rate=%s", health.get("status"), health.get("sample_rate"))
    init_info = health

    capabilities = engine_cfg.get("capabilities", [])
    model_info = {"id": engine_cfg.get("model", engine), "capabilities": capabilities,
                  "voices": [], "clone_voices": []}

    if test_ids is None:
        test_ids = list_test_texts()

    # 创建输出目录
    run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(PROJECT_ROOT, "results", engine, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)
    start_time = datetime.now()
    sections = []

    # ── 1. 基础 TTS 测试 ──
    if "tts" in capabilities and test_ids:
        logger.info("=== 基础 TTS 测试 (%d texts) ===", len(test_ids))
        rows = []
        for test_id in test_ids:
            text = read_test_text(test_id)
            logger.info("[%s] TTS (%d chars)...", test_id, len(text))
            try:
                audio_bytes, metrics = synthesize_segments(engine_cfg, text)
                wav_file = save_wav(run_dir, test_id, audio_bytes)
                save_srt(run_dir, test_id, text, metrics["audio_duration"])
                logger.info("[%s] Done (RTF=%.3f)", test_id, metrics["rtf"])
                rows.append({"id": test_id, "text": text, "audio_file": wav_file, "voice": "default", **metrics})
            except Exception as e:
                logger.error("[%s] Failed: %s", test_id, e)
                rows.append({"id": test_id, "text": text, "error": str(e)})
        sections.append({"title": "基础 TTS 测试", "description": "使用 default 音色合成所有测试文本", "rows": rows})

    # ── 2. 音色设计测试 ──
    if "voice_design" in capabilities:
        design_cases = read_voice_design_csv()
        if design_cases:
            logger.info("=== 音色设计测试 (%d cases) ===", len(design_cases))
            rows = []
            for i, case in enumerate(design_cases):
                voice_desc = case["voice"]
                design_text = case["text"]
                tag = f"v{i + 1}"
                file_id = f"design_{tag}"
                logger.info("[design/%s] voice='%s...' text='%s...'", tag, voice_desc[:15], design_text[:15])
                try:
                    audio_bytes, metrics = synthesize(engine_cfg, design_text, voice=voice_desc)
                    wav_file = save_wav(run_dir, file_id, audio_bytes)
                    save_srt(run_dir, file_id, design_text, metrics["audio_duration"])
                    logger.info("[design/%s] Done (RTF=%.3f)", tag, metrics["rtf"])
                    rows.append({"id": f"design/{tag}", "text": design_text, "audio_file": wav_file, "voice": voice_desc, **metrics})
                except Exception as e:
                    logger.error("[design/%s] Failed: %s", tag, e)
                    rows.append({"id": f"design/{tag}", "text": design_text, "voice": voice_desc, "error": str(e)})
            sections.append({"title": "音色设计测试", "description": "使用 T07_voice_design.csv 中定义的音色描述和文本", "rows": rows})
        else:
            logger.info("跳过音色设计测试: T07_voice_design.csv 不存在或为空")

    # ── 3. 语音克隆测试 ──
    if "voice_clone" in capabilities:
        clone_cases = read_voice_clone_csv()
        if clone_cases:
            logger.info("=== 语音克隆测试 (%d cases) ===", len(clone_cases))
            rows = []
            for i, case in enumerate(clone_cases):
                ref_path = case["ref_path"]
                clone_text = case["text"]
                tag = f"c{i + 1}"
                file_id = f"clone_{tag}"
                ref_filename = os.path.basename(ref_path)
                logger.info("[clone/%s] ref='%s' text='%s...'", tag, ref_filename, clone_text[:15])
                try:
                    with open(ref_path, "rb") as f:
                        ref_bytes = f.read()
                    # 使用参考音频文件名（不含扩展名）作为 prompt_text，实现终极克隆
                    prompt_text = os.path.splitext(os.path.basename(ref_path))[0]
                    audio_bytes, metrics = synthesize(
                        engine_cfg, clone_text,
                        reference_wav_bytes=ref_bytes,
                        prompt_text=prompt_text
                    )
                    wav_file = save_wav(run_dir, file_id, audio_bytes)
                    save_srt(run_dir, file_id, clone_text, metrics["audio_duration"])
                    logger.info("[clone/%s] Done (RTF=%.3f)", tag, metrics["rtf"])
                    rows.append({"id": f"clone/{tag}", "text": clone_text, "audio_file": wav_file,
                                 "ref_audio": ref_filename, **metrics})
                except Exception as e:
                    logger.error("[clone/%s] Failed: %s", tag, e)
                    rows.append({"id": f"clone/{tag}", "text": clone_text, "ref_audio": ref_filename, "error": str(e)})
            sections.append({"title": "语音克隆测试", "description": "使用 T08_voice_clone.csv 中定义的参考音频和文本", "rows": rows})
        else:
            logger.info("跳过克隆测试: T08_voice_clone.csv 不存在或为空")
            sections.append({"title": "语音克隆测试", "description": "跳过: T08_voice_clone.csv 中无测试用例", "rows": []})

    # ── 4. 多音色克隆文本测试（T09）──
    if "voice_clone" in capabilities:
        t09_cases = read_voice_clone_csv(prefix="T09")
        if t09_cases:
            logger.info("=== 多音色克隆文本测试 T09 (%d cases) ===", len(t09_cases))
            rows = []
            for i, case in enumerate(t09_cases):
                ref_path = case["ref_path"]
                clone_text = case["text"]
                tag = f"t{i + 1}"
                file_id = f"t09_{tag}"
                ref_filename = os.path.basename(ref_path)
                logger.info("[t09/%s] ref='%s' text='%s...'", tag, ref_filename, clone_text[:15])
                try:
                    with open(ref_path, "rb") as f:
                        ref_bytes = f.read()
                    prompt_text = os.path.splitext(os.path.basename(ref_path))[0]
                    audio_bytes, metrics = synthesize(
                        engine_cfg, clone_text,
                        reference_wav_bytes=ref_bytes,
                        prompt_text=prompt_text,
                    )
                    wav_file = save_wav(run_dir, file_id, audio_bytes)
                    save_srt(run_dir, file_id, clone_text, metrics["audio_duration"])
                    logger.info("[t09/%s] Done (RTF=%.3f)", tag, metrics["rtf"])
                    rows.append({"id": f"t09/{tag}", "text": clone_text, "audio_file": wav_file,
                                 "ref_audio": ref_filename, **metrics})
                except Exception as e:
                    logger.error("[t09/%s] Failed: %s", tag, e)
                    rows.append({"id": f"t09/{tag}", "text": clone_text, "ref_audio": ref_filename, "error": str(e)})
            sections.append({"title": "多音色克隆文本测试（T09）",
                             "description": "T09_voice_clone_texts.csv：4 音色 × 纯中文/术语/数字/长文本",
                             "rows": rows})
        else:
            logger.info("跳过 T09: T09_voice_clone_texts.csv 不存在或为空")

    # ── 5. 情绪克隆测试（T06）──
    if "emotion" in capabilities:
        emo_cases = read_emotion_clone_csv()
        if emo_cases:
            logger.info("=== 情绪克隆测试 T06 (%d cases) ===", len(emo_cases))
            rows = []
            for i, case in enumerate(emo_cases):
                ref_path = case["ref_path"]
                emotion = case["emotion"]
                emo_text = case["text"]
                tag = f"e{i + 1}"
                file_id = f"t06_{tag}"
                ref_filename = os.path.basename(ref_path)
                logger.info("[t06/%s] emotion='%s' ref='%s'", tag, emotion, ref_filename)
                try:
                    with open(ref_path, "rb") as f:
                        ref_bytes = f.read()
                    prompt_text = os.path.splitext(os.path.basename(ref_path))[0]
                    audio_bytes, metrics = synthesize(
                        engine_cfg, emo_text,
                        reference_wav_bytes=ref_bytes,
                        prompt_text=prompt_text,
                        emotion=emotion,
                    )
                    wav_file = save_wav(run_dir, file_id, audio_bytes)
                    save_srt(run_dir, file_id, emo_text, metrics["audio_duration"])
                    logger.info("[t06/%s] Done (RTF=%.3f)", tag, metrics["rtf"])
                    rows.append({"id": f"t06/{tag}", "text": emo_text, "audio_file": wav_file,
                                 "ref_audio": ref_filename, "voice": emotion, **metrics})
                except Exception as e:
                    logger.error("[t06/%s] Failed: %s", tag, e)
                    rows.append({"id": f"t06/{tag}", "text": emo_text, "ref_audio": ref_filename,
                                 "voice": emotion, "error": str(e)})
            sections.append({"title": "情绪克隆测试（T06）",
                             "description": "T06_emotion_clone.csv：单音色 × 6 情绪 × 同文本（兴奋/严肃/轻松/中性/悲伤/愤怒）",
                             "rows": rows})
        else:
            logger.info("跳过 T06: T06_emotion_clone.csv 不存在或为空")

    # ── 生成报告 ──
    total = sum(len(s.get("rows", [])) for s in sections)
    if total > 0:
        report_path = generate_html_report(engine_cfg.get('model', engine), model_info, run_dir, sections, start_time, init_info)
        logger.info("Report: %s", report_path)

    logger.info("Done. %d tests. Output: %s", total, run_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TTS Test Runner")
    parser.add_argument("--engine", default="voxcpm", help="引擎名称")
    parser.add_argument("--test", nargs="*", default=None, help="测试用例 ID，如 T01 T02")
    parser.add_argument("--base-url", default=None, help="覆盖 client/config.yaml 中的引擎地址")
    args = parser.parse_args()

    run(args.engine, args.test, base_url_override=args.base_url)
