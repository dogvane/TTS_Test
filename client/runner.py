"""评测客户端 — 遍历测试文本，调用网关生成音频并输出 HTML 报告。

根据模型能力自动执行：基础 TTS、音色设计、语音克隆三类测试。

用法:
    python -m client.runner
    python -m client.runner --engine voxcpm
    python -m client.runner --engine voxcpm --test T02
"""

import csv
import html
import io
import logging
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

GATEWAY_URL = "http://localhost:9000"


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


def read_voice_clone_csv() -> list[dict]:
    """读取语音克隆 CSV，返回 [{ref_path: "路径", text: "文本"}, ...]。"""
    texts_dir = os.path.join(PROJECT_ROOT, "test_texts")
    for f in os.listdir(texts_dir):
        if f.startswith("T08") and f.endswith(".csv"):
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


def list_test_texts() -> list[str]:
    texts_dir = os.path.join(PROJECT_ROOT, "test_texts")
    ids = set()
    for f in os.listdir(texts_dir):
        if f.endswith(".txt"):
            ids.add(f.split("_")[0])
    return sorted(ids)


def get_model_info(engine: str) -> dict | None:
    """从网关获取模型能力信息。"""
    try:
        resp = requests.get(f"{GATEWAY_URL}/v1/models", timeout=5)
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            if m["id"] == engine:
                return m
    except requests.RequestException:
        pass
    return None


def synthesize(engine: str, text: str, voice: str = "default",
               reference_wav_bytes: bytes | None = None) -> tuple[bytes, dict]:
    import base64
    payload = {
        "model": engine,
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }
    if reference_wav_bytes:
        payload["reference_audio"] = base64.b64encode(reference_wav_bytes).decode("ascii")

    t0 = time.perf_counter()
    resp = requests.post(f"{GATEWAY_URL}/v1/audio/speech", json=payload, timeout=300)
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
                         sections: list[dict], start_time: datetime) -> str:
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
def run(engine: str, test_ids: list[str] | None = None):
    logger.info("Engine: %s", engine)

    # 检查网关
    try:
        resp = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        logger.error("Gateway not available at %s. Start it first: python -m server.main", GATEWAY_URL)
        sys.exit(1)

    # 获取模型能力
    model_info = get_model_info(engine)
    if not model_info:
        logger.warning("Cannot get model info for '%s', using defaults", engine)
        model_info = {"id": engine, "capabilities": ["tts"], "voices": ["default"], "clone_voices": []}

    capabilities = model_info.get("capabilities", [])

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
                audio_bytes, metrics = synthesize(engine, text)
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
                    audio_bytes, metrics = synthesize(engine, design_text, voice=voice_desc)
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
                    audio_bytes, metrics = synthesize(engine, clone_text, reference_wav_bytes=ref_bytes)
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

    # ── 生成报告 ──
    total = sum(len(s.get("rows", [])) for s in sections)
    if total > 0:
        report_path = generate_html_report(engine, model_info, run_dir, sections, start_time)
        logger.info("Report: %s", report_path)

    logger.info("Done. %d tests. Output: %s", total, run_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TTS Test Runner")
    parser.add_argument("--engine", default="voxcpm", help="引擎名称")
    parser.add_argument("--test", nargs="*", default=None, help="测试用例 ID，如 T01 T02")
    parser.add_argument("--gateway", default=None, help="网关地址")
    args = parser.parse_args()

    if args.gateway:
        GATEWAY_URL = args.gateway

    run(args.engine, args.test)
