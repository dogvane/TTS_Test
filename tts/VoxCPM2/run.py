"""VoxCPM2 本地测试脚本。

不依赖 server.main、gateway 或 Web API，直接加载本地 VoxCPM2 模型执行测试。

示例:
    python -m tts.VoxCPM2.run
    python -m tts.VoxCPM2.run --test T02
    python -m tts.VoxCPM2.run --text "今天天气很好" --voice "温柔女声"
    python -m tts.VoxCPM2.run --text "测试克隆" --reference-audio reference_audio/sample.wav --prompt-text "参考音频文本"
"""

from __future__ import annotations

import argparse
import csv
import html
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "VoxCPM2_local"
DEFAULT_CAPABILITIES = ["tts", "voice_design", "voice_clone"]


def resolve_platform_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    if os.name == "nt":
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw_path)
        if match:
            drive = match.group(1).upper()
            tail = match.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{tail}")
    else:
        match = re.match(r"^([a-zA-Z]):[\\/](.*)$", raw_path)
        if match:
            drive = match.group(1).lower()
            tail = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}")

    return Path(raw_path)


def load_config() -> tuple[dict[str, Any], Path, Path]:
    import yaml

    config_path = HERE / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    source_path = resolve_platform_path(config.get("source_path"))
    model_path = resolve_platform_path(config.get("model_path"))

    if source_path is None or model_path is None:
        raise ValueError("config.yaml 缺少 source_path 或 model_path")

    if not source_path.is_dir():
        raise FileNotFoundError(f"VoxCPM 源码目录不存在: {source_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"VoxCPM 模型目录不存在: {model_path}")

    return config, source_path, model_path


def load_model(model_path: Path, source_path: Path):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

    from voxcpm import VoxCPM  # type: ignore

    t0 = time.perf_counter()
    logger.info("Loading VoxCPM2 model from %s", model_path)
    model = VoxCPM.from_pretrained(str(model_path), load_denoiser=False)
    elapsed = time.perf_counter() - t0
    sample_rate = int(model.tts_model.sample_rate)
    logger.info("Model loaded in %.1fs, sample_rate=%d", elapsed, sample_rate)
    return model, sample_rate, elapsed


def read_test_text(test_id: str) -> str:
    texts_dir = PROJECT_ROOT / "test_texts"
    for path in texts_dir.glob(f"{test_id}_*.txt"):
        return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"No test text found for {test_id}")


def list_test_texts() -> list[str]:
    texts_dir = PROJECT_ROOT / "test_texts"
    ids = {path.name.split("_")[0] for path in texts_dir.glob("T*.txt")}
    return sorted(ids)


def read_voice_design_csv() -> list[dict[str, str]]:
    texts_dir = PROJECT_ROOT / "test_texts"
    csv_path = texts_dir / "T07_voice_design.csv"
    if not csv_path.is_file():
        return []

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for line in reader:
            if len(line) >= 2 and line[0].strip() and line[1].strip():
                rows.append({"voice": line[0].strip(), "text": line[1].strip()})
    return rows


def read_voice_clone_csv() -> list[dict[str, str]]:
    texts_dir = PROJECT_ROOT / "test_texts"
    csv_path = texts_dir / "T08_voice_clone.csv"
    if not csv_path.is_file():
        return []

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for line in reader:
            if len(line) < 2 or not line[0].strip() or not line[1].strip():
                continue

            ref_path = Path(line[0].strip())
            if not ref_path.is_absolute():
                ref_path = PROJECT_ROOT / ref_path
            rows.append({"ref_path": str(ref_path), "text": line[1].strip()})
    return rows


def split_text_to_segments(text: str, max_chars: int = 100) -> list[str]:
    segments: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if len(line) <= max_chars:
            segments.append(line)
            continue

        parts = re.split(r"(?<=[。！？!?；;])\s*", line)
        parts = [part.strip() for part in parts if part.strip()]

        current = ""
        for part in parts:
            if not current:
                current = part
                continue

            if len(current) + len(part) <= max_chars:
                current += part
            else:
                segments.append(current)
                current = part

        if current:
            segments.append(current)

    return segments or [text.strip()]


def build_generate_kwargs(
    text: str,
    voice: str = "default",
    reference_wav_path: Path | None = None,
    prompt_text: str | None = None,
) -> dict[str, Any]:
    has_clone_input = bool(reference_wav_path or prompt_text)

    if has_clone_input:
        final_text = text
    elif voice and voice != "default":
        final_text = f"({voice}){text}"
    else:
        final_text = text

    ref_path: str | None = str(reference_wav_path) if reference_wav_path else None
    prompt_wav_path: str | None = None
    if ref_path and prompt_text:
        prompt_wav_path = ref_path
        ref_path = None

    return {
        "text": final_text,
        "reference_wav_path": ref_path,
        "prompt_wav_path": prompt_wav_path,
        "prompt_text": prompt_text,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "normalize": False,
        "denoise": False,
    }


def synthesize_once(
    model,
    sample_rate: int,
    text: str,
    voice: str = "default",
    reference_wav_path: Path | None = None,
    prompt_text: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import numpy as np

    kwargs = build_generate_kwargs(
        text=text,
        voice=voice,
        reference_wav_path=reference_wav_path,
        prompt_text=prompt_text,
    )

    t0 = time.perf_counter()
    wav = model.generate(**kwargs)
    elapsed = time.perf_counter() - t0

    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    duration = len(audio) / sample_rate if sample_rate > 0 else 0.0

    metrics = {
        "total_time": round(elapsed, 3),
        "audio_duration": round(duration, 3),
        "rtf": round(elapsed / duration, 3) if duration > 0 else 0.0,
        "sample_rate": sample_rate,
        "text_length": len(text),
    }
    return audio, metrics


def synthesize_segments(
    model,
    sample_rate: int,
    text: str,
    voice: str = "default",
    reference_wav_path: Path | None = None,
    prompt_text: str | None = None,
    max_chars: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    import numpy as np

    segments = split_text_to_segments(text, max_chars=max_chars)
    if not segments:
        raise ValueError("No segments to synthesize")

    all_audio: list[np.ndarray] = []
    total_tts_time = 0.0
    logger.info("Segments: %d", len(segments))

    for idx, segment in enumerate(segments, start=1):
        logger.info("  segment %d/%d (%d chars)", idx, len(segments), len(segment))
        segment_audio, metrics = synthesize_once(
            model=model,
            sample_rate=sample_rate,
            text=segment,
            voice=voice,
            reference_wav_path=reference_wav_path,
            prompt_text=prompt_text,
        )
        all_audio.append(segment_audio)
        total_tts_time += metrics["total_time"]

    merged_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
    duration = len(merged_audio) / sample_rate if sample_rate > 0 else 0.0
    merged_metrics = {
        "total_time": round(total_tts_time, 3),
        "audio_duration": round(duration, 3),
        "rtf": round(total_tts_time / duration, 3) if duration > 0 else 0.0,
        "sample_rate": sample_rate,
        "text_length": len(text),
    }
    return merged_audio, merged_metrics


def save_wav(run_dir: Path, filename: str, audio: np.ndarray, sample_rate: int) -> str:
    import soundfile as sf

    path = run_dir / f"{filename}.wav"
    sf.write(path, audio, sample_rate, format="WAV")
    return path.name


def generate_srt(text: str, audio_duration: float) -> str:
    sentences = re.split(r"(?<=[。！？!?])\s*|\n", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    total_chars = sum(len(s) for s in sentences)
    current_time = 0.0
    lines: list[str] = []

    for index, sentence in enumerate(sentences, start=1):
        ratio = len(sentence) / total_chars if total_chars > 0 else 1 / len(sentences)
        duration = audio_duration * ratio
        lines.extend(
            [
                str(index),
                f"{format_srt_time(current_time)} --> {format_srt_time(current_time + duration)}",
                sentence,
                "",
            ]
        )
        current_time += duration

    return "\n".join(lines)


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(run_dir: Path, filename: str, text: str, duration: float) -> str:
    path = run_dir / f"{filename}.srt"
    path.write_text(generate_srt(text, duration), encoding="utf-8")
    return path.name


def render_model_summary(config: dict[str, Any], sample_rate: int, load_time: float) -> str:
    source_path = html.escape(str(resolve_platform_path(config.get("source_path"))))
    model_path = html.escape(str(resolve_platform_path(config.get("model_path"))))
    return (
        '<div class="model-box">'
        f"<div><strong>sample_rate:</strong> {sample_rate} Hz</div>"
        f"<div><strong>load_time:</strong> {load_time:.1f}s</div>"
        f"<div><strong>model_path:</strong> {model_path}</div>"
        f"<div><strong>source_path:</strong> {source_path}</div>"
        "</div>"
    )


def generate_html_report(
    run_dir: Path,
    sections: list[dict[str, Any]],
    start_time: datetime,
    sample_rate: int,
    load_time: float,
    config: dict[str, Any],
) -> Path:
    end_time = datetime.now()
    duration_str = f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_time.strftime('%H:%M:%S')}"
    total_tests = sum(len(section.get("rows", [])) for section in sections)

    all_metrics = [row for section in sections for row in section.get("rows", []) if "rtf" in row]
    avg_rtf = sum(row["rtf"] for row in all_metrics) / len(all_metrics) if all_metrics else 0.0
    avg_time = sum(row["total_time"] for row in all_metrics) / len(all_metrics) if all_metrics else 0.0
    total_audio = sum(row["audio_duration"] for row in all_metrics) if all_metrics else 0.0

    body = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="UTF-8">',
        "<title>VoxCPM2 Local Test Report</title>",
        "<style>",
        "body { font-family: -apple-system, 'Microsoft YaHei', sans-serif; max-width: 1080px; margin: 0 auto; padding: 24px; background: #f5f7fb; color: #222; }",
        "h1 { margin-bottom: 8px; }",
        "h2 { margin-top: 32px; border-left: 4px solid #2b6fff; padding-left: 10px; }",
        ".meta { color: #667085; font-size: 14px; margin-bottom: 16px; }",
        ".summary { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }",
        ".card, .item, .model-box { background: #fff; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }",
        ".card { padding: 16px; min-width: 180px; flex: 1; }",
        ".card .label { color: #667085; font-size: 13px; }",
        ".card .value { font-size: 24px; font-weight: 700; margin-top: 4px; }",
        ".model-box { padding: 16px; line-height: 1.8; margin: 16px 0 20px; font-size: 14px; }",
        ".item { padding: 16px; margin: 12px 0; }",
        ".header { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; flex-wrap: wrap; }",
        ".metrics { color: #667085; font-size: 13px; }",
        ".text { background: #f8fafc; border-radius: 8px; padding: 10px 12px; margin-top: 10px; white-space: pre-wrap; line-height: 1.7; }",
        ".tag { display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 999px; background: #eef4ff; color: #2457d6; margin-left: 8px; }",
        ".ref { color: #667085; font-size: 13px; margin-top: 8px; }",
        "audio { width: 100%; margin-top: 10px; }",
        ".error { color: #c62828; margin-top: 8px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>VoxCPM2 本地测试报告</h1>",
        f'<div class="meta">运行时间: {html.escape(duration_str)}</div>',
        render_model_summary(config, sample_rate, load_time),
        '<div class="summary">',
        f'<div class="card"><div class="label">测试条数</div><div class="value">{total_tests}</div></div>',
        f'<div class="card"><div class="label">平均 RTF</div><div class="value">{avg_rtf:.3f}</div></div>',
        f'<div class="card"><div class="label">平均耗时</div><div class="value">{avg_time:.1f}s</div></div>',
        f'<div class="card"><div class="label">总音频时长</div><div class="value">{total_audio:.1f}s</div></div>',
        "</div>",
    ]

    for section in sections:
        body.append(f"<h2>{html.escape(section['title'])}</h2>")
        if section.get("description"):
            body.append(f'<div class="meta">{html.escape(section["description"])}</div>')

        for row in section.get("rows", []):
            body.append('<div class="item">')
            if row.get("error"):
                body.append(f"<div><strong>{html.escape(row['id'])}</strong></div>")
                body.append(f'<div class="text">{html.escape(row.get("text", ""))}</div>')
                body.append(f'<div class="error">失败: {html.escape(row["error"])}</div>')
                body.append("</div>")
                continue

            tag = ""
            if row.get("voice") and row["voice"] != "default":
                tag = f'<span class="tag">{html.escape(row["voice"])}</span>'

            body.append(
                '<div class="header">'
                f"<div><strong>{html.escape(row['id'])}</strong>{tag}</div>"
                f'<div class="metrics">耗时 {row["total_time"]}s | 音频 {row["audio_duration"]}s | RTF {row["rtf"]}</div>'
                "</div>"
            )
            body.append(f'<div class="text">{html.escape(row.get("text", ""))}</div>')
            if row.get("ref_audio"):
                body.append(f'<div class="ref">参考音频: {html.escape(row["ref_audio"])}</div>')
            if row.get("audio_file"):
                body.append(f'<audio controls src="{html.escape(row["audio_file"])}"></audio>')
            body.append("</div>")

    body.extend(["</body>", "</html>"])
    report_path = run_dir / "report.html"
    report_path.write_text("\n".join(body), encoding="utf-8")
    return report_path


def run_suite(
    model,
    sample_rate: int,
    test_ids: list[str] | None,
    output_dir: Path,
    max_chars: int,
    config: dict[str, Any],
    load_time: float,
) -> Path:
    if not test_ids:
        test_ids = list_test_texts()

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    sections: list[dict[str, Any]] = []

    if "tts" in DEFAULT_CAPABILITIES and test_ids:
        rows: list[dict[str, Any]] = []
        logger.info("=== 基础 TTS 测试 (%d texts) ===", len(test_ids))
        for test_id in test_ids:
            text = read_test_text(test_id)
            logger.info("[%s] TTS (%d chars)", test_id, len(text))
            try:
                audio, metrics = synthesize_segments(model, sample_rate, text, max_chars=max_chars)
                wav_file = save_wav(output_dir, test_id, audio, sample_rate)
                save_srt(output_dir, test_id, text, metrics["audio_duration"])
                rows.append({"id": test_id, "text": text, "audio_file": wav_file, "voice": "default", **metrics})
            except Exception as exc:
                logger.exception("[%s] Failed", test_id)
                rows.append({"id": test_id, "text": text, "error": str(exc)})
        sections.append({"title": "基础 TTS 测试", "description": "使用 test_texts/T*.txt 进行本地批量测试。", "rows": rows})

    if "voice_design" in DEFAULT_CAPABILITIES:
        design_cases = read_voice_design_csv()
        if design_cases:
            rows = []
            logger.info("=== 音色设计测试 (%d cases) ===", len(design_cases))
            for idx, case in enumerate(design_cases, start=1):
                tag = f"design_{idx}"
                voice = case["voice"]
                text = case["text"]
                logger.info("[%s] voice_design", tag)
                try:
                    audio, metrics = synthesize_segments(
                        model,
                        sample_rate,
                        text,
                        voice=voice,
                        max_chars=max_chars,
                    )
                    wav_file = save_wav(output_dir, tag, audio, sample_rate)
                    save_srt(output_dir, tag, text, metrics["audio_duration"])
                    rows.append({"id": tag, "text": text, "audio_file": wav_file, "voice": voice, **metrics})
                except Exception as exc:
                    logger.exception("[%s] Failed", tag)
                    rows.append({"id": tag, "text": text, "voice": voice, "error": str(exc)})
            sections.append({"title": "音色设计测试", "description": "使用 test_texts/T07_voice_design.csv 进行本地测试。", "rows": rows})

    if "voice_clone" in DEFAULT_CAPABILITIES:
        clone_cases = read_voice_clone_csv()
        if clone_cases:
            rows = []
            logger.info("=== 语音克隆测试 (%d cases) ===", len(clone_cases))
            for idx, case in enumerate(clone_cases, start=1):
                tag = f"clone_{idx}"
                ref_path = Path(case["ref_path"])
                text = case["text"]
                logger.info("[%s] clone from %s", tag, ref_path.name)
                try:
                    if not ref_path.is_file():
                        raise FileNotFoundError(f"Reference audio not found: {ref_path}")
                    prompt_text = ref_path.stem
                    audio, metrics = synthesize_segments(
                        model,
                        sample_rate,
                        text,
                        reference_wav_path=ref_path,
                        prompt_text=prompt_text,
                        max_chars=max_chars,
                    )
                    wav_file = save_wav(output_dir, tag, audio, sample_rate)
                    save_srt(output_dir, tag, text, metrics["audio_duration"])
                    rows.append(
                        {
                            "id": tag,
                            "text": text,
                            "audio_file": wav_file,
                            "ref_audio": ref_path.name,
                            **metrics,
                        }
                    )
                except Exception as exc:
                    logger.exception("[%s] Failed", tag)
                    rows.append({"id": tag, "text": text, "ref_audio": ref_path.name, "error": str(exc)})
            sections.append({"title": "语音克隆测试", "description": "使用 test_texts/T08_voice_clone.csv 进行本地测试。", "rows": rows})

    report_path = generate_html_report(
        run_dir=output_dir,
        sections=sections,
        start_time=start_time,
        sample_rate=sample_rate,
        load_time=load_time,
        config=config,
    )
    logger.info("Report written to %s", report_path)
    return report_path


def run_single(
    model,
    sample_rate: int,
    text: str,
    output_dir: Path,
    voice: str = "default",
    reference_audio: Path | None = None,
    prompt_text: str | None = None,
    max_chars: int = 100,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running single local test")
    audio, metrics = synthesize_segments(
        model=model,
        sample_rate=sample_rate,
        text=text,
        voice=voice,
        reference_wav_path=reference_audio,
        prompt_text=prompt_text,
        max_chars=max_chars,
    )
    wav_name = save_wav(output_dir, "single", audio, sample_rate)
    save_srt(output_dir, "single", text, metrics["audio_duration"])
    output_path = output_dir / wav_name
    logger.info("Single test finished: %s", output_path)
    logger.info("Metrics: total_time=%ss audio_duration=%ss rtf=%s", metrics["total_time"], metrics["audio_duration"], metrics["rtf"])
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VoxCPM2 local tests without gateway")
    parser.add_argument("--test", nargs="*", default=None, help="指定测试 ID，例如 T02")
    parser.add_argument("--text", default=None, help="单条文本测试内容")
    parser.add_argument("--voice", default="default", help="单条文本测试使用的音色描述")
    parser.add_argument("--reference-audio", default=None, help="单条文本测试的参考音频路径")
    parser.add_argument("--prompt-text", default=None, help="单条文本测试的参考音频文本")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认 results/VoxCPM2_local/<timestamp>")
    parser.add_argument("--max-chars", type=int, default=100, help="单段最大字符数，超长文本会自动分段")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, source_path, model_path = load_config()
    model, sample_rate, load_time = load_model(model_path=model_path, source_path=source_path)

    timestamp_dir = DEFAULT_RESULTS_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir) if args.output_dir else timestamp_dir

    if args.text:
        reference_audio = Path(args.reference_audio) if args.reference_audio else None
        if reference_audio and not reference_audio.is_absolute():
            reference_audio = PROJECT_ROOT / reference_audio
        run_single(
            model=model,
            sample_rate=sample_rate,
            text=args.text,
            output_dir=output_dir,
            voice=args.voice,
            reference_audio=reference_audio,
            prompt_text=args.prompt_text,
            max_chars=args.max_chars,
        )
        return

    report_path = run_suite(
        model=model,
        sample_rate=sample_rate,
        test_ids=args.test,
        output_dir=output_dir,
        max_chars=args.max_chars,
        config=config,
        load_time=load_time,
    )
    logger.info("Done. Output directory: %s", report_path.parent)


if __name__ == "__main__":
    main()
