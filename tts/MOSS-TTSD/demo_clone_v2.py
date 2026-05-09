#!/usr/bin/env python3
"""
MOSS-TTSD 独立评测脚本
执行基础 TTS（T02）+ 语音克隆（T08），输出 HTML 报告。

用法（WSL2 conda 环境）:
    conda activate moss-tts
    cd /mnt/o/ai/TTS/TTS_Test
    python -m tts.MOSS-TTSD.demo_clone_v2
"""

import csv
import html
import io
import json
import os
import re
import time
import traceback
from datetime import datetime

import gc

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

# 配置参数（WSL2 路径格式）
MODEL_PATH = "/mnt/g/ai/TTS/MOSS-TTSD/models/MOSS-TTSD-v1.0"
CODEC_PATH = "/mnt/g/ai/TTS/MOSS-TTSD/models/MOSS-Audio-Tokenizer"
PROJECT_ROOT = "/mnt/o/ai/TTS/TTS_Test"

T02_TEXT_PATH = os.path.join(PROJECT_ROOT, "test_texts/T02_mixed_news.txt")
T08_CSV_PATH = os.path.join(PROJECT_ROOT, "test_texts/T08_voice_clone.csv")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/MOSS-TTSD")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32

TTS_MAX_CHARS = 100  # 每段最大字符数，超过则按断句分割

# 量化级别: "no" | "4bit" | "8bit"
QUANT_MODE = "no"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _ensure_speaker_tag(text: str, speaker_id: int = 1) -> str:
    expected_tag = f"[S{speaker_id}]"
    if not text.lstrip().startswith(expected_tag):
        return f"{expected_tag} {text}"
    return text


def _merge_consecutive_speaker_tags(text: str) -> str:
    segments = re.split(r"(?=\[S\d+\])", text)
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


def split_text_to_segments(text: str, max_chars: int = TTS_MAX_CHARS) -> list[str]:
    """将文本按行分割，超长行再按断句符号分割。"""
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            segments.append(line)
        else:
            parts = re.split(r'(?<=[。！？；!?])\s*', line)
            parts = [p.strip() for p in parts if p.strip()]
            segments.extend(parts)
    return segments


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_model_and_processor(quant: str = "no"):
    """加载 MOSS-TTSD 模型和处理器。

    quant: "no" (bf16, ~7.8GB) | "4bit" (~2.6GB) | "8bit" (~4.2GB)
    """
    print(f"[INFO] 加载模型: {MODEL_PATH}")
    print(f"[INFO] 设备: {DEVICE}, 精度: {DTYPE}, 量化: {quant}")

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, codec_path=CODEC_PATH,
    )
    if getattr(processor, "audio_tokenizer", None) is not None:
        processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)
        processor.audio_tokenizer.eval()

    # 量化配置
    quant_kwargs = {}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        quant_kwargs["device_map"] = DEVICE
    elif quant == "8bit":
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        quant_kwargs["device_map"] = DEVICE

    def _load_model(attn_impl: str):
        kwargs = dict(
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
        kwargs.update(quant_kwargs)
        model = AutoModel.from_pretrained(MODEL_PATH, **kwargs)
        if quant == "no":
            model = model.to(DEVICE)
        return model

    if DEVICE.startswith("cuda"):
        try:
            model = _load_model("flash_attention_2")
            print(f"[INFO] 使用 flash_attention_2")
        except Exception as e:
            print(f"[WARN] flash_attention_2 不可用，降级到 sdpa: {e}")
            model = _load_model("sdpa")
    else:
        model = _load_model("sdpa")

    model.eval()
    return model, processor


def _gpu_gc():
    """清理 GPU 显存碎片。"""
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 音频工具
# ---------------------------------------------------------------------------
def load_mono_wav(audio_path: str, target_sr: int) -> torch.Tensor:
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio).transpose(0, 1).contiguous()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.contiguous()
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def save_wav_numpy(run_dir: str, filename: str, audio: np.ndarray, sr: int) -> str:
    """保存 numpy 音频为 wav 文件，返回相对文件名。"""
    path = os.path.join(run_dir, f"{filename}.wav")
    sf.write(path, audio, sr)
    return f"{filename}.wav"


def concat_wav_numpy(arrays: list[np.ndarray], sr: int) -> tuple[np.ndarray, bytes]:
    """拼接多段 numpy 音频，返回 (numpy, wav_bytes)。"""
    combined = np.concatenate(arrays)
    buf = io.BytesIO()
    sf.write(buf, combined, sr, format='WAV')
    buf.seek(0)
    return combined, buf.read()


# ---------------------------------------------------------------------------
# CSV 读取
# ---------------------------------------------------------------------------
def read_clone_csv(csv_path: str) -> list[dict]:
    """读取 T08_voice_clone.csv，返回 [{ref_path, prompt_text, target_text}, ...]。"""
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                continue
            ref_rel = row[0].strip()
            ref_path = os.path.join(PROJECT_ROOT, ref_rel) if not os.path.isabs(ref_rel) else ref_rel
            if len(row) >= 3 and row[1].strip() and row[2].strip():
                prompt_text = row[1].strip()
                target_text = row[2].strip()
            else:
                prompt_text = ""
                target_text = row[1].strip()
            samples.append({
                "ref_path": ref_path,
                "prompt_text": prompt_text,
                "target_text": target_text,
            })
    return samples


# ---------------------------------------------------------------------------
# 基础 TTS 生成（generation 模式）
# ---------------------------------------------------------------------------
def tts_generate(model, processor, text: str,
                 target_sr: int, max_new_tokens: int = 0) -> np.ndarray:
    """基础 TTS generation 模式，返回音频 numpy 数组。

    生成期间 Audio-Tokenizer 移到 CPU，节省 ~3.3GB 显存。
    """
    if max_new_tokens <= 0:
        max_new_tokens = estimate_max_tokens(text, padding=5.0)

    full_text = _ensure_speaker_tag(text)

    conversations = [[processor.build_user_message(text=full_text)]]
    batch = processor(conversations, mode="generation")
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    # ── 生成阶段：Audio-Tokenizer 移到 CPU ──
    processor.audio_tokenizer = processor.audio_tokenizer.to("cpu")
    _gpu_gc()

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=1.1, audio_top_p=0.9,
                audio_top_k=50, audio_repetition_penalty=1.1,
            )

        # ── 解码阶段：Audio-Tokenizer 移回 GPU ──
        processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)

        audio_segments = []
        for message in processor.decode(outputs):
            n_codes = len(message.audio_codes_list)
            print(f"  [DECODE] {n_codes} audio codes")
            for audio in message.audio_codes_list:
                seg = audio.detach().cpu().to(torch.float32).numpy()
                print(f"  [SEGMENT] shape={seg.shape} samples={seg.shape[-1]}")
                if seg.shape[-1] > 100:
                    audio_segments.append(seg)

        del outputs, batch
        _gpu_gc()

        if not audio_segments:
            print("  [WARN] tts_generate: 无有效音频段")
            raise RuntimeError("TTS 生成无有效音频输出")

        result = np.concatenate(audio_segments, axis=-1)
        if result.ndim > 1:
            result = result.squeeze(0)
        return result
    except Exception:
        # 确保 Audio-Tokenizer 回到 GPU
        try:
            processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)
        except Exception:
            pass
        raise


def tts_generate_with_session(model, processor, segments: list[str],
                              target_sr: int) -> np.ndarray:
    """分段 TTS：第一段 generation，后续段用第一段作为参考音频做 voice_clone。

    利用 session 自举克隆保持说话人一致。
    如果 clone 段失败，跳过并继续下一段（不中断整体流程）。
    """
    if len(segments) == 1:
        return tts_generate(model, processor, segments[0], target_sr)

    # 第一段：generation
    print(f"  [seg 1/{len(segments)}] generation...")
    ref_audio = tts_generate(model, processor, segments[0], target_sr)
    ref_tensor = torch.from_numpy(ref_audio).unsqueeze(0) if ref_audio.ndim == 1 else torch.from_numpy(ref_audio)
    ref_prompt = segments[0].strip()

    wav_arrays = [ref_audio]

    # 后续段：voice_clone
    for i, seg in enumerate(segments[1:], start=2):
        try:
            print(f"  [seg {i}/{len(segments)}] clone...")
            clone_audio = clone_generate(model, processor, ref_tensor,
                                         ref_prompt, seg, target_sr)
            wav_arrays.append(clone_audio)
        except Exception as e:
            print(f"  [seg {i}/{len(segments)}] clone FAILED: {e}")
            # 跳过失败的段，继续后续段

    if len(wav_arrays) == 0:
        raise RuntimeError("所有分段生成均失败")

    combined, _ = concat_wav_numpy(wav_arrays, target_sr)
    return combined


# ---------------------------------------------------------------------------
# 语音克隆生成（voice_clone 模式）
# ---------------------------------------------------------------------------
def estimate_max_tokens(text: str,
                        chars_per_sec: float = 4.0, tokens_per_sec: float = 12.5,
                        padding: float = 2.0) -> int:
    """根据文本长度估算合理的 max_new_tokens。

    中文语速约 4 字/秒，音频 token 约 12.5 tokens/秒。
    """
    estimated_sec = len(text) / chars_per_sec + padding
    return max(int(estimated_sec * tokens_per_sec), 200)


def clone_generate(model, processor, ref_wav: torch.Tensor,
                   prompt_text: str, target_text: str,
                   target_sr: int, max_new_tokens: int = 0) -> np.ndarray:
    """执行单次 voice_clone 推理。max_new_tokens=0 时自动估算。

    分阶段使用 Audio-Tokenizer：编码后移到 CPU，生成完移回 GPU 解码。
    """
    if max_new_tokens <= 0:
        max_new_tokens = estimate_max_tokens(target_text)

    speaker_id = 1
    dialogue_text = _ensure_speaker_tag(target_text.strip(), speaker_id)

    # ── 编码阶段：Audio-Tokenizer 在 GPU ──
    encoded_wavs = processor.encode_audios_from_wav([ref_wav], sampling_rate=target_sr)
    reference_codes = [None, None, None, None, None]
    reference_codes[speaker_id - 1] = encoded_wavs[0]
    del encoded_wavs

    conversations = [
        [processor.build_user_message(text=dialogue_text, reference=reference_codes)],
    ]

    batch = processor(conversations, mode="generation")
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    # ── 生成阶段：Audio-Tokenizer 移到 CPU，释放 ~3.3GB 显存 ──
    processor.audio_tokenizer = processor.audio_tokenizer.to("cpu")
    _gpu_gc()

    current_max_tokens = max_new_tokens
    result = None
    try:
        for attempt in range(3):
            with torch.no_grad():
                print(f"  [INFER] attempt={attempt + 1}/3 max_new_tokens={current_max_tokens}")
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=current_max_tokens,
                    audio_temperature=1.1, audio_top_p=0.9,
                    audio_top_k=50, audio_repetition_penalty=1.1,
                )

            # ── 解码阶段：Audio-Tokenizer 移回 GPU ──
            processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)

            audio_segments = []
            for message in processor.decode(outputs):
                n_codes = len(message.audio_codes_list)
                print(f"  [CLONE-DECODE] {n_codes} audio codes")
                for audio in message.audio_codes_list:
                    seg = audio.detach().cpu().to(torch.float32).numpy()
                    print(f"  [CLONE-SEGMENT] shape={seg.shape} samples={seg.shape[-1]}")
                    if seg.shape[-1] > 100:
                        audio_segments.append(seg)

            # 解码完再次移走，为下次推理腾显存
            processor.audio_tokenizer = processor.audio_tokenizer.to("cpu")
            del outputs
            _gpu_gc()

            if not audio_segments:
                current_max_tokens += 1000
                continue

            result = np.concatenate(audio_segments, axis=-1)
            if result.ndim > 1:
                result = result.squeeze(0)

            duration = result.shape[-1] / target_sr
            if duration < 0.5 and attempt < 2:
                current_max_tokens += 1000
                continue

            del batch
            _gpu_gc()
            return result
    finally:
        # 确保 Audio-Tokenizer 回到 GPU（供后续调用使用）
        try:
            if processor.audio_tokenizer.device.type != DEVICE:
                processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)
        except Exception:
            processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)

    raise RuntimeError("克隆生成失败：多次重试后仍无有效音频")


# ---------------------------------------------------------------------------
# HTML 报告生成
# ---------------------------------------------------------------------------
def generate_report(engine: str, run_dir: str, sections: list[dict],
                    start_time: datetime) -> str:
    end_time = datetime.now()
    duration_str = f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_time.strftime('%H:%M:%S')}"

    all_metrics = []
    for s in sections:
        for r in s.get("rows", []):
            if "rtf" in r:
                all_metrics.append(r)

    total_tests = len(all_metrics)
    avg_rtf = sum(r["rtf"] for r in all_metrics) / len(all_metrics) if all_metrics else 0
    avg_time = sum(r["total_time"] for r in all_metrics) / len(all_metrics) if all_metrics else 0
    total_audio = sum(r["audio_duration"] for r in all_metrics if "audio_duration" in r)

    h = [
        '<!DOCTYPE html>',
        '<html lang="zh-CN">',
        '<head>',
        '<meta charset="UTF-8">',
        f'<title>TTS 评测报告 - {html.escape(engine)}</title>',
        '<style>',
        'body { font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #f5f5f5; }',
        'h1 { color: #333; border-bottom: 2px solid #4a9eff; padding-bottom: 8px; }',
        'h2 { color: #555; margin-top: 32px; border-left: 4px solid #4a9eff; padding-left: 10px; }',
        '.meta { color: #888; font-size: 14px; margin-bottom: 20px; }',
        '.summary { display: flex; gap: 16px; margin: 16px 0; }',
        '.summary-card { background: white; border-radius: 8px; padding: 16px; flex: 1; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
        '.summary-card .label { color: #888; font-size: 13px; }',
        '.summary-card .value { font-size: 24px; font-weight: bold; color: #333; }',
        '.test-item { background: white; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
        '.test-item .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }',
        '.test-item .id { font-weight: bold; color: #333; }',
        '.test-item .metrics { font-size: 13px; color: #888; }',
        '.test-item .text { background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 8px 0; font-size: 14px; line-height: 1.6; white-space: pre-wrap; }',
        'audio { width: 100%; margin-top: 8px; }',
        '.voice-tag { display: inline-block; background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 8px; }',
        '.clone-ref { font-size: 13px; color: #666; margin: 4px 0; }',
        '.seg-info { font-size: 12px; color: #999; margin: 4px 0; }',
        '</style>',
        '</head>',
        '<body>',
        '<h1>TTS 评测报告</h1>',
        f'<div class="meta">引擎: <strong>{html.escape(engine)}</strong> &nbsp;|&nbsp; {duration_str}</div>',
        '<div class="summary">',
        f'<div class="summary-card"><div class="label">总测试数</div><div class="value">{total_tests}</div></div>',
        f'<div class="summary-card"><div class="label">平均 RTF</div><div class="value">{avg_rtf:.3f}</div></div>',
        f'<div class="summary-card"><div class="label">平均耗时</div><div class="value">{avg_time:.1f}s</div></div>',
        f'<div class="summary-card"><div class="label">总音频时长</div><div class="value">{total_audio:.1f}s</div></div>',
        '</div>',
    ]

    for section in sections:
        h.append(f'<h2>{html.escape(section["title"])}</h2>')
        if section.get("description"):
            h.append(f'<p style="color:#888; font-size:14px">{html.escape(section["description"])}</p>')

        for row in section.get("rows", []):
            if row.get("error"):
                h.append(f'<div class="test-item"><div class="header"><span class="id">{html.escape(row["id"])}</span></div>'
                         f'<div style="color:red">失败: {html.escape(row["error"])}</div></div>')
                continue

            voice_html = f'<span class="voice-tag">{html.escape(row["voice"])}</span>' if row.get("voice") else ""
            ref_html = f'<div class="clone-ref">参考音频: {html.escape(row.get("ref_audio", ""))}</div>' if row.get("ref_audio") else ""
            seg_html = f'<div class="seg-info">分段数: {row["segments"]}</div>' if row.get("segments") else ""

            h.append('<div class="test-item">')
            h.append(f'<div class="header"><span class="id">{html.escape(row["id"])}{voice_html}</span>'
                     f'<span class="metrics">耗时 {row.get("total_time", "?")}s &nbsp;|&nbsp; '
                     f'音频 {row.get("audio_duration", "?")}s &nbsp;|&nbsp; '
                     f'RTF {row.get("rtf", "?")}</span></div>')
            h.append(f'<div class="text">{html.escape(row.get("text", ""))}</div>')
            h.append(ref_html)
            h.append(seg_html)
            if row.get("audio_file"):
                h.append(f'<audio controls src="{row["audio_file"]}"></audio>')
            h.append('</div>')

    h.append('</body></html>')

    report_path = os.path.join(run_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(h))
    return report_path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("[INFO] MOSS-TTSD 独立评测")
    start_time = datetime.now()

    # 创建输出目录
    run_dir_name = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RESULTS_ROOT, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # 加载模型
    model, processor = load_model_and_processor(quant=QUANT_MODE)
    target_sr = int(processor.model_config.sampling_rate)

    sections = []

    # ── 1. 基础 TTS 测试（T02）──
    print("\n" + "=" * 60)
    print("[TTS] 基础 TTS 测试 (T02)")
    print("=" * 60)

    with open(T02_TEXT_PATH, "r", encoding="utf-8") as f:
        t02_text = f.read().strip()

    segments = split_text_to_segments(t02_text)
    print(f"[T02] 共 {len(segments)} 段")

    tts_rows = []
    try:
        t0 = time.perf_counter()
        audio = tts_generate_with_session(model, processor, segments, target_sr)
        elapsed = time.perf_counter() - t0
        duration = len(audio) / target_sr
        rtf = elapsed / duration if duration > 0 else 0

        wav_file = save_wav_numpy(run_dir, "T02", audio, target_sr)
        print(f"[T02] Done: {duration:.2f}s audio, RTF={rtf:.3f}")
        tts_rows.append({
            "id": "T02", "text": t02_text, "audio_file": wav_file,
            "total_time": round(elapsed, 3),
            "audio_duration": round(duration, 3),
            "rtf": round(rtf, 3),
            "segments": len(segments),
        })
    except Exception as e:
        print(f"[T02] Failed: {e}")
        traceback.print_exc()
        tts_rows.append({"id": "T02", "text": t02_text, "error": str(e)})

    sections.append({
        "type": "tts",
        "title": "基础 TTS 测试",
        "description": "T02 文本按行分段合成，首段 generation 后续段 voice_clone 保持音色一致",
        "rows": tts_rows,
    })

    # ── 2. 语音克隆测试（T08）──
    print("\n" + "=" * 60)
    print("[CLONE] 语音克隆测试 (T08)")
    print("=" * 60)

    clone_samples = read_clone_csv(T08_CSV_PATH)
    print(f"[T08] 共 {len(clone_samples)} 条克隆任务")

    clone_rows = []
    ref_cache: dict[str, torch.Tensor] = {}

    for i, sample in enumerate(clone_samples):
        ref_path = sample["ref_path"]
        target_text = sample["target_text"]
        prompt_text = sample["prompt_text"]
        ref_filename = os.path.basename(ref_path)
        tag = f"c{i + 1}"

        print(f"\n[{tag}] ref={ref_filename} text='{target_text[:30]}...'")

        if not os.path.isfile(ref_path):
            print(f"[ERROR] 参考音频不存在: {ref_path}")
            clone_rows.append({"id": f"clone/{tag}", "text": target_text,
                               "ref_audio": ref_filename, "error": f"文件不存在: {ref_path}"})
            continue

        # 加载参考音频（带缓存）
        if ref_path not in ref_cache:
            ref_cache[ref_path] = load_mono_wav(ref_path, target_sr)
        ref_wav = ref_cache[ref_path]

        try:
            t0 = time.perf_counter()
            audio = clone_generate(model, processor, ref_wav,
                                   prompt_text, target_text, target_sr)
            elapsed = time.perf_counter() - t0
            duration = len(audio) / target_sr
            rtf = elapsed / duration if duration > 0 else 0

            wav_file = save_wav_numpy(run_dir, f"clone_{tag}", audio, target_sr)
            print(f"[{tag}] Done: {duration:.2f}s, RTF={rtf:.3f}")
            clone_rows.append({
                "id": f"clone/{tag}", "text": target_text,
                "audio_file": wav_file, "ref_audio": ref_filename,
                "total_time": round(elapsed, 3),
                "audio_duration": round(duration, 3),
                "rtf": round(rtf, 3),
            })
        except Exception as e:
            print(f"[{tag}] Failed: {e}")
            traceback.print_exc()
            clone_rows.append({"id": f"clone/{tag}", "text": target_text,
                               "ref_audio": ref_filename, "error": str(e)})

    sections.append({
        "type": "voice_clone",
        "title": "语音克隆测试",
        "description": "使用 T08_voice_clone.csv 中定义的参考音频和文本",
        "rows": clone_rows,
    })

    # 保存段落数据和元信息（供 demo_voice_design 合并使用）
    sections_path = os.path.join(run_dir, "_sections.json")
    with open(sections_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    meta_path = os.path.join(run_dir, "_meta.json")
    meta = {"has_tts": True, "has_voice_clone": True, "has_voice_design": False}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta.update(json.load(f))
    meta["has_tts"] = True
    meta["has_voice_clone"] = True
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ── 生成报告 ──
    total = sum(len(s.get("rows", [])) for s in sections)
    if total > 0:
        report_path = generate_report("MOSS-TTSD", run_dir, sections, start_time)
        print(f"\n[REPORT] {report_path}")

    print(f"[DONE] {total} tests. Output: {run_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOSS-TTSD 独立评测")
    parser.add_argument("--quant", default="no", choices=["no", "4bit", "8bit"],
                        help="量化模式: no(bf16 ~7.8GB), 4bit(~2.6GB), 8bit(~4.2GB)")
    args = parser.parse_args()
    QUANT_MODE = args.quant

    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
