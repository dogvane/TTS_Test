#!/usr/bin/env python3
"""
MOSS-VoiceGenerator 语音设计 demo 程序
基于 T07_voice_design.csv 中的音色描述 + 文本，生成对应语音。

如果 results/MOSS-TTSD/ 下已有评测结果目录（来自 demo_clone_v2），
则将 voice design 结果追加到该目录并更新 report.html；否则新建目录。

用法（WSL2 conda 环境）:
    conda activate moss-tts
    cd /mnt/o/ai/TTS/TTS_Test
    python -m tts.MOSS-TTSD.demo_voice_design
"""

import csv
import html
import json
import os
import re
import time
import traceback
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

# 配置参数（WSL2 路径格式）
MODEL_PATH = "/mnt/g/ai/TTS/MOSS-TTSD/models/MOSS-VoiceGenerator"
CODEC_PATH = "/mnt/g/ai/TTS/MOSS-TTSD/models/MOSS-Audio-Tokenizer"
PROJECT_ROOT = "/mnt/o/ai/TTS/TTS_Test"

T07_CSV_PATH = os.path.join(PROJECT_ROOT, "test_texts/T07_voice_design.csv")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/MOSS-TTSD")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def find_or_create_run_dir(results_root: str) -> tuple[str, bool]:
    """查找最新的已有结果目录（没有 voice_design 段落的），或创建新目录。

    Returns: (run_dir, is_existing)
    """
    if not os.path.isdir(results_root):
        os.makedirs(results_root, exist_ok=True)

    # 查找已有目录
    dirs = sorted([
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
        and re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', d)
    ], reverse=True)

    for dirname in dirs:
        run_dir = os.path.join(results_root, dirname)
        meta_path = os.path.join(run_dir, "_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if not meta.get("has_voice_design"):
                return run_dir, True
        else:
            # 没有 meta 文件，检查是否有 report.html 但没有 design_*.wav
            design_files = [f for f in os.listdir(run_dir) if f.startswith("design_") and f.endswith(".wav")]
            if not design_files:
                return run_dir, True

    # 没有可合并的目录，创建新的
    run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(results_root, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, False


def load_section_data(run_dir: str) -> list[dict]:
    """从 _sections.json 加载已有的测试段落数据。"""
    path = os.path.join(run_dir, "_sections.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_section_data(run_dir: str, sections: list[dict]):
    """保存测试段落数据到 _sections.json。"""
    path = os.path.join(run_dir, "_sections.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)


def update_meta(run_dir: str, has_voice_design: bool = True):
    """更新目录元信息。"""
    meta_path = os.path.join(run_dir, "_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta["has_voice_design"] = has_voice_design
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_model_and_processor():
    """加载 MOSS-VoiceGenerator 模型和处理器。"""
    print(f"[INFO] 加载模型: {MODEL_PATH}")
    print(f"[INFO] 设备: {DEVICE}, 精度: {DTYPE}")

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        codec_path=CODEC_PATH, normalize_inputs=True,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(DEVICE)

    def _load_model(attn_impl: str):
        return AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True,
            attn_implementation=attn_impl, torch_dtype=DTYPE,
        ).to(DEVICE)

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


# ---------------------------------------------------------------------------
# CSV 读取
# ---------------------------------------------------------------------------
def read_voice_design_csv(csv_path: str) -> list[dict]:
    """读取 T07_voice_design.csv，返回 [{instruction, text}, ...]。"""
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                samples.append({
                    "instruction": row[0].strip(),
                    "text": row[1].strip(),
                })
    return samples


# ---------------------------------------------------------------------------
# HTML 报告生成
# ---------------------------------------------------------------------------
def generate_report(engine: str, run_dir: str, sections: list[dict],
                    start_time: datetime):
    """根据所有 sections 生成完整的 report.html。"""
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
    print("[INFO] MOSS-VoiceGenerator 语音设计评测")
    start_time = datetime.now()

    # 查找或创建结果目录
    run_dir, is_existing = find_or_create_run_dir(RESULTS_ROOT)
    print(f"[INFO] 输出目录: {run_dir} ({'已有' if is_existing else '新建'})")

    # 加载已有段落数据（如果有）
    sections = load_section_data(run_dir)
    # 检查是否已有 voice_design 段落，有则移除（重新生成）
    sections = [s for s in sections if s.get("type") != "voice_design"]

    # 加载模型
    model, processor = load_model_and_processor()
    sample_rate = int(processor.model_config.sampling_rate)

    # 读取 CSV
    samples = read_voice_design_csv(T07_CSV_PATH)
    print(f"[INFO] 共 {len(samples)} 条语音设计任务")

    design_rows = []

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        text = sample["text"]
        tag = f"v{i + 1}"
        out_file = f"design_{tag}.wav"
        out_path = os.path.join(run_dir, out_file)

        print(f"\n[{tag}] voice='{instruction}' text='{text[:30]}...'")

        try:
            conversations = [
                [processor.build_user_message(text=text, instruction=instruction)],
            ]

            batch = processor(conversations, mode="generation")
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    audio_temperature=1.5, audio_top_p=0.6,
                    audio_top_k=50, audio_repetition_penalty=1.1,
                )
            elapsed = time.perf_counter() - t0

            # 解码音频
            audio_segments = []
            for message in processor.decode(outputs):
                for audio in message.audio_codes_list:
                    seg = audio.detach().cpu().to(torch.float32).numpy()
                    if seg.shape[-1] > 100:
                        audio_segments.append(seg)

            if not audio_segments:
                raise RuntimeError("无有效音频输出")

            audio_np = np.concatenate(audio_segments, axis=-1)
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze(0)

            duration = len(audio_np) / sample_rate
            rtf = elapsed / duration if duration > 0 else 0

            sf.write(out_path, audio_np, sample_rate)
            print(f"[{tag}] Done: {duration:.2f}s, RTF={rtf:.3f}")

            design_rows.append({
                "id": f"design/{tag}", "text": text, "audio_file": out_file,
                "voice": instruction,
                "total_time": round(elapsed, 3),
                "audio_duration": round(duration, 3),
                "rtf": round(rtf, 3),
            })

        except Exception as e:
            print(f"[{tag}] Failed: {e}")
            traceback.print_exc()
            design_rows.append({
                "id": f"design/{tag}", "text": text,
                "voice": instruction, "error": str(e),
            })

    # 添加 voice design 段落
    design_section = {
        "type": "voice_design",
        "title": "音色设计测试",
        "description": "使用 T07_voice_design.csv 中定义的音色描述和文本",
        "rows": design_rows,
    }
    # 按顺序插入：TTS → voice_design → clone
    tts_sections = [s for s in sections if s.get("type") == "tts"]
    clone_sections = [s for s in sections if s.get("type") == "voice_clone"]
    sections = tts_sections + [design_section] + clone_sections

    # 保存数据
    save_section_data(run_dir, sections)
    update_meta(run_dir, has_voice_design=True)

    # 生成/更新报告
    total = sum(len(s.get("rows", [])) for s in sections)
    if total > 0:
        report_path = generate_report("MOSS-TTSD", run_dir, sections, start_time)
        print(f"\n[REPORT] {report_path}")

    print(f"[DONE] {len(design_rows)} design tests. Output: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
