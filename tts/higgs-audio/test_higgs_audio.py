"""Higgs Audio v3 TTS 本地测试脚本。

直接调用 Docker 中运行的 SGLang-Omni 服务 (http://localhost:8001)，
使用项目 test_texts/ 目录下的统一测试用例。

语音克隆时，参考音频会被复制到 Docker 挂载的 model 目录下，
使用容器内路径作为 audio_path 传递给 API。

使用方法:
  1. 先启动 Docker 容器中的 SGLang-Omni 服务（参考 tts/higgs-audio/../docker.txt）
  2. python test_higgs_audio.py
  3. python test_higgs_audio.py --test T02           # 只跑指定测试
  4. python test_higgs_audio.py --test T08            # 只跑语音克隆
"""

import base64
import csv
import io
import json
import os
import shutil
import struct
import sys
import time

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASE_URL = "http://localhost:8001/v1/audio/speech"
TEST_TEXTS_DIR = os.path.join(PROJECT_ROOT, "test_texts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "higgs-audio")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def check_server() -> bool:
    """检查 Docker 中的服务是否就绪。"""
    for url in ["http://localhost:8001/health", "http://localhost:8001/v1/models"]:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print("[OK] SGLang-Omni 服务已就绪")
                return True
        except requests.RequestException:
            pass
    print("[FAIL] 服务未启动，请先运行 Docker 容器并启动 sgl-omni serve")
    return False


def get_wav_duration(content: bytes) -> float:
    """从 WAV 文件头解析音频时长（秒）。"""
    if len(content) < 44:
        return 0.0
    num_channels = struct.unpack_from('<H', content, 22)[0]
    sample_rate = struct.unpack_from('<I', content, 24)[0]
    bits_per_sample = struct.unpack_from('<H', content, 34)[0]
    if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
        return 0.0
    idx = 12
    data_size = 0
    while idx < len(content) - 8:
        chunk_id = content[idx:idx+4]
        chunk_size = struct.unpack_from('<I', content, idx+4)[0]
        if chunk_id == b'data':
            data_size = chunk_size
            break
        idx += 8 + chunk_size
    if data_size == 0:
        data_size = len(content) - idx - 8
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    return data_size / byte_rate if byte_rate > 0 else 0.0


def tts_request(payload: dict, filename: str, stream: bool = False):
    """发送 TTS 请求，记录 RTF，保存音频文件。"""
    gen_start = time.perf_counter()
    filepath = os.path.join(RESULTS_DIR, filename)

    if stream:
        chunk_count = 0
        with requests.post(BASE_URL, json=payload, stream=True) as resp, \
             open(filepath, "wb") as f:
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("finish_reason") == "stop":
                    break
                audio = event.get("audio") or {}
                if audio.get("data"):
                    f.write(base64.b64decode(audio["data"]))
                    chunk_count += 1
        content = open(filepath, "rb").read()
    else:
        resp = requests.post(BASE_URL, json=payload, timeout=300)
        gen_time = time.perf_counter() - gen_start
        if resp.status_code != 200:
            print(f"    [SKIP] HTTP {resp.status_code}: {resp.text[:200]}")
            return
        with open(filepath, "wb") as f:
            f.write(resp.content)
        content = resp.content

    gen_time = time.perf_counter() - gen_start
    audio_dur = get_wav_duration(content)
    rtf = gen_time / audio_dur if audio_dur > 0 else 0

    print(f"  -> {filepath} ({len(content)} bytes)")
    print(f"     生成耗时: {gen_time:.2f}s | 音频时长: {audio_dur:.2f}s | "
          f"RTF: {rtf:.3f} ({'快于' if rtf < 1 else '慢于'}实时, "
          f"{1/rtf:.1f}x)" if rtf > 0 else "")


def read_test_text(test_id: str) -> str | None:
    """读取 test_texts/ 下的指定测试文本。"""
    for f in os.listdir(TEST_TEXTS_DIR):
        if f.startswith(test_id) and f.endswith(".txt"):
            with open(os.path.join(TEST_TEXTS_DIR, f), "r", encoding="utf-8") as fh:
                return fh.read().strip()
    return None


def list_test_texts() -> list[str]:
    """列出 test_texts/ 下所有 txt 文件的测试 ID。"""
    ids = set()
    for f in os.listdir(TEST_TEXTS_DIR):
        if f.endswith(".txt"):
            ids.add(f.split("_")[0])
    return sorted(ids)


def read_voice_clone_csv() -> list[dict]:
    """读取 T08_voice_clone.csv，返回 [{ref_path, text}, ...]。"""
    rows = []
    for f in os.listdir(TEST_TEXTS_DIR):
        if f.startswith("T08") and f.endswith(".csv"):
            with open(os.path.join(TEST_TEXTS_DIR, f), "r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                for line in reader:
                    if len(line) >= 2 and line[0].strip() and line[1].strip():
                        ref_path = line[0].strip()
                        if not os.path.isabs(ref_path):
                            ref_path = os.path.join(PROJECT_ROOT, ref_path)
                        rows.append({"ref_path": ref_path, "text": line[1].strip()})
    return rows


# ---------------------------------------------------------------------------
# Docker 模型挂载路径配置
# ---------------------------------------------------------------------------
# Docker 启动时将宿主机 models 目录挂载到容器内 /model
# 宿主机路径: G:\ai\TTS\higgs-audio-v3-tts-4b\models -> 容器内: /model
DOCKER_MODEL_HOST_DIR = r"G:\ai\TTS\higgs-audio-v3-tts-4b\models"
DOCKER_MODEL_CONTAINER_DIR = "/model"
REF_AUDIO_SUBDIR = "ref_audio"  # 参考音频在 model 目录下的子目录名


def copy_ref_to_docker(ref_path: str) -> str | None:
    """将参考音频复制到 Docker 挂载的 model/ref_audio/ 子目录，返回容器内路径。"""
    ref_filename = os.path.basename(ref_path)
    host_ref_dir = os.path.join(DOCKER_MODEL_HOST_DIR, REF_AUDIO_SUBDIR)
    os.makedirs(host_ref_dir, exist_ok=True)
    dest_path = os.path.join(host_ref_dir, ref_filename)

    if not os.path.isfile(dest_path) or os.path.getmtime(ref_path) > os.path.getmtime(dest_path):
        shutil.copy2(ref_path, dest_path)
        print(f"     已复制参考音频到: {dest_path}")
    else:
        print(f"     参考音频已存在: {dest_path}")

    # 返回容器内路径
    return f"{DOCKER_MODEL_CONTAINER_DIR}/{REF_AUDIO_SUBDIR}/{ref_filename}"


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def test_basic_tts(test_ids: list[str]):
    """使用 test_texts/ 下的文本做基础 TTS 测试。"""
    print("\n" + "=" * 60)
    print("[基础 TTS 测试]")
    print("=" * 60)
    for test_id in test_ids:
        text = read_test_text(test_id)
        if text is None:
            print(f"  [SKIP] 测试文本 {test_id} 不存在")
            continue
        print(f"\n[{test_id}] ({len(text)} chars)")
        # Higgs Audio 支持长文本直接合成
        tts_request({"input": text}, f"{test_id}_basic.wav")


def test_voice_clone():
    """使用 T08_voice_clone.csv 做语音克隆测试。"""
    cases = read_voice_clone_csv()
    if not cases:
        print("\n[语音克隆测试] T08_voice_clone.csv 无测试用例，跳过")
        return

    # 检查 Docker 模型目录是否存在
    if not os.path.isdir(DOCKER_MODEL_HOST_DIR):
        print(f"\n[语音克隆测试] Docker 模型目录不存在: {DOCKER_MODEL_HOST_DIR}")
        print("  请确认 Docker 容器已启动并挂载了模型目录")
        return

    print("\n" + "=" * 60)
    print(f"[语音克隆测试] ({len(cases)} cases)")
    print("=" * 60)
    for i, case in enumerate(cases):
        ref_path = case["ref_path"]
        text = case["text"]
        tag = f"clone_{i + 1}"
        ref_filename = os.path.basename(ref_path)

        if not os.path.isfile(ref_path):
            print(f"\n  [{tag}] [SKIP] 参考音频不存在: {ref_path}")
            continue

        print(f"\n  [{tag}] ref='{ref_filename}' text='{text[:30]}...'")

        # 将参考音频复制到 Docker 挂载的 model 目录下，获取容器内路径
        container_path = copy_ref_to_docker(ref_path)
        if container_path is None:
            print("    [SKIP] 无法复制参考音频")
            continue

        # Higgs Audio 使用 references 字段做语音克隆
        # audio_path 支持容器内本地路径
        tts_request({
            "input": text,
            "references": [{
                "audio_path": container_path,
                "text": os.path.splitext(ref_filename)[0],
            }],
            "temperature": 0.8,
            "top_k": 50,
        }, f"T08_{tag}.wav")


def test_streaming():
    """流式合成测试。"""
    print("\n" + "=" * 60)
    print("[流式合成测试]")
    print("=" * 60)
    text = ("流式语音合成技术能够在模型生成音频的同时，将已合成的部分实时推送给客户端，"
            "从而大幅降低首包延迟，提升用户的交互体验。"
            "这种技术在智能助手、实时翻译和语音对话等场景中有着广泛的应用价值。"
            "随着大语言模型推理速度的不断提升，流式合成将成为语音交互系统的标配能力。")
    tts_request({"input": text, "stream": True}, "streaming_test.wav", stream=True)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Higgs Audio v3 TTS 本地测试")
    parser.add_argument("--test", nargs="*", default=None,
                        help="指定测试用例，如 T02 T08 stream")
    parser.add_argument("--url", default=None, help="服务地址 (默认 http://localhost:8001)")
    args = parser.parse_args()

    if args.url:
        global BASE_URL
        BASE_URL = args.url.rstrip("/") + "/v1/audio/speech"

    print("=" * 60)
    print("Higgs Audio v3 TTS 测试")
    print(f"服务地址: {BASE_URL}")
    print(f"测试文本: {TEST_TEXTS_DIR}")
    print(f"输出目录: {RESULTS_DIR}")
    print("=" * 60)

    if not check_server():
        sys.exit(1)

    total_start = time.perf_counter()
    passed = 0
    failed = 0

    # 确定要运行的测试
    test_ids = args.test
    run_clone = False
    run_stream = False
    tts_ids = []

    if test_ids is None:
        # 默认运行所有测试
        tts_ids = list_test_texts()
        run_clone = True
        run_stream = True
    else:
        for t in test_ids:
            if t == "T08":
                run_clone = True
            elif t.lower() == "stream":
                run_stream = True
            else:
                tts_ids.append(t)
        if not tts_ids and not run_clone and not run_stream:
            tts_ids = list_test_texts()
            run_clone = True
            run_stream = True

    # 运行基础 TTS 测试
    if tts_ids:
        try:
            test_basic_tts(tts_ids)
            passed += 1
        except Exception as e:
            print(f"  [ERROR] 基础 TTS 测试失败: {e}")
            failed += 1

    # 运行语音克隆测试
    if run_clone:
        try:
            test_voice_clone()
            passed += 1
        except Exception as e:
            print(f"  [ERROR] 语音克隆测试失败: {e}")
            failed += 1

    # 运行流式测试
    if run_stream:
        try:
            test_streaming()
            passed += 1
        except Exception as e:
            print(f"  [ERROR] 流式测试失败: {e}")
            failed += 1

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败, 总耗时: {total_time:.2f}s")
    print(f"音频文件保存在: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
