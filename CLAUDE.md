# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TTS 统一标准评测项目，横向对比不同 TTS 引擎的语音质量和执行效率。项目运行在 WSL2（Ubuntu-22.04）环境下，Windows 磁盘通过 `/mnt/` 前缀访问（如 `G:\` → `/mnt/g/`，`O:\` → `/mnt/o/`）。

## Commands

```bash
# 启动网关（端口 9000，按需拉起各引擎后端）
python -m server.main

# 运行评测客户端
python -m client.runner                          # 默认 voxcpm
python -m client.runner --engine MOSS-TTSD       # 指定引擎
python -m client.runner --engine MOSS-TTSD --test T02  # 指定测试用例

# 单独启动引擎后端（调试用）
conda activate moss-tts && python -m tts.MOSS-TTSD.webapi --port 8004
```

## Architecture

### Two-Layer Design

```
client/runner.py  →  server/main.py (Gateway :9000)  →  tts/{Engine}/webapi.py (Backend)
                                              ↕
                                      server/base_adapter.py (Abstract interface)
                                      tts/{Engine}/adapter.py (Concrete adapter)
```

**Gateway (`server/main.py`)**: FastAPI 网关，兼容 OpenAI `/v1/audio/speech` 接口。收到请求后，通过 adapter 转发到对应引擎的 webapi 后端。网关本身不加载模型，通过 `subprocess.Popen` 懒启动后端进程（自动查找 conda 环境中的 python）。

**Backend (`tts/{Engine}/webapi.py`)**: 各引擎独立的 FastAPI 服务，在各自的 conda 环境中运行，直接加载模型推理。网关通过 HTTP 调用后端。

**Adapter (`tts/{Engine}/adapter.py`)**: 继承 `server/base_adapter.py` 的 `BaseAdapter`，封装对 webapi 后端的 HTTP 调用。网关通过 adapter 的 `synthesize()` / `synthesize_streaming()` 方法与后端通信。

### Adding a New Engine

1. 创建 `tts/{EngineName}/` 目录
2. 实现 `adapter.py`（继承 `BaseAdapter`，声明 `capabilities`）
3. 实现 `webapi.py`（FastAPI 服务，加载模型推理）
4. 编写 `config.yaml`（必须包含 `backend_url`、`webapi_port`、模型路径等）
5. 在 `server/config.yaml` 的 `engines` 列表中注册

### Engine Capabilities

每个 adapter 声明 `capabilities` 列表，决定 runner 执行哪些测试：
- `tts` → 基础文本转语音（runner 读取 `test_texts/T*.txt`）
- `voice_design` → 音色设计（runner 读取 `test_texts/T07_voice_design.csv`，CSV 格式：音色描述,文本）
- `voice_clone` → 语音克隆（runner 读取 `test_texts/T08_voice_clone.csv`，CSV 格式：参考音频路径,文本）

### MOSS-TTSD Engine Specifics

MOSS-TTSD 引擎同时使用两个模型，通过请求中的字段自动路由：
- 有 `instruction` → voice_design 模式 → **MOSS-VoiceGenerator**（官方参数：temperature=1.5, top_p=0.6）
- 有 `reference_audio` + `prompt_text` → voice_clone 模式 → **MOSS-TTSD-v1.0**（continuation 模式）
- 其他 → tts 模式 → **MOSS-TTSD-v1.0**（generation 模式）

两个模型按需懒加载，不同时占用显存。模型文件位于 `G:\ai\TTS\MOSS-TTSD\models\`。

### Conda Environments

各引擎使用独立的 conda 环境（见 `server/config.yaml`）：
- `MOSS-TTSD` → `moss-tts`
- `VoxCPM2` → `voxcpm`
