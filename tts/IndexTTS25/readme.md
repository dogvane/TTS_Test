# IndexTTS 2.5 引擎

[IndexTTS2](https://github.com/index-tts/index-tts) — Bilibili IndexTeam 出品，特色是**情绪表达（Emotionally Expressive）+ 时长控制（Duration-Controlled）+ 零样本克隆（Zero-Shot）**。

## 能力

| 能力 | 说明 |
|------|------|
| `tts` | 基础合成（无参考音频时用内置默认样本） |
| `voice_clone` | 零样本克隆，`reference_audio` 提供参考音色 |
| `emotion` | 情绪控制，`emotion` 字段经内置 Qwen 0.6B 情绪模型转向量注入 |
| `streaming` | 流式合成（`stream_return=True`） |

## 关键差异：Windows 原生 venv

本引擎与项目其余引擎（WSL2 + conda）分属**两个操作系统**：

- **后端 webapi**：跑在 IndexTTS 自带的 Windows venv（`.venv/Scripts/python.exe`）
- **网关 + adapter**：跑在 WSL2，仅通过 HTTP 调用后端

因此采用「**外部独立后端 + HTTP**」模式（同 `higgs-audio`）：网关不负责拉起后端，需手动启动。

## 启动后端（Windows）

在 Windows 终端，使用 IndexTTS 的 venv：

```bat
cd O:\ai\TTS\TTS_Test

:: 激活 IndexTTS 的 venv
G:\ai\TTS\index-tts\IndexTTS-2.5\index-tts\.venv\Scripts\activate

:: 设置 PYTHONPATH（项目根 + IndexTTS 源码根）
set PYTHONPATH=O:\ai\TTS\TTS_Test;G:\ai\TTS\index-tts\IndexTTS-2.5\index-tts

:: 启动后端（端口 8006）
python -m tts.IndexTTS25.webapi --port 8006
```

启动后访问 `http://localhost:8006/health` 应返回 `{"model":"IndexTTS2",...}`。WSL2 与 Windows 的 localhost 互通，网关可直接访问。

## 运行评测

> 命名映射：目录名 `IndexTTS25`（Python 包名不可含点），model id 为 `IndexTTS2`（adapter.name 与 health.model）。runner 用 `--engine IndexTTS2`。

后端就绪后，在 WSL2 正常跑 runner：

```bash
python -m client.runner --engine IndexTTS2
python -m client.runner --engine IndexTTS2 --test T02
```

## 配置（config.yaml）

| 字段 | 说明 |
|------|------|
| `model_dir` | 权重目录（含 `config.yaml`、`gpt.pth` 等） |
| `cfg_path` | 模型配置文件 |
| `source_path` | IndexTTS 源码根（含 `indextts` 包） |
| `backend_url` / `webapi_port` | 外部后端地址 |
| `external_mode` | 标记为外部后端，网关不拉起 |
| `device` / `use_fp16` | 推理设备与精度 |

## 情绪字段链路

`emotion` 字段不在标准 OpenAI 接口里，本项目通过三层 `**kwargs` 透传：

```
runner (emotion=...) → 网关 TTSRequest → adapter.synthesize(**kwargs) → webapi /v1/audio/speech (emotion)
```

支持的情绪标签（对应 `test_texts/T06_emotion_clone.csv`）：兴奋、严肃、轻松、中性、悲伤、愤怒。
