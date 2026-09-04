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

本引擎跑在 IndexTTS 自带的 Windows venv（`.venv/Scripts/python.exe`）中，与项目其余引擎（WSL2 + conda、Docker）环境互不相通。项目没有统一网关，本引擎需手动启动 webapi 后端。

> 注意：引擎本身在 Windows 侧，依赖 `G:` 盘的 IndexTTS 源码与模型；评测客户端（client/runner.py）可在任意一侧运行。

## 启动后端（Windows）

在 Windows 终端，使用 IndexTTS 的 venv：

```bat
cd O:\ai\TTS\TTS_Test

:: 激活 IndexTTS 的 venv
G:\ai\TTS\index-tts\IndexTTS-2.5\index-tts\.venv\Scripts\activate

:: 设置 PYTHONPATH（项目根 + IndexTTS 源码根）
set PYTHONPATH=O:\ai\TTS\TTS_Test;G:\ai\TTS\index-tts\IndexTTS-2.5\index-tts

:: 启动后端（统一端口 8002）
python -m tts.IndexTTS25.webapi --port 8002
```

启动后访问 `http://localhost:8002/health` 应返回 `{"model":"IndexTTS2",...}`。WSL2 与 Windows 的 localhost 互通。

## 运行评测

> 命名映射：目录名 `IndexTTS25`（Python 包名不可含点）；引擎名（client/config.yaml 键、`--engine` 参数）为 `index-tts`；请求体 model id 为 `IndexTTS2`（health.model）。

后端就绪后，在 WSL2 正常跑 runner：

```bash
python -m client.runner --engine index-tts
python -m client.runner --engine index-tts --test T02
```

## 配置（config.yaml）

| 字段 | 说明 |
|------|------|
| `model_dir` | 权重目录（含 `config.yaml`、`gpt.pth` 等） |
| `cfg_path` | 模型配置文件 |
| `source_path` | IndexTTS 源码根（含 `indextts` 包） |
| `backend_url` / `webapi_port` | 后端地址与端口（统一 8002） |
| `device` / `use_fp16` | 推理设备与精度 |

## 情绪字段链路

`emotion` 字段不在标准 OpenAI 接口里，为本项目的扩展字段：

```
runner (emotion=...) → webapi /v1/audio/speech (emotion) → 内置 Qwen 0.6B 情绪模型转向量注入
```

支持的情绪标签（对应 `test_texts/T06_emotion_clone.csv`）：兴奋、严肃、轻松、中性、悲伤、愤怒。
