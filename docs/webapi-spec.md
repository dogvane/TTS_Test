# TTS 后端 WebAPI 接口规范

> 本文档定义了 TTS 统一评测项目中，各引擎后端（webapi.py）必须实现的 HTTP 接口。
> 接口设计以 [OpenAI Audio Speech API](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create/) 为基础，在此基础上扩展了评测所需的字段。

## 1. 总览

```
┌─────────────┐    HTTP     ┌─────────────────┐
│ client      │ ──────────→ │ tts/{engine}/   │
│ runner.py   │  :8002      │ webapi.py       │
└─────────────┘             └─────────────────┘
```

- **客户端** (`client/runner.py`) 按 `client/config.yaml` 中的 base_url 直连引擎，统一端口 8002
- **后端** (`tts/{engine}/webapi.py`) 负责实际的模型推理，在各自 conda/venv/Docker 环境中手动启动
- **同一时刻只运行一个引擎**（统一端口），切换引擎前先停掉前一个
- 所有接口使用 JSON 请求体、二进制音频响应

### 1.1 引擎启动方式

各引擎在各自环境中手动启动 webapi.py（详见各引擎目录下的 readme.md）：

- **WSL2 + conda**（VoxCPM2 / MOSS-TTSD / qwen3-tts）：`conda activate <env>` 后 `python -m tts.<Engine>.webapi`
- **Windows venv**（IndexTTS25）：按 `tts/IndexTTS25/readme.md` 以 venv 启动
- **Docker**（higgs-audio）：容器端口映射到宿主 8002，本目录 webapi 仅做健康检查代理

统一监听 8002 端口，**同一时刻只运行一个引擎**，切换引擎前先停掉前一个。

## 2. API 端点

### 2.1 POST `/v1/audio/speech`

同步合成语音，返回音频文件二进制内容。

**请求体：**

```json
{
  "model": "VoxCPM2",
  "input": "要合成的文本",
  "voice": "default",
  "response_format": "wav",
  "speed": 1.0,
  "reference_audio": null,
  "prompt_text": null,
  "reference_wav_path": null,
  "temperature": 0.7,
  "top_p": 0.7,
  "repetition_penalty": 1.1
}
```

**响应：** 音频文件二进制，`Content-Type` 对应格式（如 `audio/wav`）。

### 2.2 POST `/v1/audio/speech/stream`

流式合成语音，返回 PCM 音频流。

**请求体：** 同 2.1

**响应：**

- `Content-Type: audio/pcm`
- `X-Sample-Rate: 44100`
- `X-Sample-Format: float32`
- Body 为连续的 float32 PCM 数据块

### 2.3 GET `/health`

健康检查。

**响应：**

```json
{
  "status": "ok",
  "model": "EngineName",
  "sample_rate": 44100
}
```

> 能力声明不再通过 HTTP 接口查询，由 `client/config.yaml` 中每个引擎的 `capabilities` 列表静态声明。

## 3. 请求字段定义

### 3.1 标准 OpenAI 字段

这些字段与 OpenAI 官方 API 对齐，所有引擎**必须**支持。

| 字段              | 类型   |  必填  | 默认值      | 说明                                     |
| ----------------- | ------ | :----: | ----------- | ---------------------------------------- |
| `model`           | string | **是** | -           | 引擎名称，如 `"VoxCPM2"`                 |
| `input`           | string | **是** | -           | 待合成文本                               |
| `voice`           | string |   否   | `"default"` | 音色名称、音色描述或情绪标签             |
| `response_format` | string |   否   | `"wav"`     | 输出格式: `wav` / `mp3` / `opus` / `pcm` |
| `speed`           | float  |   否   | `1.0`       | 语速倍率，范围 `0.25` ~ `4.0`            |

**`voice` 字段取值规则：**

| 值                                       | 含义                                      |
| ---------------------------------------- | ----------------------------------------- |
| `"default"`                              | 使用引擎默认音色                          |
| `"male_news"`, `"female_soft"` 等        | 使用引擎预设音色                          |
| `"(温柔御姐)"`, `"(沉稳男声)"` 等        | 音色描述，用于 voice_design 能力          |
| `"[happy]"`, `"[sad]"`, `"[excited]"` 等 | 情绪标签，用于 emotion 能力（方括号区分） |

### 3.2 扩展测试字段

这些字段**不在** OpenAI 标准中，是本评测项目的扩展，用于 voice clone 和推理参数控制。

| 字段                 | 类型   | 必填 | 默认值 | 说明                                           |
| -------------------- | ------ | :--: | ------ | ---------------------------------------------- |
| `reference_audio`    | string |  否  | `null` | Base64 编码的参考音频（voice clone）           |
| `prompt_text`        | string |  否  | `null` | 参考音频对应的文本转录                         |
| `reference_wav_path` | string |  否  | `null` | 参考音频的本地文件路径（仅供 webapi 本地使用） |
| `temperature`        | float  |  否  | `0.7`  | 采样温度，越高越随机                           |
| `top_p`              | float  |  否  | `0.7`  | Nucleus sampling 阈值                          |
| `repetition_penalty` | float  |  否  | `1.1`  | 重复惩罚系数                                   |

**voice clone 工作流程：**

```
1. 客户端发送 reference_audio (Base64) + prompt_text + input
2. webapi 使用参考音频 + 文本进行语音克隆合成
```

`reference_wav_path` 仅在后端本地有文件时使用，外部调用应使用 `reference_audio` (Base64)。

## 4. 能力声明

每个引擎在 `client/config.yaml` 中通过 `capabilities` 列表声明支持的功能：

| 能力           | 说明                     | 必需 |
| -------------- | ------------------------ | :--: |
| `tts`          | 基础文本转语音           |  是  |
| `voice_design` | 通过自然语言描述生成音色 |  否  |
| `voice_clone`  | 基于参考音频的语音克隆   |  否  |
| `emotion`      | 支持情绪/语气控制        |  否  |
| `streaming`    | 支持流式输出             |  否  |

## 5. 新引擎接入指南

按以下步骤接入新的 TTS 引擎：

### 5.1 创建目录结构

```
tts/YourEngine/
├── __init__.py          # 可为空
├── config.yaml          # 配置文件
├── webapi.py            # 后端 API 服务
├── readme.md            # 启动方式说明
└── voices/              # 参考音频目录
    └── readme.md
```

### 5.2 实现 webapi.py

必须实现以下端点：

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="YourEngine TTS API")

class TTSRequest(BaseModel):
    # ── 标准 OpenAI 字段 ──
    model: str = Field(default="YourEngine")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default")
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    # ── 扩展测试字段 ──
    reference_audio: Optional[str] = Field(default=None)
    prompt_text: Optional[str] = Field(default=None)
    reference_wav_path: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    top_p: float = Field(default=0.7, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0)

@app.post("/v1/audio/speech")
def create_speech(request: TTSRequest):
    # 推理并返回音频
    ...

@app.post("/v1/audio/speech/stream")
def create_speech_stream(request: TTSRequest):
    # 流式推理
    ...

@app.get("/health")
def health():
    return {"status": "ok", "model": "YourEngine", "sample_rate": 44100} # sample_rate 基于tts模型自身的来返回
```

### 5.3 编写 config.yaml

```yaml
backend_url: "http://localhost:8002" # webapi 地址
webapi_port: 8002 # 端口（统一 8002）
timeout: 120 # 超时(秒)
sample_rate: 44100 # 采样率  基于tts模型自身的来返回
model_path: "/path/to/model/weights" # 模型权重
source_path: "/path/to/source/code" # 源码路径
```

### 5.4 注册引擎

在 `client/config.yaml` 中添加引擎项，声明引擎名、model id 与能力：

```yaml
engines:
  your-engine:
    model: YourEngine
    base_url: http://localhost:8002
    engine_dir: tts/YourEngine
    capabilities: [tts, voice_clone]
```

- `model`：请求体中的 model id，需与 webapi 的 `/health` 返回的 `model` 一致
- `capabilities`：决定 runner 自动执行哪些测试段落

之后按引擎依赖准备环境（conda / venv / Docker），确保 webapi.py 所需依赖齐全。

## 6. 错误响应

所有错误使用标准 HTTP 状态码 + JSON body：

```json
{
  "detail": "error message"
}
```

| 状态码 | 含义         |
| ------ | ------------ |
| 400    | 请求参数错误 |
| 404    | 模型不存在   |
| 500    | 推理失败     |
| 503    | 后端不可用   |
