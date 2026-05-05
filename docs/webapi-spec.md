# TTS 后端 WebAPI 接口规范

> 本文档定义了 TTS 统一评测项目中，各引擎后端（webapi.py）必须实现的 HTTP 接口。
> 接口设计以 [OpenAI Audio Speech API](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create/) 为基础，在此基础上扩展了评测所需的字段。

## 1. 总览

```
┌─────────────┐    HTTP     ┌──────────────────┐   conda子进程   ┌─────────────────┐
│ client      │ ──────────→ │ server/main.py   │ ──────────────→ │ tts/{engine}/   │
│ runner.py   │  :9000      │ (网关)            │   :800x        │ webapi.py       │
└─────────────┘             └──────────────────┘                └─────────────────┘
                            转发请求到对应后端
                            按需启动/切换 conda 环境
```

- **网关** (`server/main.py`) 接收请求后，按 `model` 字段转发到对应的后端 webapi
- **后端** (`tts/{engine}/webapi.py`) 负责实际的模型推理
- **Conda 隔离**：每个引擎在各自的 conda 环境子进程中运行，同一时刻只有一个引擎的 conda 进程存活
- 所有接口使用 JSON 请求体、二进制音频响应

### 1.1 Conda 环境管理机制

网关为每个引擎维护独立的 conda 环境子进程：

```
请求 model=VoxCPM2
  → ensure_backend("VoxCPM2")
    → 活跃引擎 == VoxCPM2 且存活？ → 直接复用
    → 活跃引擎 == 其他引擎？   → 先停旧引擎，再启新引擎
    → 无活跃引擎？             → 启动新引擎
  → 正常调用 adapter.synthesize()
```

关键行为：

- **首次请求**：网关通过 `conda info --json` 查找对应 conda 环境的 python 路径，启动 webapi.py 子进程
- **连续同引擎请求**：直接复用已启动的 conda 进程，不重启
- **切换引擎请求**：先停止旧引擎的 conda 进程（terminate → wait → kill），再启动新引擎
- **进程健康检查**：通过 `/health` 端点轮询确认后端就绪，超时 600 秒
- **网关关闭**：自动清理所有 conda 子进程

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

### 2.4 GET `/v1/models`

查询可用模型（仅网关提供）。

### 2.5 GET `/v1/models/{model_id}/capabilities`

查询单个模型能力（仅网关提供）。

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
2. 网关透传到后端 webapi
3. 后端使用参考音频 + 文本进行语音克隆合成
```

`reference_wav_path` 仅在后端本地有文件时使用，不应从外部网关传入。

## 4. 能力声明

每个 adapter 通过 `capabilities` 列表声明支持的功能：

| 能力           | 说明                     | 必需 |
| -------------- | ------------------------ | :--: |
| `tts`          | 基础文本转语音           |  是  |
| `voice_design` | 通过自然语言描述生成音色 |  否  |
| `voice_clone`  | 基于参考音频的语音克隆   |  否  |
| `emotion`      | 支持情绪/语气控制        |  否  |
| `streaming`    | 支持流式输出             |  否  |

## 5. 新引擎接入指南

创建新的 TTS 引擎适配器，按以下步骤操作：

### 5.1 创建目录结构

```
tts/YourEngine/
├── __init__.py          # 可为空
├── adapter.py           # 适配器（网关调用）
├── config.yaml          # 配置文件
├── webapi.py            # 后端 API 服务
└── voices/              # 参考音频目录
    └── readme.md
```

### 5.2 实现 adapter.py

继承 `server.base_adapter.BaseAdapter`，实现以下方法：

```python
from server.base_adapter import BaseAdapter

class YourEngineAdapter(BaseAdapter):
    name = "YourEngine"
    sample_rate = 44100
    capabilities = ["tts", "voice_clone"]
    voices = ["default"]
    clone_voices = []

    def __init__(self, config_path=None):
        # 读取 config.yaml
        ...

    def synthesize(self, text, voice="default",
                   reference_wav_bytes=None, prompt_text=None,
                   cfg_value=0.7, inference_timesteps=10, **kwargs):
        # 调用后端 webapi 的 /v1/audio/speech
        # 返回 numpy float32 数组
        ...

    def synthesize_streaming(self, text, voice="default",
                             reference_wav_bytes=None, prompt_text=None,
                             cfg_value=0.7, inference_timesteps=10, **kwargs):
        # 调用后端 webapi 的 /v1/audio/speech/stream
        # yield numpy float32 chunk
        ...

    def health_check(self):
        # 调用 GET /health
        ...
```

### 5.3 实现 webapi.py

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

### 5.4 编写 config.yaml

```yaml
backend_url: "http://localhost:8003" # 后端地址
webapi_port: 8003 # 端口
timeout: 120 # 超时(秒)
sample_rate: 44100 # 采样率  基于tts模型自身的来返回
model_path: "/path/to/model/weights" # 模型权重
source_path: "/path/to/source/code" # 源码路径
```

### 5.5 注册引擎

在 `server/config.yaml` 中添加，指定引擎名和对应的 conda 环境名：

```yaml
engines:
  - name: YourEngine
    conda_env: your_conda_env
```

- `name`：引擎目录名，对应 `tts/` 下的子目录
- `conda_env`：该引擎依赖的 conda 环境名，网关按需在此环境中启动 webapi 子进程

确保 conda 环境已创建且安装了 webapi.py 所需的全部依赖：

```bash
conda env list  # 确认环境存在
conda activate your_conda_env
pip install fastapi uvicorn ...  # 安装依赖
```

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
