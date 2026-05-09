# MOSS-TTSD 引擎


## 模型能力

| 能力 | 说明 | 使用的模型 |
|------|------|-----------|
| `tts` | 基础文本转语音，支持 1-5 位说话人（通过 `[S1]`-`[S5]` 标签） | MOSS-TTSD-v1.0 |
| `voice_clone` | 基于参考音频的语音克隆，使用 `voice_clone_and_continuation` 模式 | MOSS-TTSD-v1.0 |
| `voice_design` | 基于文本描述的音色设计，通过 `instruction` 字段指定音色 | MOSS-VoiceGenerator |

- **不支持** 真正的流式输出（先整体推理再分块返回）

## Conda 环境

```
conda_env: moss-tts
```

模型代码和权重位于 `G:\ai\TTS\MOSS-TTSD\`，在 WSL2 下运行，路径自动转换为 `/mnt/g/` 前缀。

## 目录结构

```
tts/MOSS-TTSD/
├── __init__.py
├── adapter.py       # 网关适配器
├── webapi.py        # FastAPI 后端服务
├── config.yaml      # 引擎配置
├── readme.md        # 本文件
├── demo_clone_v2.py       # 语音克隆独立 demo
├── demo_voice_design.py   # 音色设计独立 demo
└── voices/                # 参考音频目录
```

## 启动方法

### 通过网关（推荐）

网关会按需自动启动后端，无需手动操作：

```bash
cd /mnt/o/ai/TTS/TTS_Test
python -m server.main
```

### 手动启动后端

```bash
conda activate moss-tts
cd /mnt/o/ai/TTS/TTS_Test
python -m tts.MOSS-TTSD.webapi --port 8004
```

### 通过 init_model 接口预热

```bash
curl -X POST http://localhost:9000/v1/init_model \
  -H "Content-Type: application/json" \
  -d '{"model": "MOSS-TTSD"}'
```

## 调用方法

以下示例直接调用 webapi 后端（端口 8004），用于单独测试本引擎。通过网关调用时改为端口 9000 并加上 `"model": "MOSS-TTSD"` 字段即可。

### 基础 TTS

```bash
curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MOSS-TTSD",
    "input": "今天天气不错，适合出去走走。",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output output.wav
```

### 多说话人对话

文本中使用 `[S1]`-`[S5]` 标签指定说话人：

```bash
curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MOSS-TTSD",
    "input": "[S1] 你好，最近怎么样？[S2] 挺好的，谢谢关心！",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output dialogue.wav
```

### 语音克隆

使用参考音频进行语音克隆（`reference_audio` 为 Base64 编码，`prompt_text` 为参考音频的文本内容）：

```bash
# 生成请求体 JSON 文件
python3 -c "
import base64, json
with open('reference_audio/老子从来就没想刮穷鬼的钱.wav', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
body = {
    'model': 'MOSS-TTSD',
    'input': '怎么才七成啊？',
    'voice': 'default',
    'response_format': 'wav',
    'reference_audio': b64,
    'prompt_text': '老子从来就没想刮穷鬼的钱'
}
with open('/tmp/clone_req.json', 'w') as f:
    json.dump(body, f)
"

curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d @/tmp/clone_req.json \
  --output clone.wav
```

### 音色设计

通过 `instruction` 字段描述想要的音色，无需参考音频：

```bash
curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MOSS-TTSD",
    "input": "夜色温柔，月光如水，让人不禁想起远方的亲人",
    "instruction": "温柔磁性的女声，适合播读情感类文章",
    "response_format": "wav"
  }' \
  --output design.wav
```

VoiceGenerator 官方推荐参数：`temperature=1.5, top_p=0.6`（有 instruction 时自动使用）。

### 推理参数调整

```bash
curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MOSS-TTSD",
    "input": "这是一段测试文本。",
    "temperature": 0.9,
    "top_p": 0.8,
    "repetition_penalty": 1.1
  }' \
  --output output.wav
```

## 模型加载策略

两个模型按需加载（懒加载），不会同时占用显存：

- **MOSS-TTSD-v1.0**：首次 tts / voice_clone 请求时加载
- **MOSS-VoiceGenerator**：首次 voice_design（有 instruction）请求时加载
- 加载后常驻显存，后续请求直接使用

## 健康检查

```bash
curl http://localhost:8004/health
```

返回示例：

```json
{"status": "ok", "model": "MOSS-TTSD", "capabilities": ["tts", "voice_clone", "voice_design"], "sample_rate": 24000}
```
