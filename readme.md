# TTS 统一标准评测项目

## 测试目标

针对科技类 IT 新闻播报场景，横向对比不同 TTS 引擎的语音质量和执行效率，为最终选型提供依据。

## 参测 TTS 引擎

| 引擎 | 说明 |
|------|------|

## 测试维度

### 1. 主观听感

- **自然度**：整体语音是否接近真人播音
- **中英文混合**：中文语句中嵌入英文单词时，发音是否与当前音色一致
- **切换流畅度**：中英文切换时是否有停顿、卡顿或音色跳变
- **特殊缩写读音**：版本号（V1、V2）、型号（iPhone 15 Pro）、协议名（HTTP/3、gRPC）等能否正确朗读
- **发音准确率**：多音字、生僻字、IT 专有名词（Kubernetes、nginx 等）读音是否正确
- **数字与符号**：版本号、日期、百分比、文件大小等朗读是否准确

### 2. 性能指标

- **首包延迟（TTFB）**：从请求发出到收到第一个音频字节的时间
- **合成耗时**：完整音频生成所需时间
- **并发能力**：同时请求时的稳定性和速度
- **流式输出**：是否支持边生成边播放，合成速度是否能达到 1:1（实时率）

### 3. 功能支持

- **情绪化输出**：是否支持指定语气/情绪（如兴奋、严肃、轻松）
- **SSML 支持**：是否支持 SSML 标记语言精细控制
- **语速/音调调节**：是否支持参数调节

## 测试用例

### 文本测试（test_texts/）

| 编号 | 类型 | 状态 | 说明 |
|------|------|------|------|
| T01 | 纯中文新闻 | 待补充 | 无英文的常规科技新闻段落 |
| T02 | 中英混合新闻 | 已有 | 含产品名、版本号的科技项目介绍（URL 短链性能测试） |
| T03 | 技术术语密集 | 待补充 | 大量 IT 术语的段落 |
| T04 | 版本号与数字 | 待补充 | 包含版本号、百分比、文件大小的句子 |
| T05 | 长文本稳定性 | 待补充 | 500 字以上连续播报 |
| T06 | 情绪变化 | 待补充 | 同一段新闻分别用不同情绪合成 |

### 音色设计测试（自动，需模型支持 voice_design）

使用 `T07_voice_design.csv` 定义音色描述和对应文本（CSV 格式：第一列音色描述，第二列文本）。

预设音色：温柔御姐、甜美萝莉、沉稳男声、标准播音员。

### 语音克隆测试（自动，需模型支持 voice_clone）

使用 `T08_voice_clone.csv` 定义参考音频路径和对应文本（CSV 格式：第一列参考音频路径，第二列文本）。

## 项目目录

```text
├── readme.md
├── test_texts/                 # 输入：统一的测试文本（所有引擎共用）
│   ├── T01_chinese_news.txt
│   ├── T02_mixed_news.txt
│   ├── T03_tech_terms.txt
│   ├── T04_version_numbers.txt
│   ├── T05_long_text.txt
│   └── T06_emotion.txt
│
├── reference_audio/            # 输入：用于 voice clone 的参考音频
│   └── sample.wav
│
├── server/                     # 统一 API 网关
│   ├── main.py                 #    FastAPI 入口，暴露 POST /v1/audio/speech
│   ├── base_adapter.py         #    适配器基类
│   └── config.yaml             #    全局配置
│
├── tts/                        # 各 TTS 适配代码（模型本体不在此目录）
│   ├── voxcpm/
│   │   ├── adapter.py          #    适配器实现（调用本地/远程模型）
│   │   └── config.yaml         #    引擎配置（API Key、endpoint、模型路径等）
│   │
│   ├── qwen-tts/               # 新增引擎只需新建目录、实现 adapter.py
│   │   ├── adapter.py
│   │   └── config.yaml
│   │
│   └── .../
│
├── results/                    # 评测结果，按引擎和时间组织
│   └── voxcpm/
│       └── 2026-05-04_14-00-00/
│           ├── T02.wav         #    生成的音频
│           ├── T02.srt         #    音频字幕（文本对齐）
│           └── report.md       #    本次评测报告
│
├── client/                     # 评测客户端
│   └── runner.py               #    遍历 test_texts，逐个请求并计时、保存结果
│
└── reports/                    # 汇总报告（跨引擎对比）
```

- `tts/` 只存放适配代码和配置，模型本体（权重文件等）不在本项目中，通过 `config.yaml` 指定模型路径或 API endpoint
- `output/` 和 `results/` 是统一顶层目录，按引擎名分子目录，便于横向对比和统一管理
- 新增引擎只需新建目录、实现 `adapter.py`，注册到 server 即可

## server 和 模型 adapter 的功能

### 1. 音频生成接口（兼容 OpenAI）

`POST /v1/audio/speech` — 所有引擎统一的 TTS 合成入口，兼容 OpenAI 接口格式。

```json
{
  "model": "voxcpm",
  "input": "要合成的文本",
  "voice": "default",
  "response_format": "wav"
}
```

### 2. 查询可用模型

`GET /v1/models` — 返回当前注册的所有可用 TTS 引擎及其能力。

响应示例：

```json
{
  "models": [
    {
      "id": "voxcpm",
      "capabilities": ["tts", "voice_design", "voice_clone"],
      "voices": ["default", "male_news", "female_soft"],
      "clone_voices": ["sample"],
      "sample_rate": 48000
    }
  ]
}
```

### 3. 查询单个模型能力

`GET /v1/models/{model}/capabilities` — 返回该引擎支持的功能。

每个 adapter 需声明以下能力：

| 能力 | 说明 |
|------|------|
| `tts` | 基础文本转语音（必须） |
| `voice_design` | 通过自然语言描述生成音色 |
| `voice_clone` | 基于参考音频的语音克隆 |
| `emotion` | 支持情绪/语气控制 |
| `ssml` | 支持 SSML 标记 |
| `streaming` | 支持流式输出 |

### 4. 音色库

每个 adapter 需提供默认音色列表，分两类：

- **预设音色**：引擎内置或通过描述生成的音色（如 `default`、`male_news`）
- **克隆音色**：基于参考音频的克隆，开发者需将样例音频放入 `tts/{engine}/voices/` 目录

```
tts/VoxCPM2/
├── adapter.py
├── config.yaml
├── webapi.py
└── voices/                 # 克隆音色的样例音频
    └── sample.wav          #    开发者放入默认克隆样例
```
