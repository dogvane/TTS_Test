# TTS 统一标准评测项目

## 测试目标

针对科技类 IT 新闻播报场景，横向对比不同 TTS 引擎的语音质量和执行效率，为最终选型提供依据。

## 参测 TTS 引擎

| 引擎名（`--engine`） | model id | 目录 | 环境 | 能力 | 说明 |
|------|------|------|------|------|------|
| voxcpm | VoxCPM2 | tts/VoxCPM2/ | WSL2 + conda(voxcpm) | tts / voice_design / voice_clone / streaming | 扩散模型，音色设计强 |
| moss-tts | MOSS-TTSD | tts/MOSS-TTSD/ | WSL2 + conda(moss-tts) | tts / voice_clone / voice_design | 双模型路由（TTSD + VoiceGenerator） |
| qwen3-tts | qwen3-tts | tts/qwen3-tts/ | WSL2 + conda(qwen3-tts) | tts / voice_design / voice_clone | 9 种预设音色 |
| higgs-audio | higgs-audio | tts/higgs-audio/ | Docker（外部后端） | tts / voice_clone | SGLang-Omni，外部 HTTP |
| index-tts | IndexTTS2 | tts/IndexTTS25/ | **Windows venv**（外部后端） | tts / voice_clone / **emotion** / streaming | 情绪表达 + 时长控制 + 零样本克隆 |

> 本项目**没有统一网关**：每个引擎目录下的 `webapi.py` 独立暴露 OpenAI 兼容的 `POST /v1/audio/speech`，需按各引擎目录下的 readme.md 手动启动（WSL2 conda / Docker / Windows venv）。所有引擎统一监听 **8002** 端口，`client/config.yaml` 中的 `base_url` 已全部指向 `http://localhost:8002`，**同一时刻只运行一个引擎**。

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
| T02 | 中英混合新闻 | 已有 | 含产品名、版本号的科技项目介绍（URL 短链性能测试），`T02_mixed_news.txt` |

> 文本测试目前仅保留 T02 作为基础 TTS 基线（default 音色）。原规划的 T01/T03/T04/T05 已迁移为克隆模式用例（见下方 T09），T06 迁移为情绪克隆用例（见下方 T06_emotion_clone.csv）。

### 音色设计测试（自动，需模型支持 voice_design）

使用 `T07_voice_design.csv` 定义音色描述和对应文本（CSV 格式：第一列音色描述，第二列文本）。

预设音色：温柔御姐、甜美萝莉、沉稳男声、标准播音员。

### 语音克隆测试（自动，需模型支持 voice_clone）

使用以下 CSV 定义参考音频路径和对应文本（CSV 格式：第一列参考音频路径，第二列文本）：

- `T08_voice_clone.csv`：基础克隆用例，验证克隆音色对原台词的复述能力（使用 `reference_audio/` 下两段《让子弹飞》台词音频）。
- `T09_voice_clone_texts.csv`：多音色 × 多文本特征克隆，4 个用例各配一个不同音色的参考音频，考察克隆音色在不同文本特征下的表现（参考音频取自 `reference_audio/indextts/` 官方示例）：

  | 用例 | 参考音频 | 考察点 |
  |------|----------|--------|
  | T01 | reference_audio/indextts/voice_05.wav | 无英文基线、克隆自然度 |
  | T03 | reference_audio/indextts/voice_03.wav | IT 专有名词发音（Kubernetes / gRPC / Nginx 等） |
  | T04 | reference_audio/indextts/voice_04.wav | 版本号与数字朗读（v3.2.1、240%、2.5 TB） |
  | T05 | reference_audio/indextts/voice_06.wav | 长文本（500 字+）音色稳定性 |

### 情绪克隆测试（自动，需模型支持 voice_clone + emotion）

使用 `T06_emotion_clone.csv`，CSV 格式为 3 字段：参考音频, 情绪, 说话内容。

固定一个中性参考音频（`reference_audio/indextts/voice_03.wav`）与同一段 IT 文本，扫描 6 种情绪：兴奋、严肃、轻松、中性、悲伤、愤怒。目的是隔离「情绪」单一变量，横向对比各引擎的情绪控制能力。

## 参考音频说明

- `reference_audio/` 根目录下为《让子弹飞》台词片段等实际音频，供 T08 基础克隆测试使用。
- T09（多音色克隆）与 T06（情绪克隆）的参考音频已改为使用 `reference_audio/indextts/` 下的 IndexTTS 官方示例音频（见下节），原先规划的 `ref_it_male_news.wav` 等占位文件不再需要。

### IndexTTS 官方示例音频（reference_audio/indextts/）

来自 IndexTTS-2.5 官方仓库的示例（源目录 `G:/ai/TTS/index-tts/IndexTTS-2.5/index-tts/examples/`），共 13 个 wav：

- `voice_01.wav ~ voice_12.wav`：**音色参考音频**，覆盖中英文、长短文本、多种音色（相声、播报、影视剧台词等），可用作克隆测试的参考音色。
- `emo_sad.wav`、`emo_hate.wav`：**情绪参考音频**（悲伤、厌恶），供 IndexTTS 的「情感参考音频」控制方式使用。

每个文件的配套合成文本与情绪控制参数标注见 [reference_audio/indextts/voice.md](reference_audio/indextts/voice.md)（依据官方 `cases.jsonl` 与 `webui.py` 整理）。

## 项目目录

```text
├── readme.md
├── test_texts/                 # 输入：统一的测试文本（所有引擎共用）
│   ├── T02_mixed_news.txt          # 基础 TTS 基线（中英混合新闻）
│   ├── T07_voice_design.csv        # 音色设计：音色描述,文本
│   ├── T08_voice_clone.csv         # 基础克隆：参考音频,文本
│   ├── T09_voice_clone_texts.csv   # 多音色克隆：T01/T03/T04/T05 迁移用例
│   └── T06_emotion_clone.csv       # 情绪克隆：参考音频,情绪,文本
│
├── reference_audio/            # 输入：用于 voice clone 的参考音频
│   ├── sample.wav
│   └── indextts/                   # IndexTTS-2.5 官方示例音频（voice_01~12 音色参考 + emo_* 情绪参考）
│       └── voice.md                #    各音频的用途、配套文本与情绪参数标注
│
├── tts/                        # 各 TTS 引擎（模型本体不在此目录），webapi 统一使用端口 8002
│   ├── VoxCPM2/
│   ├── higgs-audio/                # Docker 外部后端（容器映射到宿主 8002）
│   ├── MOSS-TTSD/
│   ├── qwen3-tts/
│   └── IndexTTS25/                 # Windows venv 外部后端
│
├── results/                    # 评测结果，按引擎和时间组织
│   └── voxcpm/
│       └── 2026-05-04_14-00-00/
│           ├── T02.wav         #    生成的音频
│           ├── T02.srt         #    音频字幕（文本对齐）
│           └── report.html     #    本次评测报告
│
└── client/                     # 评测客户端
    ├── runner.py               #    遍历 test_texts，直连各引擎请求并计时、保存结果
    └── config.yaml             #    引擎注册表：引擎名 → base_url / model / capabilities
```

- `tts/` 只存放各引擎的 webapi 代码和配置，模型本体（权重文件等）不在本项目中
- `results/` 按引擎名分子目录，便于横向对比和统一管理
- 新增引擎只需新建目录、实现 `webapi.py`（暴露 `/v1/audio/speech` 和 `/health`），并在 `client/config.yaml` 注册即可

## 使用方式

### 1. 启动引擎 webapi

按所需引擎目录下的 readme.md 启动对应后端，统一监听 **8002** 端口（各 webapi.py 的默认端口即为 8002）：

- VoxCPM2：WSL2 中 `conda activate voxcpm && python tts/VoxCPM2/webapi.py`
- IndexTTS2：Windows venv 中按 `tts/IndexTTS25/readme.md` 启动
- higgs-audio：Docker 启动 SGLang-Omni，容器端口映射到宿主 8002

> 同一时刻只运行一个引擎，端口冲突时先停掉前一个。

### 2. 运行评测客户端

```bash
python -m client.runner --engine voxcpm            # 全部测试
python -m client.runner --engine voxcpm --test T02 # 指定测试用例
```

`client/config.yaml` 中所有引擎的 `base_url` 均已统一为 `http://localhost:8002`；临时换地址时可用 `--base-url` 覆盖。

客户端会先请求 `{base_url}/health` 确认引擎在线，再按 `client/config.yaml` 中声明的能力自动执行对应测试。

## 接口约定（兼容 OpenAI）

每个引擎的 `webapi.py` 暴露 OpenAI 兼容接口：

### 1. 音频生成

`POST /v1/audio/speech`

```json
{
  "model": "VoxCPM2",
  "input": "要合成的文本",
  "voice": "default",
  "response_format": "wav",
  "reference_audio": "<base64 参考音频，克隆时可选>",
  "prompt_text": "<参考音频的转写文本，可选>",
  "emotion": "<情绪，可选>",
  "session_id": "<多段合成的会话 id，可选>",
  "session_action": "start | continue | end"
}
```

返回 WAV 音频字节流。

### 2. 健康检查

`GET /health` — 返回 `{"status": "ok", "model": "...", "sample_rate": ...}`。

### 能力声明

`client/config.yaml` 中每个引擎声明以下能力：

| 能力 | 说明 |
|------|------|
| `tts` | 基础文本转语音（必须） |
| `voice_design` | 通过自然语言描述生成音色 |
| `voice_clone` | 基于参考音频的语音克隆 |
| `emotion` | 支持情绪/语气控制 |
| `streaming` | 支持流式输出（`/v1/audio/speech/stream`） |
