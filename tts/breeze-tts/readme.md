# breeze-tts 引擎

Breeze TTS 2 接入评测框架。与其它引擎不同，**本目录不加载模型**——`webapi.py`
是评测统一协议到 Breeze OpenAI 接口的转换代理，真正的推理在
`G:\ai\TTS\Breeze-TTS-2` 的服务里（约 6.7GB bf16 模型，RTX 3080 eager RTF≈5–9）。

## 启动（两步，顺序固定）

```bat
:: 1. 启动 Breeze 推理服务（:8000，首次加载+预热约 35 秒）
cd /d G:\ai\TTS\Breeze-TTS-2\src\test_tts && start_server.bat

:: 2. 启动本引擎的转换代理（:8010）
cd /d O:\ai\TTS\TTS_Test
python -m tts.breeze-tts.webapi --port 8010
```

## 运行评测

```bat
cd /d O:\ai\TTS\TTS_Test
python -m client.runner --engine breeze-tts
python -m client.runner --engine breeze-tts --test T02   # 只跑基础 TTS
```

输出：`results/breeze-tts/<时间戳>/`（wav + srt + report.html）。

## 协议映射

| 评测统一字段 | Breeze 字段 | 说明 |
| --- | --- | --- |
| `input` | `input` | 文本 |
| `voice`="default" | （删除） | 默认说话人 S0 |
| `voice`="S0/S1/…" | `voice` | 内置说话人 |
| `voice`=描述文本 | `instruction` + `cfg_scale=4` | 音色设计（T07） |
| `reference_audio` | `ref_audio_b64` | base64 参考音频 |
| `prompt_text` / 内置表 | `ref_text` | Breeze 要求精确文字稿；`voice_XX` 用内置表（IndexTTS 官方 cases.jsonl），其余回退 prompt_text |
| `emotion` | `instruction`=“用XX的情感演绎” | 声音导演（T06） |
| `session_id`/`session_action` | （忽略） | Breeze 无状态，长文本分段独立合成 |

## 已知限制

- Breeze 单并发：runner 串行请求没问题，但不要并行跑多个 runner
- 单段文本超过 1000 字符会被 Breeze 拒绝（runner 分段逻辑只作用于基础 TTS，克隆测试发全文）
- 超长参考音频（如 voice_06 约 30 秒）+ 长文本可能逼近模型 2048 token 上下文
