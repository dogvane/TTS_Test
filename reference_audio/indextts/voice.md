# IndexTTS 官方示例音频（reference_audio/indextts）

来源：`G:/ai/TTS/index-tts/IndexTTS-2.5/index-tts/examples/`（IndexTTS-2.5 官方仓库示例）。
依据该目录的 `cases.jsonl` 与仓库 `webui.py`（`EMO_CHOICES_ALL`、情感向量 Slider 定义）整理。

IndexTTS-2.5 的每个示例 = **音色参考音频（prompt_audio）+ 合成文本 + 情绪控制方式**。
本目录中的 `voice_XX.wav` 即音色参考音频，`emo_*.wav` 为情绪参考音频。

## 情绪控制方式（emo_mode）

| emo_mode | 方式 | 说明 |
|----------|------|------|
| 0 | 与音色参考音频相同 | 情绪跟随音色参考音频，不做额外控制 |
| 1 | 使用情感参考音频 | 另给一段 `emo_audio`，按 `emo_weight` 混合其情绪 |
| 2 | 使用情感向量控制 | 八维向量 [喜, 怒, 哀, 惧, 厌恶, 低落, 惊喜, 平静]，各 0~1 |
| 3 | 使用情感描述文本控制 | 用 `emo_text` 描述情绪（实验特性，webui 默认隐藏） |

## 音频清单

| 文件 | 用途 | 合成文本（cases.jsonl） | 情绪控制 |
|------|------|--------------------------|----------|
| voice_01.wav | 音色参考 | Translate for me, what is a surprise! | mode 0：与参考音频相同 |
| voice_02.wav | 音色参考 | The palace is strict, no false rumors, Lady Qi! | mode 0 |
| voice_03.wav | 音色参考 | 这个呀，就是我们精心制作准备的纪念品，大家可以看到这个色泽和这个材质啊，哎呀多么的光彩照人。 | mode 0 |
| voice_04.wav | 音色参考 | 你就需要我这种专业人士的帮助，就像手无缚鸡之力的人进入雪山狩猎，一定需要最老练的猎人指导。 | mode 0 |
| voice_05.wav | 音色参考 | 在真正的日本剑道中，格斗过程极其短暂……（长文本，剑道对决描述） | mode 0 |
| voice_06.wav | 音色参考 | 今天呢，咱们开一部新书，叫《赛博朋克二零七七》……（长文本，相声风格） | mode 0 |
| voice_07.wav | 音色参考 | 酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。 | mode 1：情绪参考 `emo_sad.wav`，weight 0.65（悲伤） |
| voice_08.wav | 音色参考 | 你看看你，对我还有没有一点父子之间的信任了。 | mode 1：情绪参考 `emo_hate.wav`，weight 0.65（厌恶） |
| voice_09.wav | 音色参考 | 对不起嘛！我的记性真的不太好，但是和你在一起的事情，我都会努力记住的~ | mode 2：情感向量 哀=0.8，weight 0.8 |
| voice_11.wav | 音色参考 | 这些年的时光终究是错付了... | mode 3：情绪文本「极度悲伤」 |
| voice_12.wav | 音色参考 | 快躲起来！是他要来了！他要来抓我们了！ | mode 3：情绪文本「You scared me to death! What are you, a ghost?」 |
| emo_sad.wav | **情绪参考音频** | — | 供 mode 1 使用：悲伤情绪样例（配 voice_07） |
| emo_hate.wav | **情绪参考音频** | — | 供 mode 1 使用：厌恶情绪样例（配 voice_08） |

## 其他文件

- 原目录中的 `cases.jsonl`（示例元数据）与 `batch/`（批量合成文本）未复制到本项目。
