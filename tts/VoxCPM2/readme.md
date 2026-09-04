# VoxCPM2

源代码位置：/mnt/o/ai/TTS/VoxCPM/VoxCPM

模型位置：/mnt/o/ai/TTS/VoxCPM/models/VoxCPM2

Conda 环境：voxcpm

## 启动步骤

1. **激活环境**

```bash
conda activate voxcpm
cd /mnt/o/ai/TTS/TTS_Test
```

2. **启动服务**（统一端口 8002）

```bash
python -m tts.VoxCPM2.webapi
```

3. **运行评测**

```bash
python -m client.runner --engine voxcpm
```
