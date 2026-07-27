<a href="README.md">中文</a> ｜ <a href="README_EN.md">English</a>

<div align="center">

# IndexTTS-vLLM
</div>

## 项目简介
该项目在 [index-tts](https://github.com/index-tts/index-tts) 的基础上使用 vllm 库重新实现了 gpt 模型的推理，加速了 index-tts 的推理过程。

推理速度（Index-TTS-v1/v1.5）在单卡 RTX 4090 上的提升为：
- 单个请求的 RTF (Real-Time Factor)：≈0.3 -> ≈0.1
- 单个请求的 gpt 模型 decode 速度：≈90 token / s -> ≈280 token / s
- 并发量：gpu_memory_utilization 设置为 0.25（约5GB显存）的情况下，实测 16 左右的并发无压力（测速脚本参考 `simple_test.py`）

## 更新日志

- **[2025-09-22]** 支持了 vllm v1 版本，IndexTTS2 正在兼容中

- **[2025-09-28]** 支持了 IndexTTS2 的 webui 推理，并整理了权重文件，现在部署更加方便了！ \0.0/ ；但当前版本对于 IndexTTS2 的 gpt 似乎并没有加速效果，待研究

- **[2025-09-29]** 解决了 IndexTTS2 的 gpt 模型推理加速无效的问题

- **[2025-10-09]** 兼容 IndexTTS2 的 api 接口调用，请参考 [API](#api)；v1/1.5 的 api 接口以及 openai 兼容的接口可能还有 bug，晚点再修

- **[2025-10-19]** 支持 qwen0.6bemo4-merge 的 vllm 推理

- **[2026-03-03]** vllm 0.16.0 support for gpt2 inference

## TODO list
- V2 api 的并发优化：目前只有 gpt2 模型的推理是并行的，其他模块均是串行，而其中 s2mel 的推理开销大（需要 DiT 迭代 25 步），十分影响并发性能

- s2mel 的推理加速

## 使用步骤

### 1. git 本项目
```bash
git clone https://github.com/Ksuriuri/index-tts-vllm.git
cd index-tts-vllm
```


### 2. 创建并激活 conda 环境
```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
```


### 3. 安装依赖
使用强制覆盖的方式进行依赖安装，规避vllm 0.16.0与descript-audiotools 0.7.2版本中protobuf的版本冲突问题。
```bash
pip install uv
uv pip install -r requirements.txt -c overrides.txt
```


### 4. 下载模型权重

#### 自动下载（推荐）

选择对应版本的模型权重下载到 `checkpoints/` 路径下：

**From ModelScope（国内推荐）：**

```bash
# Index-TTS
modelscope download --model kusuriuri/Index-TTS-vLLM --local_dir ./checkpoints/Index-TTS-vLLM

# IndexTTS-1.5
modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM

# IndexTTS-2
modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM
```

**From Hugging Face：**

```bash
# IndexTTS-2
huggingface-cli download ksuriuri/IndexTTS-2-vLLM --local-dir ./checkpoints/IndexTTS-2-vLLM
```

#### 手动下载

- ModelScope：[Index-TTS](https://www.modelscope.cn/models/kusuriuri/Index-TTS-vLLM) | [IndexTTS-1.5](https://www.modelscope.cn/models/kusuriuri/Index-TTS-1.5-vLLM) | [IndexTTS-2](https://www.modelscope.cn/models/kusuriuri/IndexTTS-2-vLLM)
- Hugging Face：[IndexTTS-2](https://huggingface.co/ksuriuri/IndexTTS-2-vLLM)

#### 自行转换原权重（可选，不推荐）

可以使用 `convert_hf_format.sh` 自行转换官方权重文件：

```bash
bash convert_hf_format.sh /path/to/your/model_dir
```

### 5. webui 启动！

运行对应版本（第一次启动可能会久一些，因为要对 bigvgan 进行 cuda 核编译）：

```bash
# Index-TTS 1.0
python webui.py

# IndexTTS-1.5
python webui.py --version 1.5

# IndexTTS-2
python webui_v2.py
```


## API

使用 fastapi 封装了 api 接口，启动示例如下：

```bash
# Index-TTS-1.0/1.5
python api_server.py

# IndexTTS-2
python api_server_v2.py
```

### 启动参数
- `--model_dir`: 必填，模型权重路径
- `--host`: 服务ip地址，默认为 `0.0.0.0`
- `--port`: 服务端口，默认为 `6006`
- `--gpu_memory_utilization`: vllm 显存占用率，默认设置为 `0.25`

### API 请求示例
- v1/1.5 请参考 `api_example.py`
- v2 请参考 `api_example_v2.py`

### 流式 TTS（仅 IndexTTS2，`api_server_v2.py`）

`/tts_stream`（HTTP）与 `/ws/tts_stream`（WebSocket）以 token 为粒度流式返回音频：

- 输出为**无文件头**的 22050 Hz 单声道 PCM16 小端（`pcm_s16le`）裸流。
- `stream_chunk_tokens` 有效范围 **10–100**（默认 20）：值越小首包延迟（TTFA）越低，但总吞吐越差（前缀会被更频繁地重解码）。
- 情感文本模式（`emo_control_method=3`）会先调用 Qwen 情感模型，增加流式开始前的预处理延迟。
- 原有 `/tts_url` 接口保持不变，仍返回完整 WAV。

HTTP 示例（curl，输出为裸 PCM，可用 ffplay 播放）：

```bash
curl -sN http://127.0.0.1:6006/tts_stream \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，世界", "spk_audio_path": "assets/jay_promptvn.wav", "stream_chunk_tokens": 20}' \
  | ffplay -f s16le -ar 22050 -ch_layout mono -i - -autoexit -nodisp
```

HTTP 示例（Python）：

```python
import httpx

with httpx.stream("POST", "http://127.0.0.1:6006/tts_stream", json={
    "text": "你好，世界",
    "spk_audio_path": "assets/jay_promptvn.wav",
    "stream_chunk_tokens": 20,
}, timeout=180) as response:
    response.raise_for_status()
    for pcm_chunk in response.iter_bytes():
        play_or_buffer(pcm_chunk)  # 22050 Hz 单声道 PCM16 LE
```

WebSocket 示例（Python，支持中途取消）：

```python
import asyncio, json
import websockets

async def synthesize():
    async with websockets.connect("ws://127.0.0.1:6006/ws/tts_stream") as ws:
        await ws.send(json.dumps({
            "type": "synthesize",
            "request_id": "demo-1",
            "text": "你好，世界",
            "spk_audio_path": "assets/jay_promptvn.wav",
            "stream_chunk_tokens": 20,
        }))
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                play_or_buffer(message)  # 二进制帧：PCM16 LE 音频
                continue
            event = json.loads(message)  # JSON 帧：start / end / error
            if event["type"] == "end":
                print("metrics:", event["metrics"])
                break
        # 合成中途发送 {"type": "cancel"} 可取消，end 消息中 cancelled 为 true

asyncio.run(synthesize())
```

基准测试客户端参考 [`test/integration_streaming_v2.py`](test/integration_streaming_v2.py)。

### OpenAI API
- 添加 /audio/speech api 路径，兼容 OpenAI 接口
- 添加 /audio/voices api 路径， 获得 voice/character 列表

详见：[createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech)

## 新特性
- **v1/v1.5:** 支持多角色音频混合：可以传入多个参考音频，TTS 输出的角色声线为多个参考音频的混合版本（输入多个参考音频会导致输出的角色声线不稳定，可以抽卡抽到满意的声线再作为参考音频）

## 性能
Word Error Rate (WER) Results for IndexTTS and Baseline Models on the [**seed-test**](https://github.com/BytedanceSpeech/seed-tts-eval)

| model                   | zh    | en    |
| ----------------------- | ----- | ----- |
| Human                   | 1.254 | 2.143 |
| index-tts (num_beams=3) | 1.005 | 1.943 |
| index-tts (num_beams=1) | 1.107 | 2.032 |
| index-tts-vllm      | 1.12  | 1.987 |

基本保持了原项目的性能

## 并发测试
参考 [`simple_test.py`](simple_test.py)，需先启动 API 服务
