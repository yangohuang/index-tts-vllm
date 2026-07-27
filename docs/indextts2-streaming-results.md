# IndexTTS2 Token 流式推理：实现与测量结果

日期：2026-07-27
分支：`dev_stream`

## 架构

vLLM 按 request 暴露**累积声学 token 流**（`UnifiedVoice._stream_generated_codes`），
`IndexTTS2.stream_infer()` 在新增 token 达到 `stream_chunk_tokens` 阈值时，将**当前完整
token 前缀**重新走一遍既有的非流式解码路径（GPT forward → `gpt_layer` → 语义码嵌入 →
length regulator → 25 步 CFM → BigVGAN），得到完整前缀波形；request 级
`StreamingPcmSmoother` 只提交新增的稳定部分：

- 尾部 `hop_length × 8` 个采样（约 93 ms）作为"不稳定区"暂扣，下一个前缀会带着更多右侧
  上下文重新解码它；
- 新旧前缀在暂扣区用等功率（cos/sin）曲线交叉淡化，抑制边界爆音；
- 收到 stop token（或流意外结束时的兜底 flush）提交全部剩余音频，保证结尾完整。

多句文本逐句流式合成，句间插入 200 ms 静音；整个 request 用同一个 vLLM request id，
取消/断连时在 `finally` 中调用 `AsyncLLM.abort()` 释放推理资源。

前缀重解码是 O(n²) 的：这是用**不改动模型**换**流式输出**的代价，体现在 RTF 上
（见下表，流式 ≈ 0.31 vs 非流式 ≈ 0.17），但换来首音频延迟从 ~4 s 降到 ~0.7 s。

## API

两种传输共享同一 Pydantic 请求模型（`TtsStreamRequest`）与参数映射：

| 端点 | 协议 | 输出 |
| --- | --- | --- |
| `POST /tts_stream` | HTTP chunked | 裸 PCM 流 + `X-Audio-*` 响应头 |
| `WS /ws/tts_stream` | WebSocket | JSON `start` → 二进制 PCM 帧 → JSON `end`（含指标）；支持 `{"type":"cancel"}` 中途取消 |

- 音频格式：**无文件头** 22050 Hz 单声道 PCM16 小端（`pcm_s16le`）。
- `stream_chunk_tokens`：10–100，默认 **20**。
- 错误时 WebSocket 返回 `{"type":"error","error_type":"invalid_request"|"inference_error",...}`，不暴露 traceback。
- 原 `/tts_url`（完整 WAV）不变。

## 基准测量

环境：单卡 24 GB（RTX 4090 级），vLLM 0.16，`--is_fp16 --gpu_memory_utilization 0.25
--qwenemo_gpu_memory_utilization 0.10`；参考音频 `assets/jay_promptvn.wav`；
约 130 字中性中文段落（合成音频约 24 s）；每组丢弃 1 次预热后取 3 次测量的中位数。

| 传输 | chunk tokens | TTFA (ms) | 总耗时 (ms) | 音频时长 (s) | RTF | 边界跳变 max/mean（满幅） |
| --- | --- | --- | --- | --- | --- | --- |
| HTTP | 10 | 732 | 7270 | 23.3 | 0.312 | 0.059 / 0.009 |
| WS | 10 | 712 | 7543 | 24.8 | 0.304 | 0.020 / 0.008 |
| HTTP | 20 | 759 | 7606 | 24.6 | 0.310 | 0.035 / 0.014 |
| WS | 20 | 759 | 7404 | 24.1 | 0.308 | 0.009 / 0.002 |
| HTTP | 40 | 857 | 7318 | 23.8 | 0.301 | 0.020 / 0.010 |
| WS | 40 | 879 | 7941 | 25.1 | 0.313 | 0.017 / 0.007 |

- 非流式对照（`/tts_url`，同文本同参考）：总耗时 4.02 s，音频 23.05 s，RTF ≈ 0.17，
  但客户端要等 4 s 才拿到第一字节；流式首音频 **0.7–0.9 s**，快 5 倍以上。
- 削波采样占比全部为 0；峰值整卡显存 ~19.6 GiB（含服务本身 ~18 GiB 常驻）。
- 流式与非流式时长一致（23–25 s，采样开启导致自然波动），无缺尾。
- vLLM 产 token 快于前缀解码，形成天然背压：实际每次解码的增量往往大于阈值，
  WebSocket 实测每请求约 13 帧。

## 默认值选择

默认 `stream_chunk_tokens = 20`：TTFA 与 10 几乎相同（≈760 vs ≈730 ms），
边界不连续度更稳定，重解码次数更少。对延迟极端敏感可用 10，注重吞吐可用 40+。

## 已知限制

- 前缀重解码为 O(n²)，长句吞吐劣于非流式路径（RTF 0.31 vs 0.17）；超长文本请依赖
  `max_text_tokens_per_sentence` 分句摊薄。
- 采样非确定：流式与非流式输出不逐位一致，仅整体时长/内容一致。
- 边界平滑靠 ~93 ms 交叉淡化，客观跳变很小（mean ≤ 0.014 满幅），
  但严格的听感验收（爆音/重复/缺尾）仍建议人工抽听 `outputs/stream_*.wav`。
- 情感文本模式（`emo_control_method=3`）先经 Qwen 情感模型，流开始前有额外预处理延迟。

## 复现

```bash
# 启动服务
python api_server_v2.py --host 127.0.0.1 --port 6006 \
  --model_dir checkpoints/IndexTTS-2-vLLM --is_fp16 \
  --gpu_memory_utilization 0.25 --qwenemo_gpu_memory_utilization 0.10

# HTTP 基准
python test/integration_streaming_v2.py --transport http \
  --spk-audio assets/jay_promptvn.wav --stream-chunk-tokens 20 \
  --text "……足够长的中文段落……" --output-wav outputs/stream_http_20.wav

# WebSocket 基准
python test/integration_streaming_v2.py --transport websocket \
  --spk-audio assets/jay_promptvn.wav --stream-chunk-tokens 20 \
  --text "……足够长的中文段落……" --output-wav outputs/stream_ws_20.wav

# 单元测试（CPU）
python -m pytest test -v
```
