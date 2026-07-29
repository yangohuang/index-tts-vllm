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

## 迭代 4：增量前缀解码（2026-07-29）

**动机**：每轮对完整 token 前缀重解码是 O(n²) 的（见迭代 1），流式 RTF ~0.27–0.31
高于非流式的 ~0.17。

**实现**（默认开启，`incremental_decode=false` 可回退全量重解码）：

1. **mel 级缓存 + CFM prompt 复用**：`solve_euler` 原生支持把干净 mel 作为 prompt
   （原本只放说话人参考）。每轮把缓存 mel 的尾部（最多 128 帧上下文）拼在参考 mel
   之后作为 prompt，CFM 只对「重做 16 帧 + 新增帧」从噪声生成——新帧直接以已提交
   音频为条件，天然连贯；重做区覆盖 smoother 的 8 帧暂扣区。首轮无缓存时与原路径
   完全一致（TTFA 不变）。
2. **窗口化 BigVGAN + 波形缓存**：每轮只对「重做区 + 新增帧 + 16 帧左边界」声码，
   与缓存波形按精确的每帧采样数拼接；上采样因子不整除时自动回退全量声码。

**性能剖析**（长句流式，每轮各阶段耗时）：

| 阶段 | 每轮耗时 | 说明 |
| --- | --- | --- |
| GPT forward（全量） | 0.02–0.07 s | 远比预想便宜，无需增量化 |
| CFM 25 步 | 0.23–0.70 s | 绝对大头；mu 中 620 帧是 7.2 s 参考音频的固定 prompt |
| BigVGAN（窗口化后） | 0.03–0.09 s | 窗口化前随前缀增长至 ~0.13 s |

三个此前未知的事实：(a) GPT forward 很便宜，O(n²) 主项只有 CFM/vocoder；
(b) vLLM 背压使每句只有 ~4 轮解码、每轮新增 ~300 帧，前缀窗口化的边际收益有限；
(c) **每轮 CFM 成本的主导项是参考音频的固定 prompt（620 帧），不是前缀长度**。

**实测**（环境同前，丢弃预热取 3 次中位数，HTTP，默认参数）：

| 文本 | 全量重解码 RTF | 增量解码 RTF | TTFA |
| --- | --- | --- | --- |
| 标准多句（~24 s） | 0.274 | **0.248**（−10%） | 466 ms（不变） |
| 长单句逗号连接（~22 s） | 0.283 | **0.253**（−11%） | 466 ms（不变） |

对比迭代 2 时记录的流式 RTF 0.31，累计降幅约 20%。

**质量验证**（vLLM 采样按会话内请求序号确定种子，同序号请求 token 完全一致，
可逐波形对比；注意不同序号的请求是不同采样，停顿/语速本来就不同，不能用于对比）：
增量 vs 全量（同 token）逐采样等长，能量轮廓相关 0.93–0.96，RMS 差异 <2%
（尾部局部差异来自 CFM 重做区噪声采样），零削波；试听对照
`outputs/bench/final_std_r1.wav`（增量）vs `ab_full_std.wav`（全量），
人工试听确认无可辨差异（2026-07-29）。

**负结果：参考 prompt 截短（已回退）**。既然参考 prompt 是每轮主导成本，曾尝试在
有 ≥64 帧自体上下文后把参考截到最后 256 帧：RTF 降至 0.224（−18%），但整体 RMS
升高 15–20%，尾部能量逐轮爬升（4135 vs 2732）——自体上下文 + 弱化参考锚形成
增益正反馈。结论：参考 mel 必须全程全量保留；剩余的优化空间在于缩短参考音频本身
（部署侧选择 3–4 s 参考）或研究 DiT 对固定 prompt 段的跨轮缓存。

## 迭代 3：首块低步数 CFM（2026-07-28）

**动机**：TTFA 中最大的一块是首个前缀的解码（25 步 CFM 占大头）。首块只决定开头
约 0.4 s 的可听内容，感知上对扩散步数最不敏感。

**实现**：`stream_infer(first_chunk_diffusion_steps=15)`（范围 5–25，25 等价于关闭）。
仅整个 request 的**第一次**前缀解码用低步数，之后所有轮次回到 25 步；首块尾部
93 ms 不稳定区本来就会在下一轮以全步数重解码并交叉淡化，所以低步数只影响首块的
已提交区，且拼接边界仍由 smoother 保证连续。HTTP/WS 均可通过
`first_chunk_diffusion_steps` 字段调节，HTTP 响应带 `X-First-Chunk-Diffusion-Steps` 头。

**实测**（2026-07-28，环境同前两轮；约 130 字中性中文段落（合成音频约 23–26 s），
HTTP，chunk tokens 20，热缓存，丢弃 1 次预热后取 3 次中位数）：

| first_chunk_diffusion_steps | TTFA (ms) | RTF | 说明 |
| --- | --- | --- | --- |
| 25（关闭优化，基线） | 660 | 0.273 | 与迭代 2 的 ~605 ms 同量级（文本不同） |
| **15（默认）** | **467** | 0.274 | **达成 < 500 ms 目标** |
| 10 | 365 | 0.303 | 延迟敏感场景可选 |

2026-07-29 复测：HTTP 三组与上表完全复现（±3 ms）；WebSocket 15 步客户端
TTFA 中位 **448 ms**（服务端指标 ~442 ms），两种传输均达标。

**默认值调整（2026-07-29）**：三种步数的首块试听对比（`steps25/15/10_r1.wav`）
确认 10 步与 25 步无可辨差异，默认值从 15 改为 **10**；叠加增量解码后默认配置
实测 TTFA **~362 ms**。

- 首块质量抽检：三种步数下零削波，开头 0.5 s RMS 相当（2100–2600），
  首秒最大采样间跳变 0.20–0.26 满幅，与基线同量级，无可测的拼接退化。
- 三种配置合成时长一致（23–26 s，采样自然波动），无缺尾。
- TTFA 对步数近似线性（25→15→10 步：660→467→365 ms），符合首块解码
  以 CFM 迭代为主的预期；RTF 基本不变，因为只有首块降步。

## 迭代 2：说话人条件缓存（2026-07-27）

**动机**：TTFA ~720–760 ms 中，每次请求都重复计算参考音频条件
（w2v-bert 特征 → 语义码量化 → campplus 声纹 → length regulator）。

**实现**：`IndexTTS2._get_speaker_conditioning` / `_get_emo_conditioning_emb` 以
`(绝对路径, mtime)` 为 key 的 LRU 缓存（默认 20 条，文件被覆盖后 mtime 变化自动失效），
缓存 `spk_cond_emb / prompt_condition / ref_mel / style` 与情感参考的 `emo_cond_emb`。
非流式 `infer()` 与流式 `stream_infer()` 共享（都经 `_prepare_inference`）。

**实测**（同上环境与文本，HTTP，预热服务器后测量）：

| 传输 | 场景 | chunk tokens | TTFA (ms) | 对比迭代 1 |
| --- | --- | --- | --- | --- |
| HTTP | 冷缓存（该说话人首次请求） | 10 | 858 | 与无缓存持平 |
| HTTP | **热缓存** | 10 | **605** | 732 → 605（-17%） |
| HTTP | **热缓存** | 20 | **632** | 759 → 632（-17%） |
| WS | 冷缓存 | 10 | 833 | 与无缓存持平 |
| WS | **热缓存** | 10 | **600** | 712 → 600（-16%） |
| WS | **热缓存** | 20 | **624** | 759 → 624（-18%） |

RTF 与音频时长不变（~0.30 / ~24 s）。缓存在引擎层（`_prepare_inference`），
对 HTTP / WebSocket / 非流式 `/tts_url` / WebUI 全局共享且互通：任一入口用过的
参考音频，其余入口直接命中。WS `end` 消息内的服务端指标（如 TTFA 622 ms）与
客户端实测（624 ms）一致。

**结论**：参考音频预处理实际耗时 ~250 ms（低于最初 300–450 ms 的估计），缓存后如数
兑现；剩余 ~600 ms 由 vLLM prefill+首批 token（~200 ms）与首块 CFM+BigVGAN 解码
（~400 ms）构成。若需进一步压低 TTFA，下一个杠杆是**首块降低 CFM 步数**（25 → 10–15，
预估再省 100–200 ms）或裁短参考音频（prompt_condition 变短使每块 CFM 都变快）。

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
