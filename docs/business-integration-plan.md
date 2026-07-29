# IndexTTS2 流式服务业务接入方案

日期：2026-07-29
适用版本：本 fork 当前 master（TTFA ~365 ms，流式 RTF ~0.25，单卡 4090 实用并发 ~4 路）

## 目标场景

1. 数字人项目（实时对话驱动）
2. CIV / KYC 语音机器人（电话外呼/接听）
3. 高端理财业务（品牌音色播报、通知外呼）

三类场景共享同一核心链路：**LLM/文案 → 按句切分 → IndexTTS2 流式合成 → 播放端**。

```
LLM(对话/文案生成)──按句切分──▶ IndexTTS2 流式服务 ──▶ 播放端
                                    │                    ├─ 数字人前端(WS + PCM)
   打断信号 ◀────── cancel ──────────┘                    ├─ 电话网关(重采样 + G.711)
                                                         └─ Web/App(<audio> + wav/opus)
```

## 场景一：数字人（匹配度最高，接口基本现成）

- 传输：`WS /ws/tts_stream`（JSON `start` → 二进制 PCM 帧 → JSON `end`）。
- 口型驱动：音频驱动方案直接消费 PCM 帧的能量/频谱；如需 viseme 时间戳，
  属后续增强（见缺口 6）。
- 打断（barge-in）：发送 `{"type":"cancel"}`，服务端级联 `vLLM abort()`，
  不浪费 GPU。
- 延迟预算：TTS 占 ~0.4 s（TTFA 365 ms），端到端首响的大头在 LLM 侧；
  接入「文本增量输入」（缺口 1）后可与 LLM 生成重叠。
- 容量：单张 4090 稳定 4 路并发会话（见 docs/indextts2-streaming-results.md 迭代 5）。

## 场景二：CIV / KYC 语音机器人

- 合成侧走同一流式接口；主要增量工作在**电话网关**：
  - 线路要求 8k/16k 采样率 + G.711（μ-law/A-law）或 Opus；
  - 方案 A：网关层转码（FreeSWITCH / Asterisk / WebRTC SFU）；
  - 方案 B：服务端加 `sample_rate` 输出参数（见缺口 4）。
- **合规要点（金融审计必查）**：
  - 依据《互联网信息服务深度合成管理规定》，合成语音需显著标识
    （开场白如"您好，我是智能语音助理"）；
  - 克隆真人音色必须取得本人授权并书面留档；
  - KYC 本身是对抗语音伪造的场景，禁止将本服务用于伪装真人身份。

## 场景三：高端理财业务

- 用法：投研快讯/报告播报、专属顾问音色通知。多为整段文案，
  HTTP `format=ogg_opus` 流式拉取即可，低频场景可用非流式 `/tts_url`。
- **音色资产管理**：固定 1–2 个品牌音色参考音频并纳入版本管理；
  说话人条件缓存会使常用音色始终热缓存（TTFA 最优）。
- 情感基调：通过现有 `emo_vec` / `emo_text` 参数按业务预设（沉稳/亲和），
  见缺口 5。

## 接入捷径：OpenAI 兼容

Agent 框架（LangChain / Dify / 自研）中凡支持 OpenAI TTS 的，改 `base_url`
即接入，无需适配代码：

```python
client = OpenAI(base_url="http://<host>:6006/v1", api_key="<key>")
client.audio.speech.with_streaming_response.create(
    model="index-tts-2", voice="/path/to/ref.wav", input="...", response_format="opus")
```

## 缺口与优先级

| # | 缺口 | 需要它的场景 | 说明 |
| --- | --- | --- | --- |
| 1 | **文本增量输入**（WS 上文本分片追加，LLM 出一句合成一句） | 数字人、KYC 对话 | 当前接口需一次性给全文本；这是对话场景端到端首响的最大剩余优化 |
| 2 | 参考音频上传 / URL 接口 | 全部（跨机部署、多租户） | roadmap 已列 |
| 3 | API Key 鉴权 + 限流 | 全部（金融必须） | roadmap 已列 |
| 4 | 输出采样率参数（8k/16k 重采样） | KYC 电话 | Web/数字人不需要 |
| 5 | 情感参数业务化预设 | 理财、数字人 | 引擎已支持，封装预设即可 |
| 6 | viseme/时间戳输出 | 数字人（高级口型） | 可选增强 |

**落地顺序建议**：数字人先做 1 → 2；KYC 先做 3 → 4 → 2；理财先做 3 → 5。

## 容量与部署基线

- 单卡 RTX 4090（gpu_memory_utilization 0.25 + qwenemo 0.10，常驻 ~18 GB）：
  4 路并发流式（单流 RTF < 1），聚合吞吐 ~8× 实时。
- 并发瓶颈为 s2mel（25 步 DiT）+ BigVGAN 串行（上游已知），突破需 s2mel
  批处理或多 worker，属独立后续工作。
- 扩容模型：无状态服务，按 GPU 水平扩 + 前置负载均衡即可；说话人缓存
  按实例独立（常用音色每实例预热一次）。
