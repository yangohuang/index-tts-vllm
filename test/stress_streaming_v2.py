"""Concurrent streaming stress benchmark for api_server_v2.

Launches N concurrent /tts_stream requests against a running server and
reports per-request and aggregate TTFA / elapsed / RTF.

Usage:
    python test/stress_streaming_v2.py --concurrency 4 \
        --base-url http://127.0.0.1:6006 --spk-audio assets/jay_promptvn.wav
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2

TEXTS = [
    "人工智能技术正在深刻改变我们的生活方式。从语音助手到自动驾驶，机器学习的应用已经渗透到社会的各个角落。",
    "清晨的阳光洒在湖面上，微风吹过，泛起一圈圈涟漪。远处的山峦在薄雾中若隐若现，构成一幅宁静的画卷。",
    "科学研究表明，保持规律的作息和适度的运动，对身心健康都有显著的益处，值得每个人长期坚持。",
    "这座城市的历史可以追溯到一千多年前，古老的街巷与现代的高楼交相辉映，讲述着时代变迁的故事。",
]


async def one_request(client, base_url, payload, index):
    started = time.perf_counter()
    first = None
    received = 0
    async with client.stream("POST", f"{base_url}/tts_stream", json=payload) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if first is None:
                first = time.perf_counter()
            received += len(chunk)
    finished = time.perf_counter()
    duration = received / (SAMPLE_RATE * SAMPLE_WIDTH)
    return {
        "index": index,
        "ttfa_ms": (first - started) * 1000 if first else None,
        "elapsed_ms": (finished - started) * 1000,
        "audio_s": duration,
        "rtf": (finished - started) / duration if duration else None,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:6006")
    parser.add_argument("--spk-audio", default="assets/jay_promptvn.wav")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
        payloads = [
            {"text": TEXTS[i % len(TEXTS)], "spk_audio_path": args.spk_audio}
            for i in range(args.concurrency)
        ]
        started = time.perf_counter()
        results = await asyncio.gather(*[
            one_request(client, args.base_url, payload, i)
            for i, payload in enumerate(payloads)
        ])
        wall = time.perf_counter() - started

    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    ttfa = sorted(r["ttfa_ms"] for r in results if r["ttfa_ms"])
    rtf = sorted(r["rtf"] for r in results if r["rtf"])
    total_audio = sum(r["audio_s"] for r in results)
    print(json.dumps({
        "concurrency": args.concurrency,
        "ttfa_median_ms": round(statistics.median(ttfa), 1),
        "ttfa_max_ms": round(ttfa[-1], 1),
        "rtf_median": round(statistics.median(rtf), 3),
        "rtf_max": round(rtf[-1], 3),
        "wall_s": round(wall, 2),
        "total_audio_s": round(total_audio, 1),
        "aggregate_rtf": round(wall / total_audio, 3) if total_audio else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
