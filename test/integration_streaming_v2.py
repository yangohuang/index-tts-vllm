"""Opt-in GPU integration benchmark for IndexTTS2 streaming.

Targets an already-running api_server_v2.py instance; this script starts no
server. Concatenated PCM is written to a WAV file for listening checks.

Usage:
    python test/integration_streaming_v2.py \
        --base-url http://127.0.0.1:6006 \
        --spk-audio assets/jay_promptvn.wav \
        --text "..." \
        --stream-chunk-tokens 20 \
        --transport http \
        --output-wav /tmp/stream.wav
"""

import argparse
import asyncio
import json
import time
import wave

import httpx
import websockets

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2


def write_wav(path: str, pcm: bytes) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def report(started: float, first: float | None, finished: float, chunks: list[bytes]) -> dict:
    pcm = b"".join(chunks)
    duration = len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)
    stats = {
        "ttfa_ms": ((first - started) * 1000) if first is not None else None,
        "elapsed_ms": (finished - started) * 1000,
        "audio_duration_ms": duration * 1000,
        "rtf": ((finished - started) / duration) if duration else None,
        "chunks": len(chunks),
        "bytes": len(pcm),
    }
    print(json.dumps(stats, ensure_ascii=False))
    return stats


async def benchmark_http(base_url: str, payload: dict) -> tuple[dict, bytes]:
    started = time.perf_counter()
    first = None
    chunks = []
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        async with client.stream("POST", f"{base_url}/tts_stream", json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if first is None:
                    first = time.perf_counter()
                chunks.append(chunk)
    finished = time.perf_counter()
    return report(started, first, finished, chunks), b"".join(chunks)


async def benchmark_websocket(base_url: str, payload: dict) -> tuple[dict, bytes]:
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    started = time.perf_counter()
    first = None
    chunks = []
    async with websockets.connect(f"{ws_url}/ws/tts_stream", max_size=None) as ws:
        await ws.send(json.dumps({"type": "synthesize", **payload}))
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=180)
            if isinstance(message, bytes):
                if first is None:
                    first = time.perf_counter()
                chunks.append(message)
                continue
            decoded = json.loads(message)
            if decoded["type"] == "start":
                continue
            if decoded["type"] == "end":
                print("server metrics:", json.dumps(decoded.get("metrics", {})))
                break
            if decoded["type"] == "error":
                raise RuntimeError(f"server error: {decoded}")
    finished = time.perf_counter()
    return report(started, first, finished, chunks), b"".join(chunks)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:6006")
    parser.add_argument("--spk-audio", default="assets/jay_promptvn.wav")
    parser.add_argument("--text", required=True)
    parser.add_argument("--stream-chunk-tokens", type=int, default=20)
    parser.add_argument("--first-chunk-diffusion-steps", type=int, default=None)
    parser.add_argument("--transport", choices=["http", "websocket"], default="http")
    parser.add_argument("--output-wav", default=None)
    args = parser.parse_args()

    payload = {
        "text": args.text,
        "spk_audio_path": args.spk_audio,
        "stream_chunk_tokens": args.stream_chunk_tokens,
    }
    if args.first_chunk_diffusion_steps is not None:
        payload["first_chunk_diffusion_steps"] = args.first_chunk_diffusion_steps
    if args.transport == "http":
        _, pcm = await benchmark_http(args.base_url, payload)
    else:
        _, pcm = await benchmark_websocket(args.base_url, payload)

    if args.output_wav:
        write_wav(args.output_wav, pcm)
        print(f"wrote {args.output_wav}")


if __name__ == "__main__":
    asyncio.run(main())
