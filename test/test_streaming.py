import asyncio
import struct

import numpy as np
import pytest
import torch

from indextts.streaming import (
    DEFAULT_FIRST_CHUNK_DIFFUSION_STEPS,
    DEFAULT_STREAM_CHUNK_TOKENS,
    FULL_DIFFUSION_STEPS,
    IncrementalMelCache,
    MAX_STREAM_CHUNK_TOKENS,
    MIN_STREAM_CHUNK_TOKENS,
    PCM16_SAMPLE_WIDTH,
    StreamMetrics,
    StreamingPcmSmoother,
    pcm16le_bytes,
    plan_mel_window,
    stream_ogg_opus,
    validate_first_chunk_diffusion_steps,
    validate_stream_chunk_tokens,
    validate_stream_format,
    wav_stream_header,
)


@pytest.mark.parametrize("value", [10, 20, 100])
def test_validate_stream_chunk_tokens_accepts_supported_values(value):
    assert validate_stream_chunk_tokens(value) == value


@pytest.mark.parametrize("value", [9, 101, 20.5, "20", True])
def test_validate_stream_chunk_tokens_rejects_unsupported_values(value):
    with pytest.raises(ValueError):
        validate_stream_chunk_tokens(value)


@pytest.mark.parametrize("value", [5, DEFAULT_FIRST_CHUNK_DIFFUSION_STEPS, FULL_DIFFUSION_STEPS])
def test_validate_first_chunk_diffusion_steps_accepts_supported_values(value):
    assert validate_first_chunk_diffusion_steps(value) == value


@pytest.mark.parametrize("value", [4, 26, 15.0, "15", True])
def test_validate_first_chunk_diffusion_steps_rejects_unsupported_values(value):
    with pytest.raises(ValueError):
        validate_first_chunk_diffusion_steps(value)


def test_plan_mel_window_first_round_decodes_everything():
    plan = plan_mel_window(cached_frames=0, total_frames=40)
    assert plan.keep_frames == 0
    assert plan.window_start == 0
    assert plan.context_frames == 0
    assert plan.generated_frames == 40


def test_plan_mel_window_redoes_tail_and_limits_context():
    plan = plan_mel_window(
        cached_frames=300, total_frames=340, context_frames=128, redo_frames=16
    )
    assert plan.keep_frames == 284  # 300 - 16 redo
    assert plan.window_start == 156  # 284 - 128 context
    assert plan.context_frames == 128
    assert plan.generated_frames == 56  # 16 redo + 40 new


def test_plan_mel_window_short_cache_keeps_nothing():
    plan = plan_mel_window(
        cached_frames=10, total_frames=50, context_frames=128, redo_frames=16
    )
    assert plan.keep_frames == 0
    assert plan.window_start == 0
    assert plan.generated_frames == 50


def test_plan_mel_window_never_keeps_more_than_total():
    plan = plan_mel_window(
        cached_frames=100, total_frames=60, context_frames=128, redo_frames=16
    )
    assert plan.keep_frames == 60
    assert plan.generated_frames == 0


def test_incremental_mel_cache_reports_frames():
    cache = IncrementalMelCache()
    assert cache.frames == 0
    cache.mel = torch.zeros(1, 80, 37)
    assert cache.frames == 37


def test_validate_stream_format():
    for value in ("pcm", "wav", "ogg_opus"):
        assert validate_stream_format(value) == value
    with pytest.raises(ValueError):
        validate_stream_format("mp3")


def test_wav_stream_header_fields():
    header = wav_stream_header()
    assert len(header) == 44
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    assert header[4:8] == b"\xff\xff\xff\xff"  # unknown-length sentinel
    fmt = struct.unpack("<IHHIIHH", header[16:36])
    assert fmt == (16, 1, 1, 22050, 22050 * 2, 2, 16)
    assert header[36:40] == b"data" and header[40:44] == b"\xff\xff\xff\xff"


def test_stream_ogg_opus_produces_ogg_pages():
    async def pcm():
        # 0.5 s of silence in two chunks
        for _ in range(2):
            yield b"\x00\x00" * (22050 // 4)

    async def run():
        chunks = []
        async for data in stream_ogg_opus(pcm()):
            chunks.append(data)
        return b"".join(chunks)

    encoded = asyncio.run(run())
    assert encoded[:4] == b"OggS"
    assert b"OpusHead" in encoded[:200]
    assert b"OpusTags" in encoded


def test_pcm16le_bytes_clips_and_encodes_little_endian():
    waveform = torch.tensor([-40000.0, -1.0, 0.0, 1.0, 40000.0])
    encoded = pcm16le_bytes(waveform)
    decoded = np.frombuffer(encoded, dtype="<i2")
    np.testing.assert_array_equal(decoded, [-32767, -1, 0, 1, 32767])
    assert len(encoded) % PCM16_SAMPLE_WIDTH == 0


def test_smoother_holds_overlap_until_next_prefix_and_flushes_tail():
    smoother = StreamingPcmSmoother(hop_length=4, overlap_mel_frames=2)

    first = smoother.push(torch.arange(16, dtype=torch.float32), is_final=False)
    second = smoother.push(torch.arange(24, dtype=torch.float32), is_final=False)
    tail = smoother.flush()

    assert first.numel() == 8
    assert second.numel() > 0
    assert tail.numel() == 8
    assert smoother.flush().numel() == 0


def test_smoother_rejects_prefix_shorter_than_committed_audio():
    smoother = StreamingPcmSmoother(hop_length=4, overlap_mel_frames=2)
    smoother.push(torch.arange(16, dtype=torch.float32), is_final=False)
    with pytest.raises(ValueError, match="prefix"):
        smoother.push(torch.arange(4, dtype=torch.float32), is_final=False)


def test_smoother_final_push_commits_everything_and_clears_tail():
    smoother = StreamingPcmSmoother(hop_length=4, overlap_mel_frames=2)
    smoother.push(torch.arange(16, dtype=torch.float32), is_final=False)
    final = smoother.push(torch.arange(20, dtype=torch.float32), is_final=True)
    assert final.numel() == 12
    assert smoother.flush().numel() == 0


def test_stream_metrics_summary_reports_ttfa_and_rtf():
    metrics = StreamMetrics(started_at=100.0)
    metrics.record_chunk(b"\x00" * 44100, now=100.5)
    metrics.record_chunk(b"\x00" * 44100, now=101.0)
    summary = metrics.summary(now=102.0)
    assert summary["ttfa_ms"] == pytest.approx(500.0)
    assert summary["audio_duration_ms"] == pytest.approx(2000.0)
    assert summary["rtf"] == pytest.approx(1.0)
    assert summary["chunks"] == 2
    assert summary["cancelled"] is False
