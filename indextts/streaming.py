"""Transport-independent streaming primitives for IndexTTS2 PCM streaming."""

from dataclasses import dataclass
import time

import numpy as np
import torch

SAMPLE_RATE = 22050
CHANNELS = 1
SAMPLE_FORMAT = "pcm_s16le"
PCM16_MAX = 32767.0
PCM16_SAMPLE_WIDTH = 2
DEFAULT_STREAM_CHUNK_TOKENS = 20
MIN_STREAM_CHUNK_TOKENS = 10
MAX_STREAM_CHUNK_TOKENS = 100
FULL_DIFFUSION_STEPS = 25
DEFAULT_FIRST_CHUNK_DIFFUSION_STEPS = 15
MIN_FIRST_CHUNK_DIFFUSION_STEPS = 5
# Incremental prefix decoding: cached mel frames fed back to the CFM as clean
# prompt context, so each round only generates redo + new frames.
DEFAULT_CFM_CONTEXT_FRAMES = 128  # ~1.5 s of mel context at hop 256 / 22050 Hz
DEFAULT_CFM_REDO_FRAMES = 16  # regenerated tail; must cover the smoother holdback (8)
# NOTE: trimming the speaker reference prompt in incremental rounds (keeping
# only its tail once own-speech context exists) was tried and reverted: the
# weakened anchor lets energy drift upward round over round (see
# docs/indextts2-streaming-results.md, iteration 4).
DEFAULT_VOCODER_MARGIN_FRAMES = 16  # left context re-vocoded and discarded per round


def validate_stream_chunk_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("stream_chunk_tokens must be an integer")
    if not MIN_STREAM_CHUNK_TOKENS <= value <= MAX_STREAM_CHUNK_TOKENS:
        raise ValueError(
            f"stream_chunk_tokens must be between "
            f"{MIN_STREAM_CHUNK_TOKENS} and {MAX_STREAM_CHUNK_TOKENS}"
        )
    return value


def validate_first_chunk_diffusion_steps(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("first_chunk_diffusion_steps must be an integer")
    if not MIN_FIRST_CHUNK_DIFFUSION_STEPS <= value <= FULL_DIFFUSION_STEPS:
        raise ValueError(
            f"first_chunk_diffusion_steps must be between "
            f"{MIN_FIRST_CHUNK_DIFFUSION_STEPS} and {FULL_DIFFUSION_STEPS}"
        )
    return value


def should_decode_prefix(
    usable_tokens: int,
    decoded_tokens: int,
    threshold: int,
    final: bool = False,
) -> bool:
    pending = usable_tokens - decoded_tokens
    return pending > 0 and (final or pending >= threshold)


def pcm16le_bytes(waveform: torch.Tensor) -> bytes:
    values = (
        waveform.detach()
        .to(device="cpu", dtype=torch.float32)
        .flatten()
        .clamp(-PCM16_MAX, PCM16_MAX)
        .to(dtype=torch.int16)
        .numpy()
        .astype("<i2", copy=False)
    )
    return values.tobytes()


@dataclass(frozen=True)
class PcmChunk:
    pcm: bytes
    is_final: bool = False


@dataclass(frozen=True)
class MelWindowPlan:
    """Frame bookkeeping for one incremental decode round.

    ``keep_frames`` of cached mel survive unchanged; frames
    [``window_start``, ``keep_frames``) are fed to the CFM as clean prompt
    context; frames [``keep_frames``, ``total_frames``) are generated.
    """

    keep_frames: int
    window_start: int
    total_frames: int

    @property
    def context_frames(self) -> int:
        return self.keep_frames - self.window_start

    @property
    def generated_frames(self) -> int:
        return self.total_frames - self.keep_frames


def plan_mel_window(
    cached_frames: int,
    total_frames: int,
    context_frames: int = DEFAULT_CFM_CONTEXT_FRAMES,
    redo_frames: int = DEFAULT_CFM_REDO_FRAMES,
) -> MelWindowPlan:
    if cached_frames < 0 or total_frames < 0:
        raise ValueError("frame counts must be non-negative")
    if context_frames < 0 or redo_frames < 0:
        raise ValueError("context_frames and redo_frames must be non-negative")
    keep = max(0, min(cached_frames - redo_frames, total_frames))
    window_start = max(0, keep - context_frames)
    return MelWindowPlan(
        keep_frames=keep, window_start=window_start, total_frames=total_frames
    )


class IncrementalMelCache:
    """Per-sentence mel + waveform cache for incremental prefix decoding."""

    def __init__(self):
        self.mel = None  # [B, n_mels, frames] or None before the first decode
        self.wav = None  # float waveform aligned with ``mel`` (frames * hop samples)
        self.samples_per_frame = None  # inferred from the first vocoder call

    @property
    def frames(self) -> int:
        return 0 if self.mel is None else self.mel.size(-1)


@dataclass
class StreamMetrics:
    started_at: float
    first_audio_at: float | None = None
    emotion_preprocess_seconds: float = 0.0
    emitted_bytes: int = 0
    chunks: int = 0
    cancelled: bool = False

    @classmethod
    def start(cls) -> "StreamMetrics":
        return cls(started_at=time.perf_counter())

    def record_chunk(self, chunk: bytes, now: float | None = None) -> None:
        timestamp = time.perf_counter() if now is None else now
        if self.first_audio_at is None:
            self.first_audio_at = timestamp
        self.emitted_bytes += len(chunk)
        self.chunks += 1

    def summary(self, now: float | None = None) -> dict[str, float | int | bool]:
        finished_at = time.perf_counter() if now is None else now
        elapsed = finished_at - self.started_at
        duration = self.emitted_bytes / (SAMPLE_RATE * CHANNELS * PCM16_SAMPLE_WIDTH)
        return {
            "ttfa_ms": (
                (self.first_audio_at - self.started_at) * 1000
                if self.first_audio_at is not None else 0.0
            ),
            "emotion_preprocess_ms": self.emotion_preprocess_seconds * 1000,
            "elapsed_ms": elapsed * 1000,
            "audio_duration_ms": duration * 1000,
            "rtf": elapsed / duration if duration else 0.0,
            "chunks": self.chunks,
            "cancelled": self.cancelled,
        }


class StreamingPcmSmoother:
    """Commits stable PCM from successive decoded waveform prefixes.

    Each ``push`` receives the full waveform decoded from the current token
    prefix. The trailing ``overlap_samples`` region is held back (it will be
    re-decoded with more right context next time) and crossfaded against the
    next prefix with an equal-power curve.
    """

    def __init__(self, hop_length: int, overlap_mel_frames: int = 8):
        if hop_length <= 0 or overlap_mel_frames <= 0:
            raise ValueError("hop_length and overlap_mel_frames must be positive")
        self.overlap_samples = hop_length * overlap_mel_frames
        self.committed_samples = 0
        self._tail = torch.empty(0, dtype=torch.float32)

    def push(self, prefix_waveform: torch.Tensor, is_final: bool) -> torch.Tensor:
        waveform = prefix_waveform.detach().float().flatten()
        if waveform.numel() < self.committed_samples:
            raise ValueError("decoded prefix is shorter than committed audio")

        stable_end = waveform.numel() if is_final else max(
            self.committed_samples,
            waveform.numel() - self.overlap_samples,
        )
        new_audio = waveform[self.committed_samples:stable_end].clone()

        overlap = min(self._tail.numel(), new_audio.numel())
        if overlap:
            phase = torch.linspace(
                0.0, torch.pi / 2, overlap, dtype=new_audio.dtype
            )
            new_audio[:overlap] = (
                self._tail[-overlap:] * torch.cos(phase)
                + new_audio[:overlap] * torch.sin(phase)
            )

        self.committed_samples = stable_end
        self._tail = waveform[stable_end:].clone()
        if is_final:
            self._tail = torch.empty(0, dtype=torch.float32)
        return new_audio

    def flush(self) -> torch.Tensor:
        tail = self._tail
        self.committed_samples += tail.numel()
        self._tail = torch.empty(0, dtype=torch.float32)
        return tail
