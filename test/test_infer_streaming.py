import pytest
import torch
from omegaconf import OmegaConf

from indextts.gpt.model_vllm_v2 import AcousticTokenChunk
from indextts.infer_vllm_v2 import IndexTTS2, PreparedInference
from indextts.streaming import should_decode_prefix


def test_decode_triggers_after_configured_new_tokens():
    assert not should_decode_prefix(usable_tokens=19, decoded_tokens=0, threshold=20)
    assert should_decode_prefix(usable_tokens=20, decoded_tokens=0, threshold=20)


def test_final_partial_prefix_triggers_but_empty_final_does_not():
    assert should_decode_prefix(usable_tokens=7, decoded_tokens=0, threshold=20, final=True)
    assert not should_decode_prefix(usable_tokens=7, decoded_tokens=7, threshold=20, final=True)


class FakeTokenizer:
    def convert_tokens_to_ids(self, tokens):
        return list(range(len(tokens)))


class FakeGpt:
    def __init__(self, chunk_batches):
        self.chunk_batches = chunk_batches
        self.calls = 0
        self.aborted = []

    async def inference_speech_stream(self, *args, request_id="", **kwargs):
        chunks = self.chunk_batches[self.calls]
        self.calls += 1

        async def token_stream():
            for chunk in chunks:
                yield chunk

        return token_stream(), torch.zeros(1)

    async def abort(self, request_id):
        self.aborted.append(request_id)


def make_engine(chunk_batches, sentences):
    engine = IndexTTS2.__new__(IndexTTS2)
    engine.device = "cpu"
    engine.cfg = OmegaConf.create(
        {"s2mel": {"preprocess_params": {"spect_params": {"hop_length": 4}}}}
    )
    engine.tokenizer = FakeTokenizer()
    engine.gpt = FakeGpt(chunk_batches)

    prepared = PreparedInference(
        spk_cond_emb=torch.zeros(1, 8, 4),
        emo_cond_emb=torch.zeros(1, 8, 4),
        emovec=torch.zeros(1, 4),
        prompt_condition=torch.zeros(1, 4, 4),
        ref_mel=torch.zeros(1, 4, 4),
        style=torch.zeros(1, 4),
        sentences=sentences,
        emotion_preprocess_seconds=0.0,
    )

    async def fake_prepare(*args, **kwargs):
        return prepared

    engine._prepare_inference = fake_prepare

    def fake_decode(prepared_arg, text_tokens, token_ids, latent):
        # 100 samples per token so committed audio grows with the prefix
        return torch.ones(len(token_ids) * 100, dtype=torch.float32)

    engine._decode_stream_prefix = fake_decode
    return engine


@pytest.mark.asyncio
async def test_stream_infer_emits_multiple_pcm_chunks_and_metrics():
    engine = make_engine(
        chunk_batches=[[
            AcousticTokenChunk(tuple(range(20)), False),
            AcousticTokenChunk(tuple(range(30)), True),
        ]],
        sentences=[["a", "b"]],
    )

    chunks = [
        chunk async for chunk in engine.stream_infer(
            spk_audio_prompt="speaker.wav",
            text="stream this",
            stream_chunk_tokens=20,
        )
    ]
    assert len(chunks) == 2
    assert all(chunk.pcm for chunk in chunks)
    assert chunks[-1].is_final
    assert engine.gpt.aborted  # cleanup always aborts the request


@pytest.mark.asyncio
async def test_stream_infer_skips_subthreshold_updates():
    engine = make_engine(
        chunk_batches=[[
            AcousticTokenChunk(tuple(range(5)), False),
            AcousticTokenChunk(tuple(range(20)), False),
            AcousticTokenChunk(tuple(range(25)), True),
        ]],
        sentences=[["a"]],
    )

    chunks = [
        chunk async for chunk in engine.stream_infer(
            spk_audio_prompt="speaker.wav",
            text="short",
            stream_chunk_tokens=20,
        )
    ]
    assert len(chunks) == 2
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_stream_infer_flushes_tail_when_stop_token_missing():
    engine = make_engine(
        chunk_batches=[[
            AcousticTokenChunk(tuple(range(20)), False),
            AcousticTokenChunk(tuple(range(28)), False),
        ]],
        sentences=[["a"]],
    )

    chunks = [
        chunk async for chunk in engine.stream_infer(
            spk_audio_prompt="speaker.wav",
            text="truncated",
            stream_chunk_tokens=20,
        )
    ]
    total_samples = sum(len(chunk.pcm) for chunk in chunks) // 2
    assert total_samples == 2800
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_stream_infer_inserts_silence_between_sentences():
    batch = [
        AcousticTokenChunk(tuple(range(20)), False),
        AcousticTokenChunk(tuple(range(30)), True),
    ]
    engine = make_engine(
        chunk_batches=[list(batch), list(batch)],
        sentences=[["a"], ["b"]],
    )

    chunks = [
        chunk async for chunk in engine.stream_infer(
            spk_audio_prompt="speaker.wav",
            text="two sentences",
            stream_chunk_tokens=20,
            interval_silence=200,
        )
    ]
    silence_bytes = b"\x00" * (2 * int(22050 * 0.2))
    assert any(chunk.pcm == silence_bytes for chunk in chunks)
    assert chunks[-1].is_final
    assert not any(chunk.is_final for chunk in chunks[:-1])
