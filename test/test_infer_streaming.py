import pytest
import torch
from omegaconf import OmegaConf

from indextts.gpt.model_vllm_v2 import AcousticTokenChunk
from indextts.infer_vllm_v2 import IndexTTS2, PreparedInference
from indextts.streaming import (
    DEFAULT_FIRST_CHUNK_DIFFUSION_STEPS,
    IncrementalMelCache,
    should_decode_prefix,
)


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

    engine.decode_steps_log = []
    engine.decode_mel_states = []

    def fake_decode(prepared_arg, text_tokens, token_ids, latent, diffusion_steps=25, mel_state=None):
        # 100 samples per token so committed audio grows with the prefix
        engine.decode_steps_log.append(diffusion_steps)
        engine.decode_mel_states.append(mel_state)
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


@pytest.mark.asyncio
async def test_stream_infer_uses_low_steps_only_for_first_decode_of_request():
    batch = [
        AcousticTokenChunk(tuple(range(20)), False),
        AcousticTokenChunk(tuple(range(30)), True),
    ]
    engine = make_engine(
        chunk_batches=[list(batch), list(batch)],
        sentences=[["a"], ["b"]],
    )

    async for _ in engine.stream_infer(
        spk_audio_prompt="speaker.wav",
        text="two sentences",
        stream_chunk_tokens=20,
        first_chunk_diffusion_steps=12,
    ):
        pass
    assert engine.decode_steps_log == [12, 25, 25, 25]


@pytest.mark.asyncio
async def test_stream_infer_defaults_to_reduced_first_chunk_steps():
    engine = make_engine(
        chunk_batches=[[
            AcousticTokenChunk(tuple(range(20)), False),
            AcousticTokenChunk(tuple(range(30)), True),
        ]],
        sentences=[["a"]],
    )

    async for _ in engine.stream_infer(
        spk_audio_prompt="speaker.wav",
        text="default steps",
        stream_chunk_tokens=20,
    ):
        pass
    assert engine.decode_steps_log == [DEFAULT_FIRST_CHUNK_DIFFUSION_STEPS, 25]


class FakeCfm:
    """Returns mel filled with a per-call constant so stitching is checkable."""

    def __init__(self):
        self.calls = []
        self.level = 0.0

    def inference(self, mu, x_lens, prompt, style, f0, steps, inference_cfg_rate):
        self.level += 1e-4
        self.calls.append({
            "mu_frames": mu.size(1),
            "prompt_frames": prompt.size(-1),
            "steps": steps,
        })
        return torch.full((1, 80, mu.size(1)), self.level)


def make_decode_engine(ref_frames=30):
    dim = 8
    engine = IndexTTS2.__new__(IndexTTS2)
    engine.device = "cpu"

    def fake_gpt(cond_latent, text, text_len, codes, code_len, emo, **kwargs):
        return torch.zeros(1, codes.shape[-1], dim)

    engine.gpt = fake_gpt

    class Quantizer:
        def vq2emb(self, codes):
            return torch.zeros(1, dim, codes.shape[-1])

    class Codec:
        quantizer = Quantizer()

    engine.semantic_codec = Codec()

    class S2Mel:
        models = {
            "gpt_layer": lambda latent: latent,
            "length_regulator": lambda s, ylens, n_quantizers, f0: (
                torch.zeros(1, int(ylens[0]), dim), None
            ),
            "cfm": FakeCfm(),
        }

    engine.s2mel = S2Mel()
    # 4 samples per mel frame, first mel channel as the waveform value
    engine.bigvgan = lambda mel: mel[:, :1, :].repeat_interleave(4, dim=-1)

    prepared = PreparedInference(
        spk_cond_emb=torch.zeros(1, 8, 4),
        emo_cond_emb=torch.zeros(1, 8, 4),
        emovec=torch.zeros(1, 4),
        prompt_condition=torch.zeros(1, ref_frames, dim),
        ref_mel=torch.zeros(1, 80, ref_frames),
        style=torch.zeros(1, 4),
        sentences=[["a"]],
        emotion_preprocess_seconds=0.0,
    )
    return engine, prepared


def test_decode_stream_prefix_reuses_cached_mel_as_cfm_prompt():
    engine, prepared = make_decode_engine()
    cfm = engine.s2mel.models["cfm"]
    text_tokens = torch.zeros(1, 5, dtype=torch.int32)
    state = IncrementalMelCache()

    # Round 1: 60 codes -> 103 mel frames, no cache: full decode
    wav1 = engine._decode_stream_prefix(
        prepared, text_tokens, tuple(range(60)), torch.zeros(1), mel_state=state
    )
    assert cfm.calls[0]["mu_frames"] == 30 + 103
    assert cfm.calls[0]["prompt_frames"] == 30  # speaker ref only
    assert state.frames == 103
    assert wav1.numel() == 103 * 4

    # Round 2: 100 codes -> 172 frames; keep 103-16=87, all kept frames fit
    # in the 128-frame context window, so prompt = ref + 87 cached frames
    # and only 172-87=85 frames are generated.
    wav2 = engine._decode_stream_prefix(
        prepared, text_tokens, tuple(range(100)), torch.zeros(1), mel_state=state
    )
    assert cfm.calls[1]["prompt_frames"] == 30 + 87
    assert cfm.calls[1]["mu_frames"] == 30 + 172
    assert state.frames == 172
    assert wav2.numel() == 172 * 4
    # kept region still carries round-1 mel, tail carries round-2 mel
    level1, level2 = 32767 * 1e-4, 32767 * 2e-4
    assert torch.allclose(wav2[: 87 * 4], torch.full((87 * 4,), level1))
    assert torch.allclose(wav2[-85 * 4:], torch.full((85 * 4,), level2))


def test_decode_stream_prefix_keeps_full_reference_in_every_round():
    # Ref trimming in incremental rounds was tried and reverted (energy drift):
    # the full speaker reference must stay in the CFM prompt every round.
    engine, prepared = make_decode_engine(ref_frames=600)
    cfm = engine.s2mel.models["cfm"]
    text_tokens = torch.zeros(1, 5, dtype=torch.int32)
    state = IncrementalMelCache()

    engine._decode_stream_prefix(
        prepared, text_tokens, tuple(range(60)), torch.zeros(1), mel_state=state
    )
    engine._decode_stream_prefix(
        prepared, text_tokens, tuple(range(100)), torch.zeros(1), mel_state=state
    )
    assert cfm.calls[0]["prompt_frames"] == 600
    assert cfm.calls[1]["prompt_frames"] == 600 + 87


def test_decode_stream_prefix_vocodes_only_new_frames_with_cache():
    engine, prepared = make_decode_engine()
    text_tokens = torch.zeros(1, 5, dtype=torch.int32)
    state = IncrementalMelCache()
    vocoded_frames = []
    inner_bigvgan = engine.bigvgan

    def counting_bigvgan(mel):
        vocoded_frames.append(mel.size(-1))
        return inner_bigvgan(mel)

    engine.bigvgan = counting_bigvgan

    engine._decode_stream_prefix(
        prepared, text_tokens, tuple(range(60)), torch.zeros(1), mel_state=state
    )
    engine._decode_stream_prefix(
        prepared, text_tokens, tuple(range(100)), torch.zeros(1), mel_state=state
    )
    assert vocoded_frames[0] == 103  # first round has no wav cache
    assert vocoded_frames[1] == 172 - (87 - 16)  # margin + regenerated tail only
    assert state.wav.numel() == 172 * 4


def test_decode_stream_prefix_without_state_regenerates_everything():
    engine, prepared = make_decode_engine()
    cfm = engine.s2mel.models["cfm"]
    text_tokens = torch.zeros(1, 5, dtype=torch.int32)

    engine._decode_stream_prefix(prepared, text_tokens, tuple(range(60)), torch.zeros(1))
    engine._decode_stream_prefix(prepared, text_tokens, tuple(range(100)), torch.zeros(1))
    assert cfm.calls[1]["prompt_frames"] == 30  # no cached context
    assert cfm.calls[1]["mu_frames"] == 30 + 172


@pytest.mark.asyncio
async def test_stream_infer_applies_diffusion_steps_to_all_later_rounds():
    batch = [
        AcousticTokenChunk(tuple(range(20)), False),
        AcousticTokenChunk(tuple(range(30)), True),
    ]
    engine = make_engine(
        chunk_batches=[list(batch), list(batch)],
        sentences=[["a"], ["b"]],
    )

    async for _ in engine.stream_infer(
        spk_audio_prompt="speaker.wav",
        text="two sentences",
        stream_chunk_tokens=20,
        first_chunk_diffusion_steps=5,
        diffusion_steps=5,
    ):
        pass
    assert engine.decode_steps_log == [5, 5, 5, 5]


@pytest.mark.asyncio
async def test_stream_infer_rejects_invalid_diffusion_steps():
    engine = make_engine(
        chunk_batches=[[AcousticTokenChunk(tuple(range(20)), True)]],
        sentences=[["a"]],
    )

    with pytest.raises(ValueError):
        async for _ in engine.stream_infer(
            spk_audio_prompt="speaker.wav",
            text="bad steps",
            diffusion_steps=26,
        ):
            pass


@pytest.mark.asyncio
async def test_stream_infer_uses_one_mel_cache_per_sentence():
    batch = [
        AcousticTokenChunk(tuple(range(20)), False),
        AcousticTokenChunk(tuple(range(30)), True),
    ]
    engine = make_engine(
        chunk_batches=[list(batch), list(batch)],
        sentences=[["a"], ["b"]],
    )

    async for _ in engine.stream_infer(
        spk_audio_prompt="speaker.wav",
        text="two sentences",
        stream_chunk_tokens=20,
    ):
        pass
    states = engine.decode_mel_states
    assert len(states) == 4
    assert all(state is not None for state in states)
    assert states[0] is states[1]  # same sentence shares one cache
    assert states[2] is states[3]
    assert states[0] is not states[2]  # a fresh cache per sentence


@pytest.mark.asyncio
async def test_stream_infer_can_disable_incremental_decode():
    engine = make_engine(
        chunk_batches=[[
            AcousticTokenChunk(tuple(range(20)), False),
            AcousticTokenChunk(tuple(range(30)), True),
        ]],
        sentences=[["a"]],
    )

    async for _ in engine.stream_infer(
        spk_audio_prompt="speaker.wav",
        text="legacy path",
        stream_chunk_tokens=20,
        incremental_decode=False,
    ):
        pass
    assert engine.decode_mel_states == [None, None]


@pytest.mark.asyncio
async def test_stream_infer_rejects_invalid_first_chunk_steps():
    engine = make_engine(
        chunk_batches=[[AcousticTokenChunk(tuple(range(20)), True)]],
        sentences=[["a"]],
    )

    with pytest.raises(ValueError):
        async for _ in engine.stream_infer(
            spk_audio_prompt="speaker.wav",
            text="bad steps",
            first_chunk_diffusion_steps=4,
        ):
            pass
