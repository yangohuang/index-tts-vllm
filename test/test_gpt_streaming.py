from dataclasses import dataclass

import pytest

from indextts.gpt.model_vllm_v2 import AcousticTokenChunk, UnifiedVoice


@dataclass
class FakeCompletion:
    token_ids: list[int]


@dataclass
class FakeOutput:
    outputs: list[FakeCompletion]


class FakeLlm:
    def __init__(self):
        self.aborted = []

    async def generate(self, *args, **kwargs):
        yield FakeOutput([FakeCompletion([10, 11])])
        yield FakeOutput([FakeCompletion([10, 11, 8193, 8193])])

    async def abort(self, request_id):
        self.aborted.append(request_id)


@pytest.mark.asyncio
async def test_stream_yields_cumulative_codes_and_marks_final():
    voice = UnifiedVoice.__new__(UnifiedVoice)
    voice.llm = FakeLlm()
    voice.stop_mel_token = 8193
    voice.start_mel_token = 8192
    voice.sampling_params = None

    chunks = [
        chunk async for chunk in voice._stream_generated_codes(
            tokens_prompt=object(), request_id="request-1"
        )
    ]

    assert chunks == [
        AcousticTokenChunk((10, 11), False),
        AcousticTokenChunk((10, 11), True),
    ]


@pytest.mark.asyncio
async def test_abort_forwards_request_id_to_vllm():
    voice = UnifiedVoice.__new__(UnifiedVoice)
    voice.llm = FakeLlm()
    await voice.abort("request-2")
    assert voice.llm.aborted == ["request-2"]
