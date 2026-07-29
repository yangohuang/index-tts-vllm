import asyncio

import pytest
from fastapi.testclient import TestClient

import api_server_v2
from api_server_v2 import TtsStreamRequest, app
from indextts.streaming import PcmChunk


def test_stream_request_defaults_to_twenty_tokens():
    request = TtsStreamRequest(
        text="hello",
        spk_audio_path="speaker.wav",
    )
    assert request.stream_chunk_tokens == 20


def test_stream_request_rejects_empty_text():
    with pytest.raises(ValueError):
        TtsStreamRequest(text="", spk_audio_path="speaker.wav")


def test_stream_request_rejects_out_of_range_chunk_tokens():
    with pytest.raises(ValueError):
        TtsStreamRequest(
            text="hello", spk_audio_path="speaker.wav", stream_chunk_tokens=5
        )


def test_stream_request_defaults_first_chunk_diffusion_steps():
    request = TtsStreamRequest(text="hello", spk_audio_path="speaker.wav")
    assert request.first_chunk_diffusion_steps == 10
    kwargs = api_server_v2.stream_infer_kwargs(request)
    assert kwargs["first_chunk_diffusion_steps"] == 10


def test_stream_request_maps_incremental_decode_flag():
    request = TtsStreamRequest(text="hello", spk_audio_path="speaker.wav")
    assert request.incremental_decode is True
    assert api_server_v2.stream_infer_kwargs(request)["incremental_decode"] is True
    disabled = TtsStreamRequest(
        text="hello", spk_audio_path="speaker.wav", incremental_decode=False
    )
    assert api_server_v2.stream_infer_kwargs(disabled)["incremental_decode"] is False


def test_stream_request_rejects_out_of_range_first_chunk_diffusion_steps():
    with pytest.raises(ValueError):
        TtsStreamRequest(
            text="hello", spk_audio_path="speaker.wav", first_chunk_diffusion_steps=4
        )
    with pytest.raises(ValueError):
        TtsStreamRequest(
            text="hello", spk_audio_path="speaker.wav", first_chunk_diffusion_steps=26
        )


def test_stream_request_rejects_overweight_emotion_vector():
    request = TtsStreamRequest(
        text="hello",
        spk_audio_path="speaker.wav",
        emo_control_method=2,
        emo_vec=[0.5] * 8,
    )
    with pytest.raises(ValueError):
        api_server_v2.stream_infer_kwargs(request)


class FakeTts:
    def __init__(self, chunks, block_after_first=False):
        self.chunks = chunks
        self.block_after_first = block_after_first
        self.calls = []

    async def stream_infer(self, **kwargs):
        self.calls.append(kwargs)
        for index, chunk in enumerate(self.chunks):
            yield chunk
            if self.block_after_first and index == 0:
                await asyncio.sleep(30)


@pytest.fixture
def client(monkeypatch):
    fake = FakeTts([
        PcmChunk(pcm=b"\x01\x00", is_final=False),
        PcmChunk(pcm=b"\x02\x00", is_final=True),
    ])
    monkeypatch.setattr(api_server_v2, "tts", fake)
    with TestClient(app) as test_client:
        test_client.fake_tts = fake
        yield test_client


def test_http_stream_returns_pcm_headers_and_body(client):
    response = client.post("/tts_stream", json={
        "text": "hello",
        "spk_audio_path": "speaker.wav",
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/pcm")
    assert response.headers["x-audio-sample-rate"] == "22050"
    assert response.headers["x-audio-channels"] == "1"
    assert response.headers["x-audio-sample-format"] == "pcm_s16le"
    assert response.headers["x-first-chunk-diffusion-steps"] == "10"
    assert response.content == b"\x01\x00\x02\x00"


def test_http_stream_rejects_invalid_payload(client):
    response = client.post("/tts_stream", json={
        "text": "",
        "spk_audio_path": "speaker.wav",
    })
    assert response.status_code == 422


def test_websocket_stream_orders_start_audio_end(client):
    with client.websocket_connect("/ws/tts_stream") as ws:
        ws.send_json({
            "type": "synthesize",
            "request_id": "test-1",
            "text": "hello",
            "spk_audio_path": "speaker.wav",
        })
        assert ws.receive_json()["type"] == "start"
        assert ws.receive_bytes() == b"\x01\x00"
        assert ws.receive_bytes() == b"\x02\x00"
        end = ws.receive_json()
        assert end["type"] == "end"
        assert end["request_id"] == "test-1"
        assert end["cancelled"] is False
        assert "metrics" in end


def test_websocket_cancel_stops_stream(monkeypatch):
    fake = FakeTts(
        [
            PcmChunk(pcm=b"\x01\x00", is_final=False),
            PcmChunk(pcm=b"\x02\x00", is_final=True),
        ],
        block_after_first=True,
    )
    monkeypatch.setattr(api_server_v2, "tts", fake)
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws/tts_stream") as ws:
            ws.send_json({
                "type": "synthesize",
                "request_id": "test-2",
                "text": "hello",
                "spk_audio_path": "speaker.wav",
            })
            assert ws.receive_json()["type"] == "start"
            assert ws.receive_bytes() == b"\x01\x00"
            ws.send_json({"type": "cancel"})
            end = ws.receive_json()
            assert end["type"] == "end"
            assert end["request_id"] == "test-2"
            assert end["cancelled"] is True


def test_websocket_invalid_request_reports_error(client):
    with client.websocket_connect("/ws/tts_stream") as ws:
        ws.send_json({
            "type": "synthesize",
            "request_id": "test-3",
            "text": "",
            "spk_audio_path": "speaker.wav",
        })
        message = ws.receive_json()
        assert message["type"] == "error"
        assert message["error_type"] == "invalid_request"
