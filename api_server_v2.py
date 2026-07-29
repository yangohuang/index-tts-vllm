import os
import asyncio
import io
import traceback
import uuid
from fastapi import FastAPI, Request, Response, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import List, Optional, Union

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2
from indextts.streaming import (
    CHANNELS,
    DEFAULT_FIRST_CHUNK_DIFFUSION_STEPS,
    DEFAULT_STREAM_CHUNK_TOKENS,
    FULL_DIFFUSION_STEPS,
    SAMPLE_FORMAT,
    SAMPLE_RATE,
    STREAM_FORMAT_MEDIA_TYPES,
    StreamMetrics,
    stream_ogg_opus,
    validate_diffusion_steps,
    validate_first_chunk_diffusion_steps,
    validate_stream_chunk_tokens,
    validate_stream_format,
    wav_stream_header,
)

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    if tts is None:
        tts = IndexTTS2(
            model_dir=args.model_dir,
            is_fp16=args.is_fp16,
            gpu_memory_utilization=args.gpu_memory_utilization,
            qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
        )
    yield


class TtsStreamRequest(BaseModel):
    text: str
    spk_audio_path: str
    emo_control_method: int = 0
    emo_ref_path: str | None = None
    emo_weight: float = 1.0
    emo_vec: list[float] = Field(default_factory=lambda: [0.0] * 8)
    emo_text: str | None = None
    emo_random: bool = False
    max_text_tokens_per_sentence: int = 120
    stream_chunk_tokens: int = DEFAULT_STREAM_CHUNK_TOKENS
    first_chunk_diffusion_steps: int = DEFAULT_FIRST_CHUNK_DIFFUSION_STEPS
    diffusion_steps: int = FULL_DIFFUSION_STEPS
    incremental_decode: bool = True
    format: str = "pcm"
    request_id: str | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("text must not be empty")
        return value

    @field_validator("stream_chunk_tokens")
    @classmethod
    def validate_chunk_tokens(cls, value):
        return validate_stream_chunk_tokens(value)

    @field_validator("first_chunk_diffusion_steps")
    @classmethod
    def validate_first_chunk_steps(cls, value):
        return validate_first_chunk_diffusion_steps(value)

    @field_validator("diffusion_steps")
    @classmethod
    def validate_diffusion(cls, value):
        return validate_diffusion_steps(value)

    @field_validator("format")
    @classmethod
    def validate_format(cls, value):
        return validate_stream_format(value)


def stream_infer_kwargs(payload: TtsStreamRequest) -> dict:
    """Map the transport-level request onto IndexTTS2.stream_infer arguments."""
    emo_ref_path = payload.emo_ref_path
    emo_weight = payload.emo_weight
    vec = None
    if payload.emo_control_method == 0:
        emo_ref_path = None
        emo_weight = 1.0
    if payload.emo_control_method == 2:
        if len(payload.emo_vec) != 8:
            raise ValueError("emo_vec must contain exactly 8 values")
        if sum(payload.emo_vec) > 1.5:
            raise ValueError("情感向量之和不能超过1.5，请调整后重试。")
        vec = payload.emo_vec
    return dict(
        spk_audio_prompt=payload.spk_audio_path,
        text=payload.text,
        emo_audio_prompt=emo_ref_path,
        emo_alpha=emo_weight,
        emo_vector=vec,
        use_emo_text=(payload.emo_control_method == 3),
        emo_text=payload.emo_text,
        use_random=payload.emo_random,
        max_text_tokens_per_sentence=payload.max_text_tokens_per_sentence,
        stream_chunk_tokens=payload.stream_chunk_tokens,
        first_chunk_diffusion_steps=payload.first_chunk_diffusion_steps,
        diffusion_steps=payload.diffusion_steps,
        incremental_decode=payload.incremental_decode,
        request_id=payload.request_id,
    )


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        # logger.info(f"Emo control mode:{emo_control_method}, vec:{vec}")
        sr, wav = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                        output_path=None,
                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                        emo_vector=vec,
                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


def formatted_stream_body(kwargs: dict, stream_format: str):
    """Wrap the engine PCM stream in the requested container format."""

    async def pcm_body():
        async for chunk in tts.stream_infer(**kwargs):
            yield chunk.pcm

    if stream_format == "ogg_opus":
        return stream_ogg_opus(pcm_body())

    async def body():
        if stream_format == "wav":
            yield wav_stream_header()
        async for chunk in pcm_body():
            yield chunk

    return body()


@app.post("/tts_stream")
async def tts_stream_api(payload: TtsStreamRequest):
    try:
        kwargs = stream_infer_kwargs(payload)
    except ValueError as ex:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "error": str(ex)},
        )

    return StreamingResponse(
        formatted_stream_body(kwargs, payload.format),
        media_type=STREAM_FORMAT_MEDIA_TYPES[payload.format],
        headers={
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Channels": str(CHANNELS),
            "X-Audio-Sample-Format": SAMPLE_FORMAT,
            "X-Stream-Format": payload.format,
            "X-Stream-Chunk-Tokens": str(payload.stream_chunk_tokens),
            "X-First-Chunk-Diffusion-Steps": str(payload.first_chunk_diffusion_steps),
        },
    )


class OpenAISpeechRequest(BaseModel):
    """Subset of the OpenAI /v1/audio/speech schema mapped onto IndexTTS2.

    ``voice`` is the speaker reference audio path; ``model`` is accepted and
    ignored; ``response_format`` supports pcm / wav / opus (Ogg-Opus).
    """

    input: str
    voice: str
    model: str = "index-tts-2"
    response_format: str = "wav"
    speed: float = 1.0
    stream_format: str | None = None

    @field_validator("input")
    @classmethod
    def validate_input(cls, value):
        if not value.strip():
            raise ValueError("input must not be empty")
        return value

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, value):
        supported = {"pcm": "pcm", "wav": "wav", "opus": "ogg_opus"}
        if value not in supported:
            raise ValueError(
                f"response_format must be one of {', '.join(supported)}"
            )
        return value


@app.post("/v1/audio/speech")
async def openai_speech_api(payload: OpenAISpeechRequest):
    if not os.path.isfile(payload.voice):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"voice audio not found: {payload.voice}",
                               "type": "invalid_request_error"}},
        )
    stream_format = {"pcm": "pcm", "wav": "wav", "opus": "ogg_opus"}[payload.response_format]
    kwargs = dict(
        spk_audio_prompt=payload.voice,
        text=payload.input,
    )
    return StreamingResponse(
        formatted_stream_body(kwargs, stream_format),
        media_type=STREAM_FORMAT_MEDIA_TYPES[stream_format],
        headers={"X-Stream-Format": stream_format},
    )


@app.websocket("/ws/tts_stream")
async def ws_tts_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        message = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    request_id = str(message.get("request_id") or uuid.uuid4().hex)
    if message.get("type") != "synthesize":
        await websocket.send_json({
            "type": "error",
            "request_id": request_id,
            "error_type": "invalid_request",
            "message": "first message must have type 'synthesize'",
        })
        await websocket.close()
        return

    try:
        payload = TtsStreamRequest(**{k: v for k, v in message.items() if k != "type"})
        kwargs = stream_infer_kwargs(payload)
    except ValueError:
        await websocket.send_json({
            "type": "error",
            "request_id": request_id,
            "error_type": "invalid_request",
            "message": "invalid synthesis request",
        })
        await websocket.close()
        return

    metrics = StreamMetrics.start()

    async def produce():
        async for chunk in tts.stream_infer(**kwargs):
            await websocket.send_bytes(chunk.pcm)
            metrics.record_chunk(chunk.pcm)

    async def receive_control():
        while True:
            control = await websocket.receive_json()
            if control.get("type") == "cancel":
                return

    await websocket.send_json({
        "type": "start",
        "request_id": request_id,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_format": SAMPLE_FORMAT,
    })

    producer = asyncio.create_task(produce())
    controller = asyncio.create_task(receive_control())
    client_gone = False
    try:
        done, _ = await asyncio.wait(
            {producer, controller}, return_when=asyncio.FIRST_COMPLETED
        )
        if producer in done:
            producer.result()  # surface inference errors
        else:
            metrics.cancelled = True
            producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass
            if controller.done() and controller.exception() is not None:
                client_gone = True
        if not client_gone:
            await websocket.send_json({
                "type": "end",
                "request_id": request_id,
                "cancelled": metrics.cancelled,
                "metrics": metrics.summary(),
            })
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception(f"[{request_id}] websocket synthesis failed")
        try:
            await websocket.send_json({
                "type": "error",
                "request_id": request_id,
                "error_type": "inference_error",
                "message": "synthesis failed",
            })
        except Exception:
            pass
    finally:
        for task in (producer, controller):
            task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)