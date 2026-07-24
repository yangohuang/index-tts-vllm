from os import PathLike

import torch
import torchaudio

PCM16_MAX = 32767.0


def save_pcm16_waveform(
    output_path: str | PathLike[str],
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    normalized = (
        waveform.detach()
        .to(dtype=torch.float32)
        .clamp(-PCM16_MAX, PCM16_MAX)
        .div(PCM16_MAX)
    )
    torchaudio.save(str(output_path), normalized, sample_rate)
