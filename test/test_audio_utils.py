import torch
from torchcodec.decoders import AudioDecoder

from indextts.audio_utils import save_pcm16_waveform


def test_save_pcm16_waveform_preserves_normalized_amplitude(tmp_path):
    output_path = tmp_path / "output.wav"
    waveform = torch.tensor([[-16384.0, 0.0, 16384.0]])

    save_pcm16_waveform(output_path, waveform, 22050)

    decoded = AudioDecoder(str(output_path)).get_all_samples()
    assert decoded.sample_rate == 22050
    assert decoded.data.shape == (1, 3)
    torch.testing.assert_close(
        decoded.data,
        waveform / 32767.0,
        atol=2 / 32767.0,
        rtol=0,
    )
