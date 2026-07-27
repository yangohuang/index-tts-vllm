import os
from collections import OrderedDict

import torch

from indextts.infer_vllm_v2 import IndexTTS2


def make_engine():
    engine = IndexTTS2.__new__(IndexTTS2)
    engine._spk_cond_cache = OrderedDict()
    engine._emo_cond_cache = OrderedDict()
    engine._cond_cache_size = 2
    engine.compute_calls = []

    def fake_compute(path):
        engine.compute_calls.append(path)
        return torch.zeros(1)

    engine._compute_speaker_conditioning = fake_compute
    return engine


def write_audio(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"fake")
    return str(path)


def test_same_file_computes_once(tmp_path):
    engine = make_engine()
    path = write_audio(tmp_path, "a.wav")

    first = engine._get_speaker_conditioning(path)
    second = engine._get_speaker_conditioning(path)

    assert engine.compute_calls == [path]
    assert first is second


def test_modified_file_recomputes(tmp_path):
    engine = make_engine()
    path = write_audio(tmp_path, "a.wav")

    engine._get_speaker_conditioning(path)
    stat = os.stat(path)
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    engine._get_speaker_conditioning(path)

    assert engine.compute_calls == [path, path]


def test_lru_evicts_oldest_entry(tmp_path):
    engine = make_engine()
    paths = [write_audio(tmp_path, f"{i}.wav") for i in range(3)]

    engine._get_speaker_conditioning(paths[0])
    engine._get_speaker_conditioning(paths[1])
    engine._get_speaker_conditioning(paths[2])  # evicts paths[0]
    engine._get_speaker_conditioning(paths[1])  # still cached
    engine._get_speaker_conditioning(paths[0])  # recomputed

    assert engine.compute_calls == [paths[0], paths[1], paths[2], paths[0]]
    assert len(engine._spk_cond_cache) == 2
