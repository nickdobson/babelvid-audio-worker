"""Handler tests with fake models (no GPU): real ffmpeg decoding/encoding, windowing and validation.

    pip install numpy soundfile imageio-ffmpeg pytest && pytest workers/audio
"""

import base64
import importlib.util
import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

try:
    import imageio_ffmpeg

    os.environ.setdefault("FFMPEG_BIN", imageio_ffmpeg.get_ffmpeg_exe())
except ImportError:
    pass

# Load by path under a unique name: every worker has its own handler.py.
_spec = importlib.util.spec_from_file_location("audio_handler", Path(__file__).parent / "handler.py")
handler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(handler)


def wav_b64(seconds: float, sr: int = 44100, channels: int = 2, freq: float = 220.0) -> str:
    t = np.arange(int(seconds * sr)) / sr
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


class FakeSeparator:
    output_dir = ""
    model_instance = None

    def separate(self, path, names):
        audio, sr = sf.read(path)
        sf.write(os.path.join(self.output_dir, names["Vocals"] + ".wav"), audio * 0.5, sr)
        sf.write(os.path.join(self.output_dir, names["Instrumental"] + ".wav"), audio * 0.25, sr)
        return [names["Vocals"] + ".wav", names["Instrumental"] + ".wav"]


class FakeSenseVoice:
    """Laughter in windows that start after 1 s; the whole clip sounds angry."""

    def generate(self, input, **kw):
        clips = input if isinstance(input, list) else [input]
        out = []
        for i, clip in enumerate(clips):
            if i == 0:
                out.append({"text": "<|en|><|ANGRY|><|Speech|><|woitn|>text"})
            else:
                out.append({"text": "<|en|><|NEUTRAL|><|Laughter|><|woitn|>ha"})
        return out


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(handler, "SEPARATOR", FakeSeparator())
    monkeypatch.setattr(handler, "SENSEVOICE", FakeSenseVoice())


def decode_ogg(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    pcm = handler.ffmpeg("-i", "pipe:0", "-f", "f32le", "-ac", "2", "-ar", "48000", "pipe:1", input_bytes=raw)
    return np.frombuffer(pcm, dtype=np.float32)


def test_separate_returns_two_opus_stems():
    out = handler.handler({"input": {"task": "separate", "audio_b64": wav_b64(3.0)}})
    assert out["format"] == "ogg"
    vocals, background = decode_ogg(out["vocals_b64"]), decode_ogg(out["background_b64"])
    assert abs(len(vocals) / 2 / 48000 - 3.0) < 0.1
    assert np.abs(vocals).max() > np.abs(background).max()  # fake separator halves vs quarters


def test_analyze_locates_events_and_emotion():
    out = handler.handler({"input": {"task": "analyze", "clips": [
        {"id": "a", "audio_b64": wav_b64(3.0, sr=16000, channels=1)},
        {"id": "b", "audio_b64": wav_b64(0.2, sr=16000, channels=1)},  # too short to analyse
    ]}})
    a, b = out["results"]
    assert a["id"] == "a" and a["emotion"] == "ANGRY"
    assert a["events"] and a["events"][0]["kind"] == "Laughter"
    assert b == {"id": "b", "emotion": None, "events": []}


def test_validation_and_warmup():
    assert handler.handler({"input": {"warmup": True}}) == {"warm": True}
    with pytest.raises(ValueError):
        handler.handler({"input": {"task": "separate"}})
    with pytest.raises(ValueError):
        handler.handler({"input": {"task": "nope"}})
    with pytest.raises(ValueError):
        handler.handler({"input": {"task": "analyze", "clips": []}})
