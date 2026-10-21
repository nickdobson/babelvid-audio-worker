"""RunPod Serverless queue worker: dialog/background separation + SenseVoice delivery analysis.

Jobs:
    {"task": "separate", "audio_b64": "<base64 audio, any ffmpeg-readable format>"}
        -> {"vocals_b64": "<base64 Ogg/Opus>", "background_b64": "<base64 Ogg/Opus>", "format": "ogg"}

    {"task": "analyze", "clips": [{"id": "s1", "audio_b64": "<base64 audio of one dialog line>"}]}
        -> {"results": [{"id": "s1", "emotion": "ANGRY" | null, "events": [{"kind": "Laughter", "time": 1.2}]}]}

    {"warmup": true} -> {"warm": true}

Requests are capped at 10 MB by RunPod's /run, so the client sends long audio in chunks.
"""

from __future__ import annotations

import base64
import glob
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
SEPARATOR_MODEL = os.environ.get("SEPARATOR_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
SENSEVOICE_MODEL = os.environ.get("SENSEVOICE_MODEL", os.path.join(MODELS_DIR, "sensevoice"))
FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
MAX_INPUT_BYTES = 9 * 1024 * 1024
MAX_CLIPS = 200
MAX_CLIP_SECONDS = 30.0

SR = 16000
EMOTIONS = {"HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED"}
EVENTS = {"Laughter", "Cry", "Cough", "Sneeze", "Breath", "Applause"}
WINDOW, HOP = 1.2, 0.6
_TOKEN_RE = re.compile(r"<\|([^|]+)\|>")

SEPARATOR = None  # audio_separator.separator.Separator
SENSEVOICE = None  # funasr.AutoModel


def log(msg: str) -> None:
    print(f"[audio-worker] {msg}", flush=True)


# ---------------------------------------------------------------------------- helpers
def ffmpeg(*args: str, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", *args],
                          input=input_bytes, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[-500:]}")
    return proc.stdout


def decode_b64(data: str, what: str) -> bytes:
    if not isinstance(data, str) or not data:
        raise ValueError(f"{what} must be a non-empty base64 string")
    raw = base64.b64decode(data, validate=True)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError(f"{what} is larger than {MAX_INPUT_BYTES // (1024 * 1024)} MB - send shorter chunks")
    return raw


def to_mono16k(raw: bytes) -> np.ndarray:
    pcm = ffmpeg("-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1", input_bytes=raw)
    return np.frombuffer(pcm, dtype=np.float32).copy()


# ---------------------------------------------------------------------------- separation
def separate(raw: bytes) -> dict:
    work = tempfile.mkdtemp(prefix="sep-")
    try:
        src = os.path.join(work, "input.wav")
        with open(os.path.join(work, "input.bin"), "wb") as f:
            f.write(raw)
        ffmpeg("-y", "-i", os.path.join(work, "input.bin"), "-ac", "2", "-ar", "44100", src)
        SEPARATOR.output_dir = work
        if getattr(SEPARATOR, "model_instance", None) is not None:
            SEPARATOR.model_instance.output_dir = work
        outputs = SEPARATOR.separate(src, {"Vocals": "vocals", "Instrumental": "instrumental"})
        files = [o if os.path.isabs(o) else os.path.join(work, o) for o in outputs]
        files = [f for f in files if os.path.exists(f)] or glob.glob(os.path.join(work, "*.wav"))
        vocals = next((f for f in files if "vocal" in os.path.basename(f).lower()), None)
        background = next((f for f in files if "instrumental" in os.path.basename(f).lower()), None)
        if not vocals or not background:
            raise RuntimeError(f"separator produced unexpected files: {[os.path.basename(f) for f in files]}")

        def opus(path: str, bitrate: str) -> str:
            data = ffmpeg("-i", path, "-c:a", "libopus", "-b:a", bitrate, "-ar", "48000", "-f", "ogg", "pipe:1")
            return base64.b64encode(data).decode("ascii")

        return {"vocals_b64": opus(vocals, "128k"), "background_b64": opus(background, "160k"), "format": "ogg"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------- SenseVoice
def parse_labels(raw: str) -> tuple[str | None, str | None]:
    emotion = event = None
    for tok in _TOKEN_RE.findall(raw):
        if tok in EMOTIONS:
            emotion = tok
        elif tok in EVENTS:
            event = tok
    return emotion, event


def infer(clips: list[np.ndarray]) -> list[str]:
    try:
        res = SENSEVOICE.generate(input=clips, cache={}, language="auto", use_itn=False, batch_size=len(clips))
        if len(res) == len(clips):
            return [r.get("text", "") for r in res]
    except Exception as exc:  # noqa: BLE001 - some funasr versions don't batch arrays
        log(f"batched SenseVoice call failed ({exc}); falling back to one clip at a time")
    out = []
    for clip in clips:
        res = SENSEVOICE.generate(input=clip, cache={}, language="auto", use_itn=False)
        out.append(res[0].get("text", "") if res else "")
    return out


def analyze_clip(clip: np.ndarray) -> tuple[str | None, list[dict]]:
    """Emotion of the whole line, plus events located with short overlapping windows."""
    if len(clip) < int(0.3 * SR):
        return None, []
    n, step = int(WINDOW * SR), int(HOP * SR)
    wins = [(i / SR, clip[i:i + n]) for i in range(0, len(clip) - n + 1, step)] if len(clip) >= int(1.8 * SR) else []
    labels = [parse_labels(t) for t in infer([clip] + [w for _, w in wins])]
    emotion, whole_event = labels[0]
    events: list[dict] = []
    for (start, _), (_, ev) in zip(wins, labels[1:]):
        if ev and (not events or events[-1]["kind"] != ev or start - events[-1]["time"] > WINDOW):
            events.append({"kind": ev, "time": round(start + WINDOW / 2, 2)})
    if whole_event and not any(e["kind"] == whole_event for e in events):
        events.append({"kind": whole_event, "time": round(len(clip) / SR / 2, 2)})
    return emotion, events


def analyze(clips: list) -> dict:
    if not isinstance(clips, list) or not 1 <= len(clips) <= MAX_CLIPS:
        raise ValueError(f"clips must be a list of 1-{MAX_CLIPS} items")
    results = []
    for c in clips:
        if not isinstance(c, dict) or "id" not in c:
            raise ValueError("each clip needs an id and audio_b64")
        audio = to_mono16k(decode_b64(c.get("audio_b64", ""), "clip audio_b64"))
        if len(audio) > MAX_CLIP_SECONDS * SR:
            raise ValueError(f"clip {c['id']} is longer than {MAX_CLIP_SECONDS:.0f} s")
        emotion, events = analyze_clip(audio)
        results.append({"id": str(c["id"]), "emotion": emotion, "events": events})
    return {"results": results}


# ---------------------------------------------------------------------------- worker
def init() -> None:
    global SEPARATOR, SENSEVOICE
    from audio_separator.separator import Separator
    from funasr import AutoModel

    log(f"loading separator {SEPARATOR_MODEL}")
    SEPARATOR = Separator(output_dir=tempfile.gettempdir(), model_file_dir=os.path.join(MODELS_DIR, "separator"),
                          output_format="WAV", sample_rate=44100)
    SEPARATOR.load_model(model_filename=SEPARATOR_MODEL)
    log(f"loading SenseVoice from {SENSEVOICE_MODEL}")
    SENSEVOICE = AutoModel(model=SENSEVOICE_MODEL, device="cuda:0", disable_update=True)
    log("ready")


def handler(job: dict) -> dict:
    job_input = job.get("input") or {}
    if job_input.get("warmup"):
        return {"warm": True}
    task = job_input.get("task")
    if task == "separate":
        return separate(decode_b64(job_input.get("audio_b64", ""), "audio_b64"))
    if task == "analyze":
        return analyze(job_input.get("clips"))
    raise ValueError("input.task must be 'separate' or 'analyze'")


if __name__ == "__main__":
    import runpod

    init()
    runpod.serverless.start({"handler": handler})
