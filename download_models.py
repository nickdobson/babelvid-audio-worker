"""Bake the separator checkpoint and SenseVoice into the image (no downloads at cold start)."""

import os

from audio_separator.separator import Separator
from huggingface_hub import snapshot_download

models = os.environ.get("MODELS_DIR", "/models")
sep = Separator(model_file_dir=os.path.join(models, "separator"), output_format="WAV")
sep.load_model(model_filename=os.environ.get("SEPARATOR_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"))
snapshot_download("FunAudioLLM/SenseVoiceSmall", local_dir=os.path.join(models, "sensevoice"))
print("models downloaded")
