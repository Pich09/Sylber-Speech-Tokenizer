"""Encoder-agnostic adapters for the SLM encoder benchmark (Step 5 "Baseline
Comparison" in docs/audio-tokenizer-comparison.md).

Sylber pools features per detected syllable (~4-5 Hz); HuBERT and Whisper
have no notion of syllable boundaries, so their adapters return raw
frame-level features at the model's native stride instead. Every adapter
exposes the same call signature as `segmentation.load_segmenter`'s
`Segmenter` — `__call__(wav_path, in_second=True) -> dict` with a
`"segment_features"` key (and an optional `"segments"` key, empty for the
frame-level encoders) — so `discretization.py` and the tokenizer path in
`train_slm.py` can stay encoder-agnostic.

Usage:
    from encoders import load_encoder
    enc = load_encoder("hubert")
    out = enc("path/to.wav", in_second=True)
    feats = out["segment_features"]  # (T, D)
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = {
    "hubert": "facebook/hubert-base-ls960",
    "whisper": "openai/whisper-base",
}


class HubertEncoder:
    """Frame-level HuBERT features (~50 Hz, 20ms stride) — no discretization
    or syllable pooling; that's left to k-means in discretization.py, same
    as the Sylber path, so the two are compared on an equal footing."""

    name = "hubert"

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINTS["hubert"], device: str | None = None):
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
        self.model = HubertModel.from_pretrained(checkpoint).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, wav_path: str, in_second: bool = True) -> dict:
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype="float32")
        inputs = self.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden_states = self.model(**inputs).last_hidden_state  # (1, T, D)
        feats = hidden_states.squeeze(0).cpu().numpy()
        return {"segment_features": feats, "segments": []}


class WhisperEncoder:
    """Frame-level Whisper encoder features (~50 Hz).

    Whisper's feature extractor always pads/truncates log-mel input to a
    fixed 30s window, so the raw encoder output is a fixed length (1500
    frames) regardless of the utterance's real duration. We trim back to
    the frame count implied by the true audio duration so token-rate stats
    (and everything downstream that assumes rate ~= n_tokens/duration)
    aren't silently inflated by padding.
    """

    name = "whisper"
    _FRAMES_PER_SEC = 50  # 2x conv stride over 100Hz mel frames

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINTS["whisper"], device: str | None = None):
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
        self.model = WhisperModel.from_pretrained(checkpoint).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, wav_path: str, in_second: bool = True) -> dict:
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype="float32")
        duration = len(audio) / sr
        inputs = self.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        hidden_states = self.model.encoder(input_features).last_hidden_state  # (1, 1500, D)
        feats = hidden_states.squeeze(0).cpu().numpy()

        valid_frames = min(int(np.ceil(duration * self._FRAMES_PER_SEC)), feats.shape[0])
        feats = feats[:valid_frames]
        return {"segment_features": feats, "segments": []}


def load_encoder(name: str, checkpoint: str | None = None, device: str | None = None):
    """Factory matching `segmentation.load_segmenter`'s role for the
    non-Sylber encoders. `name` selects the adapter; `checkpoint` overrides
    the default HF model id (ignored for "sylber", which uses its own
    checkpoint-name convention)."""
    if name == "sylber":
        from segmentation import load_segmenter

        return load_segmenter(checkpoint or "sylber")
    if name == "hubert":
        return HubertEncoder(checkpoint or DEFAULT_CHECKPOINTS["hubert"], device=device)
    if name == "whisper":
        return WhisperEncoder(checkpoint or DEFAULT_CHECKPOINTS["whisper"], device=device)
    raise ValueError(f"unknown encoder {name!r}; expected one of sylber, hubert, whisper")


class SpeechTokenizer:
    """Encoder-agnostic counterpart to tokenizer.py's KhmerSyllableTokenizer,
    used by train_slm.py's `encode` command when --encoder != sylber."""

    def __init__(
        self,
        encoder_name: str,
        kmeans_path: str,
        checkpoint: str | None = None,
        special_tokens: dict[str, int] | None = None,
        device: str | None = None,
    ):
        self.encoder_name = encoder_name
        self.encoder = load_encoder(encoder_name, checkpoint=checkpoint, device=device)
        self.kmeans = joblib.load(kmeans_path)
        self.vocab_size = self.kmeans.n_clusters
        self.special_tokens = special_tokens or {}

    def encode(self, wav_path: str, add_bos_eos: bool = False) -> dict:
        out = self.encoder(wav_path, in_second=True)
        feats = np.asarray(out["segment_features"])
        segments = out.get("segments", [])

        token_ids = self.kmeans.predict(feats).tolist() if feats.shape[0] > 0 else []

        if add_bos_eos and "bos" in self.special_tokens and "eos" in self.special_tokens:
            token_ids = [self.special_tokens["bos"]] + token_ids + [self.special_tokens["eos"]]

        return {"token_ids": token_ids, "segments": list(segments)}
