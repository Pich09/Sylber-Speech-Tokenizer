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
    fixed 30s window, so a single forward pass silently drops anything past
    30s of audio. We chunk longer utterances into consecutive 30s windows
    and concatenate their encoder outputs so no audio is silently dropped;
    each chunk's output is trimmed back to the frame count implied by its
    real (possibly < 30s, for the last chunk) duration so token-rate stats
    aren't inflated by padding either.
    """

    name = "whisper"
    _FRAMES_PER_SEC = 50  # 2x conv stride over 100Hz mel frames
    _CHUNK_SEC = 30.0  # WhisperFeatureExtractor's fixed input window

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINTS["whisper"], device: str | None = None):
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
        self.model = WhisperModel.from_pretrained(checkpoint).to(self.device).eval()

    @torch.no_grad()
    def _encode_chunk(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        chunk_duration = len(chunk) / sr
        inputs = self.feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        hidden_states = self.model.encoder(input_features).last_hidden_state  # (1, 1500, D)
        feats = hidden_states.squeeze(0).cpu().numpy()
        valid_frames = min(int(np.ceil(chunk_duration * self._FRAMES_PER_SEC)), feats.shape[0])
        return feats[:valid_frames]

    def __call__(self, wav_path: str, in_second: bool = True) -> dict:
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype="float32")
        if len(audio) == 0:
            return {"segment_features": np.zeros((0, self.model.config.d_model), dtype=np.float32), "segments": []}

        chunk_samples = int(self._CHUNK_SEC * sr)
        chunks = [self._encode_chunk(audio[i : i + chunk_samples], sr) for i in range(0, len(audio), chunk_samples)]
        feats = np.concatenate(chunks, axis=0)
        return {"segment_features": feats, "segments": []}


def resolve_checkpoint(name: str, checkpoint: str | None) -> str:
    """The checkpoint/model-id `load_encoder` will actually use for `name`,
    without instantiating anything — used to record provenance (see
    discretization.py's `extract`) so later steps can detect a mismatch."""
    if checkpoint:
        return checkpoint
    if name == "sylber":
        return "sylber"
    if name in DEFAULT_CHECKPOINTS:
        return DEFAULT_CHECKPOINTS[name]
    raise ValueError(f"unknown encoder {name!r}; expected one of sylber, hubert, whisper")


def load_encoder(
    name: str,
    checkpoint: str | None = None,
    device: str | None = None,
    norm_threshold: float | None = None,
    merge_threshold: float | None = None,
):
    """Factory matching `segmentation.load_segmenter`'s role for the
    non-Sylber encoders. `name` selects the adapter; `checkpoint` overrides
    the default HF model id (ignored for "sylber", which uses its own
    checkpoint-name convention). `norm_threshold`/`merge_threshold` override
    Sylber's fixed segmentation heuristic (sylber-only; ignored otherwise —
    see `segmentation.load_segmenter` and `docs/post-benchmark-roadmap.md`'s
    "Free lever found" section for why the default merge_threshold=0.8
    under-segments Khmer)."""
    resolved = resolve_checkpoint(name, checkpoint)
    if name == "sylber":
        from segmentation import load_segmenter

        return load_segmenter(resolved, norm_threshold=norm_threshold, merge_threshold=merge_threshold)
    if name == "hubert":
        return HubertEncoder(resolved, device=device)
    if name == "whisper":
        return WhisperEncoder(resolved, device=device)
    raise ValueError(f"unknown encoder {name!r}; expected one of sylber, hubert, whisper")


def load_kmeans_provenance(kmeans_path: str) -> dict | None:
    """Read the sidecar `<kmeans_path>.meta.json` written by
    discretization.py's `fit`, if present, recording which encoder/checkpoint
    produced the embeddings this k-means model was fit on."""
    sidecar = Path(str(kmeans_path) + ".meta.json")
    if not sidecar.exists():
        return None
    import json

    return json.loads(sidecar.read_text())


class SpeechTokenizer:
    """Encoder-agnostic counterpart to tokenizer.py's KhmerSyllableTokenizer,
    used by train_slm.py's `encode` command when --encoder != sylber.

    Special token IDs (pad/bos/eos) are computed from the fitted k-means
    model's actual `n_clusters` (see special_tokens.py), not a fixed config
    constant, so they stay correct across encoders whose k-means K differs.
    """

    def __init__(
        self,
        encoder_name: str,
        kmeans_path: str,
        checkpoint: str | None = None,
        device: str | None = None,
    ):
        from special_tokens import special_token_ids

        # If this encoder's k-means was fit on a specific checkpoint
        # (discretization.py extract --checkpoint ...), default to that same
        # checkpoint here rather than silently falling back to
        # DEFAULT_CHECKPOINTS — a checkpoint mismatch between the embeddings
        # k-means was fit on and the encoder used at inference time makes
        # k-means.predict() meaningless (different feature space).
        if checkpoint is None:
            provenance = load_kmeans_provenance(kmeans_path)
            if provenance:
                if provenance.get("encoder") not in (None, encoder_name):
                    log.warning(
                        "kmeans_path=%s was fit on encoder=%r but --encoder=%r was requested; "
                        "token assignments will likely be meaningless.",
                        kmeans_path, provenance.get("encoder"), encoder_name,
                    )
                checkpoint = provenance.get("checkpoint")
                if checkpoint:
                    log.info("Using checkpoint=%r recorded alongside %s", checkpoint, kmeans_path)

        self.encoder_name = encoder_name
        self.encoder = load_encoder(encoder_name, checkpoint=checkpoint, device=device)
        self.kmeans = joblib.load(kmeans_path)
        self.vocab_size = self.kmeans.n_clusters
        self.special_tokens = special_token_ids(self.vocab_size)

    def encode(self, wav_path: str, add_bos_eos: bool = False) -> dict:
        out = self.encoder(wav_path, in_second=True)
        feats = np.asarray(out["segment_features"])
        segments = out.get("segments", [])

        token_ids = self.kmeans.predict(feats).tolist() if feats.shape[0] > 0 else []

        if add_bos_eos and "bos" in self.special_tokens and "eos" in self.special_tokens:
            token_ids = [self.special_tokens["bos"]] + token_ids + [self.special_tokens["eos"]]

        return {"token_ids": token_ids, "segments": list(segments)}
