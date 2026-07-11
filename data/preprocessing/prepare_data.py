"""Step 1: Gather & curate Khmer audio data.

Downloads/loads the DDD-Cambodia Khmer corpora from Hugging Face, resamples
to 16 kHz, strips leading/trailing silence, and writes train/val/test
manifests (CSV: path,duration_sec,transcript,speaker,split).

Usage:
    python data/preprocessing/prepare_data.py --config configs/tokenizer_config.yaml
    python data/preprocessing/prepare_data.py --dataset DDD-Cambodia/khm-asr-cultural --out data/khm_asr_cultural
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import webrtcvad
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_SR = 16000


def trim_silence(audio: np.ndarray, sr: int, aggressiveness: int = 2, frame_ms: int = 30) -> np.ndarray:
    """Trim leading/trailing silence using WebRTC VAD (voice activity detection).

    WebRTC VAD requires 16-bit PCM mono at 8/16/32/48 kHz and frames of
    10/20/30 ms, so this must run after resampling to TARGET_SR.
    """
    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"webrtcvad requires sr in {{8000,16000,32000,48000}}, got {sr}")

    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    n_frames = len(pcm16) // frame_len
    if n_frames == 0:
        return audio

    voiced_frames = []
    for i in range(n_frames):
        frame = pcm16[i * frame_len : (i + 1) * frame_len]
        is_speech = vad.is_speech(frame.tobytes(), sr)
        voiced_frames.append(is_speech)

    if not any(voiced_frames):
        return audio  # nothing detected as speech; keep as-is rather than dropping the clip

    first = voiced_frames.index(True)
    last = len(voiced_frames) - 1 - voiced_frames[::-1].index(True)
    start = first * frame_len
    end = min((last + 1) * frame_len, len(pcm16))
    return audio[start:end]


def resample_and_trim(audio: np.ndarray, orig_sr: int, do_vad: bool = True) -> np.ndarray:
    import librosa

    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=TARGET_SR)
    if do_vad:
        audio = trim_silence(audio, TARGET_SR)
    return audio


def process_hf_dataset(
    dataset_name: str, out_dir: Path, splits: dict, do_vad: bool = True, max_samples: int | None = None
) -> pd.DataFrame:
    import io

    from datasets import Audio, load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "wav16k"
    audio_dir.mkdir(exist_ok=True)

    # `max_samples` uses HF streaming mode so only the first N examples are
    # ever fetched, instead of `load_dataset`'s default behavior of
    # downloading the entire corpus (~495GB, 1065h) before anything can be
    # subset — needed for a cheap pilot run on a disk/time-limited machine
    # like Colab.
    if max_samples is not None:
        import itertools

        log.info("Streaming %s from Hugging Face, capped at %d examples...", dataset_name, max_samples)
        ds = load_dataset(dataset_name, split="train", streaming=True)
        # `decode=False` + manual soundfile decode instead of `datasets`'
        # default Audio decoder, which (as of datasets>=~4) shells out to
        # torchcodec and needs system ffmpeg/libavutil — not guaranteed
        # present outside e.g. Colab's preinstalled image. soundfile is
        # already a hard dependency of this repo (used for VAD/resampling
        # below), so this avoids that extra system requirement entirely.
        # Must cast before islice: islice's plain generator has no
        # cast_column, so this has to happen on the IterableDataset first.
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = itertools.islice(ds, max_samples)
    else:
        log.info("Loading %s from Hugging Face (this downloads the full corpus)...", dataset_name)
        ds = load_dataset(dataset_name, split="train")
        ds = ds.cast_column("audio", Audio(decode=False))

    rows = []
    for i, example in enumerate(tqdm(ds, desc=f"processing {dataset_name}", total=max_samples)):
        audio = example["audio"]
        y, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = resample_and_trim(y, sr, do_vad=do_vad)

        out_path = audio_dir / f"{dataset_name.split('/')[-1]}_{i:07d}.wav"
        sf.write(out_path, y, TARGET_SR)

        rows.append(
            {
                "path": str(out_path),
                "duration_sec": len(y) / TARGET_SR,
                "transcript": example.get("transcript") or example.get("transcription") or example.get("text") or "",
                "speaker": example.get("speaker_id", example.get("speaker", "unknown")),
            }
        )

    df = pd.DataFrame(rows)
    empty_frac = (df["transcript"] == "").mean() if len(df) else 0.0
    if empty_frac > 0.5:
        log.warning(
            "%.0f%% of rows have an empty transcript — this dataset's transcript field likely "
            "doesn't match any of 'transcript'/'transcription'/'text'; check its actual column "
            "names (e.g. via the HF dataset viewer) and adjust the .get(...) fallback chain above. "
            "Downstream steps (train_ctc.py) will silently skip every utterance otherwise.",
            empty_frac * 100,
        )
    df = assign_splits(df, splits)
    return df


def assign_splits(df: pd.DataFrame, splits: dict, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.permutation(n)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])

    split_col = np.empty(n, dtype=object)
    split_col[idx[:n_train]] = "train"
    split_col[idx[n_train : n_train + n_val]] = "val"
    split_col[idx[n_train + n_val :]] = "test"
    df = df.copy()
    df["split"] = split_col
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/tokenizer_config.yaml")
    parser.add_argument("--dataset", type=str, default="DDD-Cambodia/khmer-speech-dataset")
    parser.add_argument("--out", type=str, default="data/khmer_asr_cultural_v2")
    parser.add_argument("--no-vad", action="store_true", help="skip silence trimming")
    parser.add_argument("--manifest-name", type=str, default=None)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="stream only the first N examples instead of downloading the full corpus (cheap pilot run, e.g. on Colab)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = cfg["data"]["splits"]
    manifest_dir = Path(cfg["data"]["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)

    df = process_hf_dataset(args.dataset, Path(args.out), splits, do_vad=not args.no_vad, max_samples=args.max_samples)

    manifest_name = args.manifest_name or (args.dataset.split("/")[-1] + "_manifest.csv")
    manifest_path = manifest_dir / manifest_name
    df.to_csv(manifest_path, index=False)

    log.info("Wrote %d utterances (%.1f h) to %s", len(df), df["duration_sec"].sum() / 3600, manifest_path)
    log.info("Split sizes: %s", df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
