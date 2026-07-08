"""Steps 2/3: Sylber segmentation — zero-shot baseline eval and fine-tuning
on Khmer (and optionally Khmer+English code-switched, Step 3b).

Zero-shot eval (Step 2):
    python src/segmentation.py eval --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
        --n-samples 100 --out results/zero_shot_evaluation.txt

Fine-tuning (Step 3):
    python src/segmentation.py finetune --manifest ... --mode head_only \
        --init-ckpt sylber --out-ckpt checkpoints/sylber_khmer_v1.pth
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_segmenter(checkpoint: str = "sylber"):
    """Load a pretrained Sylber segmenter.

    Requires the `sylber` package (Berkeley-Speech-Group/sylber). Import is
    lazy so the rest of this module (config/manifest handling) can be tested
    without the heavy model dependency installed.
    """
    from sylber import Segmenter

    return Segmenter(model_ckpt=checkpoint)


@dataclass
class SegmentationResult:
    path: str
    duration_sec: float
    segments: list[tuple[float, float]]

    @property
    def n_syllables(self) -> int:
        return len(self.segments)

    @property
    def token_rate_hz(self) -> float:
        return self.n_syllables / self.duration_sec if self.duration_sec > 0 else 0.0


def run_segmentation(segmenter, wav_paths: list[str]) -> list[SegmentationResult]:
    import soundfile as sf

    results = []
    for wav_path in tqdm(wav_paths, desc="segmenting"):
        info = sf.info(wav_path)
        duration = info.frames / info.samplerate
        out = segmenter(wav_path, in_second=True)
        # sylber returns {"segments": [(start, end), ...], "boundaries": [...]}
        # depending on version; normalize to a list of (start, end) tuples.
        segments = out["segments"] if isinstance(out, dict) else out
        results.append(SegmentationResult(path=wav_path, duration_sec=duration, segments=list(segments)))
    return results


def summarize(results: list[SegmentationResult]) -> dict:
    rates = [r.token_rate_hz for r in results if r.duration_sec > 0]
    total_syllables = sum(r.n_syllables for r in results)
    total_duration = sum(r.duration_sec for r in results)
    return {
        "n_utterances": len(results),
        "total_duration_sec": total_duration,
        "total_syllables": total_syllables,
        "mean_token_rate_hz": float(np.mean(rates)) if rates else 0.0,
        "median_token_rate_hz": float(np.median(rates)) if rates else 0.0,
        "std_token_rate_hz": float(np.std(rates)) if rates else 0.0,
    }


def cmd_eval(args):
    df = pd.read_csv(args.manifest)
    sample = df.sample(n=min(args.n_samples, len(df)), random_state=42)

    segmenter = load_segmenter(args.checkpoint)
    results = run_segmentation(segmenter, sample["path"].tolist())
    stats = summarize(results)

    log.info("Zero-shot segmentation stats: %s", json.dumps(stats, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Sylber Zero-Shot Segmentation — Khmer\n")
        f.write("=" * 50 + "\n\n")
        f.write(json.dumps(stats, indent=2))
        f.write("\n\n")
        f.write("Sanity check: expected token rate is 4-5 Hz.\n")
        if 4.0 <= stats["mean_token_rate_hz"] <= 5.0:
            f.write("PASS: mean token rate within expected 4-5 Hz range.\n")
        else:
            f.write("WARNING: mean token rate outside 4-5 Hz range — inspect segmentation quality.\n")
        f.write("\nPer-utterance detail:\n")
        for r in results:
            f.write(f"{r.path}\t dur={r.duration_sec:.2f}s\t syllables={r.n_syllables}\t rate={r.token_rate_hz:.2f}Hz\n")

    log.info("Wrote results to %s", out_path)
    log.info(
        "Decision point: if manual inspection of boundaries is >=80%% accurate, "
        "skip Step 3 and go straight to discretization (Step 4). Otherwise fine-tune."
    )


# ---------------------------------------------------------------------------
# Step 3: Fine-tuning
# ---------------------------------------------------------------------------
# Sylber's released checkpoint wraps a HuBERT-style CNN+Transformer backbone
# plus a segmentation head that scores frame-to-frame boundary probability.
# We fine-tune that boundary head (and optionally the backbone) against
# pseudo-labels: zero-shot segmentations, manually corrected on a small
# subset, used as the boundary-detection training target.


class BoundaryHead(nn.Module):
    """Frame-level boundary classifier on top of backbone hidden states."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)  # (B, T) boundary logits


def frames_to_boundary_labels(segments: list[tuple[float, float]], n_frames: int, frame_stride_sec: float) -> torch.Tensor:
    """Convert (start, end) syllable segments into a per-frame binary label
    that is 1 at the frame closest to each segment boundary, 0 elsewhere."""
    labels = torch.zeros(n_frames)
    for start, _end in segments:
        frame_idx = min(int(round(start / frame_stride_sec)), n_frames - 1)
        labels[frame_idx] = 1.0
    return labels


class KhmerSegmentationDataset(torch.utils.data.Dataset):
    """Pairs of (wav_path, pseudo/gold boundary segments)."""

    def __init__(self, manifest_df: pd.DataFrame, boundaries: dict[str, list[tuple[float, float]]]):
        self.rows = manifest_df.to_dict("records")
        self.boundaries = boundaries

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return row["path"], self.boundaries[row["path"]]


def finetune_segmenter(
    manifest_path: str,
    boundaries_json: str,
    init_ckpt: str,
    out_ckpt: str,
    mode: Literal["head_only", "full_model"] = "head_only",
    lr: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 8,
    frame_stride_sec: float = 0.02,
    device: str | None = None,
):
    """Fine-tune Sylber's boundary head (Step 3).

    `boundaries_json` maps wav path -> list of [start, end] pseudo/corrected
    segment boundaries (seconds), produced by zero-shot inference + manual
    correction on ~500-1000 utterances (see doc Step 3.1).

    `mode="head_only"` freezes the CNN+Transformer backbone and only trains
    BoundaryHead — the conservative, sample-efficient strategy. `mode="full_model"`
    unfreezes everything, viable once enough Khmer hours are available (~728h).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    segmenter = load_segmenter(init_ckpt)
    backbone = segmenter.model  # HuBERT-style nn.Module exposed by sylber.Segmenter
    hidden_size = backbone.config.hidden_size if hasattr(backbone, "config") else backbone.hidden_size

    if mode == "head_only":
        for p in backbone.parameters():
            p.requires_grad = False
    elif mode == "full_model":
        for p in backbone.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown mode {mode!r}")

    head = BoundaryHead(hidden_size).to(device)
    backbone.to(device)

    df = pd.read_csv(manifest_path)
    boundaries = {k: [tuple(seg) for seg in v] for k, v in json.loads(Path(boundaries_json).read_text()).items()}
    df = df[df["path"].isin(boundaries.keys())].reset_index(drop=True)
    if df.empty:
        raise ValueError("No manifest rows match the provided boundaries_json paths.")

    dataset = KhmerSegmentationDataset(df, boundaries)

    trainable_params = list(head.parameters())
    if mode == "full_model":
        trainable_params += list(backbone.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    import soundfile as sf

    backbone.train(mode == "full_model")
    head.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b
        )
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            batch_loss = 0.0
            for wav_path, segs in batch:
                audio, sr = sf.read(wav_path, dtype="float32")
                wav_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

                hidden_states = backbone.extract_features(wav_tensor) if hasattr(
                    backbone, "extract_features"
                ) else backbone(wav_tensor).last_hidden_state

                logits = head(hidden_states)  # (1, T)
                labels = frames_to_boundary_labels(segs, logits.shape[1], frame_stride_sec).to(device)
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(0), labels)
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            epoch_loss += batch_loss
            n_batches += 1

        log.info("epoch %d/%d mean loss=%.4f", epoch + 1, epochs, epoch_loss / max(n_batches, 1))

    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "boundary_head_state_dict": head.state_dict(),
            "mode": mode,
            "init_ckpt": init_ckpt,
        },
        out_path,
    )
    log.info("Saved fine-tuned checkpoint to %s", out_path)
    return out_path


def cmd_finetune(args):
    finetune_segmenter(
        manifest_path=args.manifest,
        boundaries_json=args.boundaries,
        init_ckpt=args.init_ckpt,
        out_ckpt=args.out_ckpt,
        mode=args.mode,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Step 2: zero-shot segmentation eval")
    p_eval.add_argument("--manifest", required=True)
    p_eval.add_argument("--checkpoint", default="sylber")
    p_eval.add_argument("--n-samples", type=int, default=100)
    p_eval.add_argument("--out", default="results/zero_shot_evaluation.txt")
    p_eval.set_defaults(func=cmd_eval)

    p_ft = sub.add_parser("finetune", help="Step 3: fine-tune boundary head on Khmer")
    p_ft.add_argument("--manifest", required=True)
    p_ft.add_argument("--boundaries", required=True, help="JSON: wav_path -> [[start,end], ...]")
    p_ft.add_argument("--init-ckpt", default="sylber")
    p_ft.add_argument("--out-ckpt", default="models/sylber_checkpoints/sylber_khmer_v1.pth")
    p_ft.add_argument("--mode", choices=["head_only", "full_model"], default="head_only")
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--epochs", type=int, default=10)
    p_ft.add_argument("--batch-size", type=int, default=8)
    p_ft.set_defaults(func=cmd_finetune)

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.func(args)
