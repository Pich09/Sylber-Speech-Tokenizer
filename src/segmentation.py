"""Steps 2/3: Sylber segmentation — zero-shot baseline eval and fine-tuning
on Khmer (and optionally Khmer+English code-switched, Step 3b).

Zero-shot eval (Step 2):
    python src/segmentation.py eval --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
        --n-samples 100 --out results/zero_shot_evaluation.txt

Threshold sweep (free, zero-training check before committing to Step 3):
    python src/segmentation.py sweep-thresholds --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
        --n-samples 30 --out results/threshold_sweep.csv

Fine-tuning (Step 3):
    python src/segmentation.py finetune --manifest ... --mode last_layer \
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
import torch.nn.functional as F
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_segmenter(checkpoint: str = "sylber", norm_threshold: float | None = None, merge_threshold: float | None = None):
    """Load a pretrained (or Khmer-fine-tuned) Sylber segmenter.

    Requires the `sylber` package (Berkeley-Speech-Group/sylber). Import is
    lazy so the rest of this module (config/manifest handling) can be tested
    without the heavy model dependency installed.

    `sylber.Segmenter(model_ckpt=...)` expects `checkpoint` to point at its
    *own* raw HubertModel state_dict (or the "sylber" HF Hub name) — not the
    `{"backbone_state_dict": ..., "init_ckpt": ...}` wrapper this module's
    `finetune_segmenter`/`publish_checkpoint.py` write. Passing one of those
    straight through would silently no-op under `strict=False` (no keys
    would match). Detect that case and load it the right way: build the
    Segmenter from its recorded base checkpoint, then overlay the
    fine-tuned backbone weights.

    `norm_threshold`/`merge_threshold` override Sylber's fixed segmentation
    heuristic (see `sylber.utils.segment_utils.get_segment`) — the L2-norm
    voice-activity cutoff and the cosine-similarity merge cutoff, English-
    tuned defaults of 2.6/0.8 respectively. None keeps Sylber's own default.
    """
    from sylber import Segmenter

    kwargs = {}
    if norm_threshold is not None:
        kwargs["norm_threshold"] = norm_threshold
    if merge_threshold is not None:
        kwargs["merge_threshold"] = merge_threshold

    if checkpoint not in (None, "sylber") and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
            segmenter = Segmenter(model_ckpt=ckpt.get("init_ckpt", "sylber"), **kwargs)
            segmenter.speech_model.load_state_dict(ckpt["backbone_state_dict"], strict=False)
            return segmenter

    return Segmenter(model_ckpt=checkpoint, **kwargs)


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
        if isinstance(out, dict):
            if "segments" not in out:
                raise RuntimeError(
                    f"Installed sylber version's output has no 'segments' key (got keys={list(out.keys())}); "
                    "adjust the marked spots in segmentation.py/discretization.py/tokenizer.py (search for 'segment_features')."
                )
            segments = out["segments"]
        else:
            segments = out
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
# Threshold sweep — a free, zero-training alternative/precursor to Step 3
# ---------------------------------------------------------------------------
# get_segment()'s norm_threshold/merge_threshold are fixed, English-tuned
# constants (2.6/0.8), never swept. Since Sylber's zero-shot Khmer token
# count runs ~34% below the grapheme-cluster label length on average (see
# docs/post-benchmark-roadmap.md's pilot result), it's worth checking how
# much of that gap a different threshold closes before committing to the
# annotation-heavy Step 3 fine-tune. Each utterance's backbone hidden
# states are computed once and cached, then get_segment() (cheap — no
# backbone forward pass) is re-run per threshold combo directly on the
# cached states, so the sweep doesn't pay repeated GPU cost per combo.


def extract_hidden_states_and_labels(segmenter, df: pd.DataFrame) -> list[dict]:
    """One backbone forward pass per utterance, cached alongside the
    transcript's grapheme-cluster label length (see train_ctc.py's
    tokenize_transcript) so the threshold sweep below can re-run
    get_segment() cheaply per combo without repeating this."""
    import soundfile as sf

    from train_ctc import tokenize_transcript  # local import: src/ on path

    cache = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting hidden states"):
        audio, sr = sf.read(row["path"], dtype="float32")
        # match sylber.Segmenter.__call__'s own preprocessing
        audio = (audio - audio.mean()) / audio.std()
        wav_tensor = torch.from_numpy(audio).unsqueeze(0).to(segmenter.device)
        with torch.no_grad():
            hidden_states = segmenter.speech_model(wav_tensor).last_hidden_state.squeeze(0).cpu().numpy()

        transcript = row.get("transcript", "") or ""
        label_len = len(tokenize_transcript(transcript)) if transcript else None
        cache.append(
            {
                "path": row["path"],
                "hidden_states": hidden_states,
                "label_len": label_len,
                "duration_sec": row.get("duration_sec"),
            }
        )
    return cache


def sweep_thresholds(cache: list[dict], norm_thresholds: list[float], merge_thresholds: list[float]) -> pd.DataFrame:
    from sylber.utils.segment_utils import get_segment

    rows = []
    for nt in norm_thresholds:
        for mt in merge_thresholds:
            token_rates, ratios = [], []
            n_viable = n_with_label = 0
            for item in cache:
                segments = get_segment(item["hidden_states"], nt, mt)
                T = len(segments)
                if item["duration_sec"]:
                    token_rates.append(T / item["duration_sec"])
                if item["label_len"]:
                    n_with_label += 1
                    ratios.append(T / item["label_len"])
                    if T >= item["label_len"]:
                        n_viable += 1
            rows.append(
                {
                    "norm_threshold": nt,
                    "merge_threshold": mt,
                    "mean_token_rate_hz": float(np.mean(token_rates)) if token_rates else float("nan"),
                    "mean_T_over_L": float(np.mean(ratios)) if ratios else float("nan"),
                    "pct_ctc_viable": n_viable / n_with_label if n_with_label else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def cmd_sweep_thresholds(args):
    df = pd.read_csv(args.manifest)
    df["transcript"] = df.get("transcript", pd.Series(dtype=str)).fillna("")
    sample = df.sample(n=min(args.n_samples, len(df)), random_state=42)

    segmenter = load_segmenter(args.checkpoint)
    cache = extract_hidden_states_and_labels(segmenter, sample)

    result = sweep_thresholds(cache, args.norm_thresholds, args.merge_thresholds)
    result = result.sort_values(["pct_ctc_viable", "mean_T_over_L"], ascending=[False, True])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    log.info("Threshold sweep (n=%d utterances):\n%s", len(cache), result.to_string(index=False))
    log.info("Wrote %s", out_path)
    best = result.iloc[0]
    log.info(
        "Best by pct_ctc_viable: norm_threshold=%.2f merge_threshold=%.2f -> "
        "pct_ctc_viable=%.1f%% mean_T/L=%.2f mean_token_rate_hz=%.2f "
        "(Sylber's own default is norm_threshold=2.6, merge_threshold=0.8)",
        best["norm_threshold"], best["merge_threshold"], best["pct_ctc_viable"] * 100,
        best["mean_T_over_L"], best["mean_token_rate_hz"],
    )


# ---------------------------------------------------------------------------
# Step 3: Fine-tuning
# ---------------------------------------------------------------------------
# Sylber has no separate learned boundary-prediction head to fine-tune.
# Segment boundaries come from a fixed, rule-based heuristic applied to the
# backbone's own hidden states (sylber.utils.segment_utils.get_segment): a
# frame counts as voiced if its L2-norm >= norm_threshold, and consecutive
# voiced frames merge into one segment while their running-average cosine
# similarity stays >= merge_threshold (English-tuned defaults 2.6/0.8).
# There's nothing to attach a classifier head to, and get_segment() would
# never consult one anyway. So instead of training a disconnected boundary
# classifier, this fine-tunes the backbone directly with a loss that targets
# the same quantity get_segment() actually thresholds: pull adjacent-frame
# cosine similarity up within a true target segment (so get_segment merges
# them) and down across a true target boundary (so get_segment splits there).


def frames_to_segment_ids(segments: list[tuple[float, float]], n_frames: int, frame_stride_sec: float) -> torch.Tensor:
    """Per-frame target segment index (by `segments`' order); -1 for frames
    not covered by any target segment (e.g. silence/gaps — get_segment's
    separate norm_threshold voice-activity check already handles those, so
    they're excluded from the boundary-contrastive loss below rather than
    guessed at)."""
    seg_ids = torch.full((n_frames,), -1, dtype=torch.long)
    for i, (start, end) in enumerate(segments):
        s = min(int(round(start / frame_stride_sec)), n_frames - 1)
        e = min(max(int(round(end / frame_stride_sec)), s + 1), n_frames)
        seg_ids[s:e] = i
    return seg_ids


def boundary_contrastive_loss(hidden_states: torch.Tensor, seg_ids: torch.Tensor, margin: float) -> torch.Tensor | None:
    """(T, D) hidden states + (T,) target segment ids -> a scalar loss
    mirroring get_segment()'s own merge decision: adjacent-frame cosine
    similarity should be high within a target segment (so get_segment merges
    them) and below `margin` across a target boundary (so it splits there).
    `margin` should generally be at or below the merge_threshold get_segment
    will actually run with at inference, so a successful fit produces
    similarities its own check will act on. Returns None if there are no
    valid (non-gap) adjacent frame pairs to learn from."""
    seg_i, seg_j = seg_ids[:-1], seg_ids[1:]
    valid = (seg_i >= 0) & (seg_j >= 0)
    if not valid.any():
        return None

    sims = F.cosine_similarity(hidden_states[:-1][valid], hidden_states[1:][valid], dim=-1)
    same = seg_i[valid] == seg_j[valid]

    losses = []
    if same.any():
        losses.append((1.0 - sims[same]).mean())
    if (~same).any():
        losses.append(F.relu(sims[~same] - margin).mean())
    return torch.stack(losses).mean() if losses else None


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
    mode: Literal["last_layer", "full_model"] = "last_layer",
    lr: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 8,
    frame_stride_sec: float = 0.02,
    margin: float = 0.75,
    device: str | None = None,
):
    """Fine-tune Sylber's backbone on Khmer boundary supervision (Step 3).

    `boundaries_json` maps wav path -> list of [start, end] pseudo/corrected
    segment boundaries (seconds), produced by zero-shot inference + manual
    correction on ~500-1000 utterances (see doc Step 3.1).

    `mode="last_layer"` freezes all but the backbone's last transformer
    layer — the conservative, sample-efficient strategy. `mode="full_model"`
    unfreezes everything, viable once enough Khmer hours are available
    (~728h). (No "head_only" option — see the module-level note above on
    why Sylber has no boundary head to isolate.)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    segmenter = load_segmenter(init_ckpt)
    backbone = segmenter.speech_model  # the HubertModel sylber.Segmenter actually wraps

    for p in backbone.parameters():
        p.requires_grad = False
    if mode == "last_layer":
        if not hasattr(backbone, "encoder") or not hasattr(backbone.encoder, "layers"):
            raise RuntimeError(
                "backbone.encoder.layers not found; the installed sylber/transformers version's "
                "HubertModel structure doesn't match what mode='last_layer' assumes — use "
                "mode='full_model' instead, or adjust this spot."
            )
        for p in backbone.encoder.layers[-1].parameters():
            p.requires_grad = True
    elif mode == "full_model":
        for p in backbone.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown mode {mode!r}")

    backbone.to(device)

    df = pd.read_csv(manifest_path)
    boundaries = {k: [tuple(seg) for seg in v] for k, v in json.loads(Path(boundaries_json).read_text()).items()}
    df = df[df["path"].isin(boundaries.keys())].reset_index(drop=True)
    if df.empty:
        raise ValueError("No manifest rows match the provided boundaries_json paths.")

    dataset = KhmerSegmentationDataset(df, boundaries)

    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    import soundfile as sf

    backbone.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        n_skipped = 0
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b
        )
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            batch_losses = []
            for wav_path, segs in batch:
                audio, sr = sf.read(wav_path, dtype="float32")
                # Match sylber.Segmenter.__call__'s own preprocessing so
                # fine-tuning sees the same input distribution as inference.
                audio = (audio - audio.mean()) / audio.std()
                wav_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

                hidden_states = backbone(wav_tensor).last_hidden_state.squeeze(0)  # (T, D)
                seg_ids = frames_to_segment_ids(segs, hidden_states.shape[0], frame_stride_sec).to(device)
                loss = boundary_contrastive_loss(hidden_states, seg_ids, margin)
                if loss is None:
                    n_skipped += 1
                    continue
                batch_losses.append(loss)

            if not batch_losses:
                continue
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            n_batches += 1

        log.info(
            "epoch %d/%d mean loss=%.4f (batches used=%d, utterances skipped=%d — fewer than 2 "
            "non-gap frames of target segmentation)",
            epoch + 1, epochs, epoch_loss / max(n_batches, 1), n_batches, n_skipped,
        )

    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"backbone_state_dict": backbone.state_dict(), "mode": mode, "init_ckpt": init_ckpt},
        out_path,
    )
    log.info("Saved fine-tuned checkpoint to %s", out_path)
    return out_path


def cmd_finetune(args):
    from log_utils import add_file_handler

    add_file_handler(f"logs/segmentation_finetune_{Path(args.out_ckpt).stem}.log")
    finetune_segmenter(
        manifest_path=args.manifest,
        boundaries_json=args.boundaries,
        init_ckpt=args.init_ckpt,
        out_ckpt=args.out_ckpt,
        mode=args.mode,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        margin=args.margin,
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

    p_sweep = sub.add_parser(
        "sweep-thresholds",
        help="free, zero-training check: does a different norm_threshold/merge_threshold close the T<L gap",
    )
    p_sweep.add_argument("--manifest", required=True)
    p_sweep.add_argument("--checkpoint", default="sylber")
    p_sweep.add_argument("--n-samples", type=int, default=30)
    p_sweep.add_argument("--norm-thresholds", type=float, nargs="+", default=[1.5, 2.0, 2.3, 2.6, 3.0, 3.5])
    p_sweep.add_argument("--merge-thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    p_sweep.add_argument("--out", default="results/threshold_sweep.csv")
    p_sweep.set_defaults(func=cmd_sweep_thresholds)

    p_ft = sub.add_parser("finetune", help="Step 3: fine-tune Sylber's backbone on Khmer boundary supervision")
    p_ft.add_argument("--manifest", required=True)
    p_ft.add_argument("--boundaries", required=True, help="JSON: wav_path -> [[start,end], ...]")
    p_ft.add_argument("--init-ckpt", default="sylber")
    p_ft.add_argument("--out-ckpt", default="models/sylber_checkpoints/sylber_khmer_v1.pth")
    p_ft.add_argument("--mode", choices=["last_layer", "full_model"], default="last_layer")
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--epochs", type=int, default=10)
    p_ft.add_argument("--batch-size", type=int, default=8)
    p_ft.add_argument(
        "--margin", type=float, default=0.75,
        help="cosine-similarity margin for cross-boundary frame pairs; keep <= the merge_threshold "
        "get_segment() will run with at inference (Sylber's default is 0.8)",
    )
    p_ft.set_defaults(func=cmd_finetune)

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.func(args)
