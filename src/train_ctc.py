"""CTC probe: a cheap, direct test of whether an encoder's features are
usable for Khmer ASR, without needing the ~1K+ hours a discrete-token SLM
comparison (train_slm.py/compare_encoders.py) requires to be trustworthy
(see docs/path-a-encoder-comparison.md's Sylber-vs-HuBERT/Whisper results
and the Sylber papers' own uLM data-scale floor).

This is the scenario Sylber 2.0's paper validates at exactly this data
scale (Section 6.3, "Low-Resource ASR": 20-50h per language) — a frozen (or
lightly fine-tuned) encoder feeding a small supervised CTC decoder trained
on the manifest's `transcript` column, evaluated by character error rate
(CER). Works with any encoder from encoders.py/segmentation.py (sylber,
hubert, whisper) via the same `segment_features` interface, so it can also
be used as a quick per-encoder screening step if you want to compare
without waiting for the full discrete-SLM pipeline.

Usage:
    python src/train_ctc.py train --manifest <manifest.csv> --encoder sylber
    python src/train_ctc.py train --manifest <manifest.csv> --encoder hubert --checkpoint facebook/hubert-base-ls960
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import regex
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def tokenize_transcript(text: str) -> list[str]:
    """Split into Unicode extended grapheme clusters (UAX #29) rather than
    raw codepoints. Khmer syllables are typically encoded as a base
    consonant plus one or more combining codepoints (e.g. the coeng
    U+17D2 subscript-forming sequence, vowel signs, diacritics) — splitting
    on raw codepoints overcounts label length by roughly 2-4x relative to
    the number of spoken syllables, which made every utterance fail CTC's
    input_length >= target_length requirement against Sylber's ~1-token-
    per-syllable rate (HuBERT/Whisper's ~50Hz rate has enough headroom
    that this didn't surface there). Grapheme clusters keep label units
    close to one per syllable instead."""
    return regex.findall(r"\X", text)


def build_char_vocab(transcripts: list[str]) -> dict[str, int]:
    """Grapheme-cluster vocabulary from training transcripts (see
    `tokenize_transcript`). CTC's blank class is appended as the final ID
    (len(vocab)), not included here."""
    units = sorted({u for t in transcripts for u in tokenize_transcript(t)})
    return {u: i for i, u in enumerate(units)}


def edit_distance(pred: list, ref: list) -> int:
    """Levenshtein distance, used for character error rate."""
    n, m = len(pred), len(ref)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if pred[i - 1] == ref[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def greedy_ctc_decode(logits: torch.Tensor, blank_id: int) -> list[int]:
    """Collapse repeats and drop blanks from an argmax decode. `logits` is
    (T, vocab_size+1)."""
    ids = logits.argmax(dim=-1).tolist()
    out = []
    prev = None
    for i in ids:
        if i != prev and i != blank_id:
            out.append(i)
        prev = i
    return out


class CTCHead(nn.Module):
    """Small supervised decoder on top of frozen encoder features — the
    "probe" in CTC-probe. One hidden layer rather than a bare linear layer,
    for a bit more capacity without turning this into a full ASR model."""

    def __init__(self, hidden_size: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size + 1),  # +1 for CTC blank
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (T, vocab_size+1) logits


def extract_features(encoder, wav_path: str, device: str) -> torch.Tensor | None:
    """Frozen encoder forward pass -> (T, D) tensor on `device`, or None if
    the utterance produced no segments/frames (skip it)."""
    with torch.no_grad():
        out = encoder(wav_path, in_second=True)
        feats = np.asarray(out["segment_features"])
    if feats.shape[0] == 0:
        return None
    return torch.from_numpy(feats).float().to(device)


def run_epoch(
    encoder,
    head: CTCHead,
    df: pd.DataFrame,
    char2id: dict[str, int],
    blank_id: int,
    device: str,
    ctc_loss_fn: nn.CTCLoss,
    optimizer: torch.optim.Optimizer | None,
    scheduler=None,
    batch_size: int = 8,
    desc: str = "",
) -> dict:
    """One pass over `df`. If `optimizer` is None, runs in eval mode (no
    backward, computes CER instead of training). Skips utterances where the
    encoder's output is shorter than the transcript (CTC requires
    input_length >= target_length) or the transcript is empty — see
    `skip_reasons` in the returned dict for a breakdown of which."""
    training = optimizer is not None
    head.train(training)

    total_loss = 0.0
    total_edits = 0
    total_ref_chars = 0
    n_skipped = 0
    n_used = 0
    skip_reasons = {"empty_transcript": 0, "no_labelable_units": 0, "input_shorter_than_target": 0}
    length_skip_examples = []  # up to 5 (feats_shape, label_length) pairs for diagnosing *why* T < L

    rows = df.to_dict("records")
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    for batch in tqdm(batches, desc=desc):
        if training:
            optimizer.zero_grad()
        n_in_batch = 0

        for row in batch:
            transcript = row.get("transcript", "")
            if not isinstance(transcript, str) or not transcript:
                n_skipped += 1
                skip_reasons["empty_transcript"] += 1
                continue
            label_ids = [char2id[c] for c in tokenize_transcript(transcript) if c in char2id]
            if not label_ids:
                n_skipped += 1
                skip_reasons["no_labelable_units"] += 1
                continue

            feats = extract_features(encoder, row["path"], device)
            if feats is None or feats.shape[0] < len(label_ids):
                # CTC requires input_length >= target_length; syllable-rate
                # encoders (Sylber) hit this more often than frame-rate ones
                # (HuBERT/Whisper) on short utterances.
                n_skipped += 1
                skip_reasons["input_shorter_than_target"] += 1
                if len(length_skip_examples) < 5:
                    length_skip_examples.append((tuple(feats.shape) if feats is not None else None, len(label_ids)))
                continue

            # Eval passes never call .backward(), so skip building the
            # autograd graph through the head for them.
            with torch.enable_grad() if training else torch.no_grad():
                logits = head(feats)  # (T, V+1)
                log_probs = F.log_softmax(logits, dim=-1).unsqueeze(1)  # (T, 1, V+1)
                input_length = torch.tensor([feats.shape[0]])
                target_length = torch.tensor([len(label_ids)])
                targets = torch.tensor(label_ids, dtype=torch.long)
                loss = ctc_loss_fn(log_probs, targets, input_length, target_length)
            if training:
                loss.backward()
            total_loss += loss.item()
            n_in_batch += 1
            n_used += 1

            if not training:
                pred_ids = greedy_ctc_decode(logits, blank_id)
                total_edits += edit_distance(pred_ids, label_ids)
                total_ref_chars += len(label_ids)

        if training and n_in_batch > 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    result = {
        "mean_loss": total_loss / max(n_used, 1),
        "n_used": n_used,
        "n_skipped": n_skipped,
        "skip_reasons": skip_reasons,
        "length_skip_examples": length_skip_examples,
    }
    if not training:
        result["cer"] = total_edits / max(total_ref_chars, 1)
    return result


def cmd_train(args):
    from encoders import load_encoder
    from log_utils import add_file_handler

    add_file_handler(f"logs/ctc_probe_{args.encoder}.log")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.manifest)
    df["transcript"] = df["transcript"].fillna("")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if args.max_utterances:
        train_df = train_df.iloc[: args.max_utterances].reset_index(drop=True)
        val_df = val_df.iloc[: max(args.max_utterances // 5, 1)].reset_index(drop=True)

    if train_df.empty:
        raise ValueError(f"No rows with split=='train' in {args.manifest}")

    char2id = build_char_vocab(train_df["transcript"].tolist())
    blank_id = len(char2id)
    log.info("Grapheme-cluster vocab size=%d (+1 CTC blank) built from %d train transcripts", len(char2id), len(train_df))

    encoder = load_encoder(
        args.encoder, checkpoint=args.checkpoint, device=device,
        norm_threshold=args.norm_threshold, merge_threshold=args.merge_threshold,
    )

    # Infer hidden size from one utterance's features rather than assuming
    # a fixed dim, since sylber/hubert/whisper each expose a different one.
    probe_feats = extract_features(encoder, train_df.iloc[0]["path"], device)
    if probe_feats is None:
        raise RuntimeError(f"First training utterance {train_df.iloc[0]['path']} produced no encoder output.")
    hidden_size = probe_feats.shape[1]

    head = CTCHead(hidden_size, len(char2id), dropout=args.dropout).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    n_train_batches = max(-(-len(train_df) // args.batch_size), 1)  # ceil division: batches() below always rounds up
    total_steps = n_train_batches * args.epochs
    scheduler = None
    if args.warmup_ratio > 0:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
        )

    history = []
    for epoch in range(args.epochs):
        train_stats = run_epoch(
            encoder, head, train_df, char2id, blank_id, device, ctc_loss_fn,
            optimizer=optimizer, scheduler=scheduler, batch_size=args.batch_size,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
        )
        eval_stats = run_epoch(
            encoder, head, val_df, char2id, blank_id, device, ctc_loss_fn,
            optimizer=None, batch_size=args.batch_size, desc=f"val epoch {epoch + 1}/{args.epochs}",
        ) if not val_df.empty else {
            "mean_loss": None, "cer": None, "n_used": 0, "n_skipped": 0, "skip_reasons": {}, "length_skip_examples": [],
        }

        log.info(
            "epoch %d/%d train_loss=%.4f (used=%d skipped=%d) val_loss=%s val_cer=%s (used=%d skipped=%d)",
            epoch + 1, args.epochs, train_stats["mean_loss"], train_stats["n_used"], train_stats["n_skipped"],
            eval_stats["mean_loss"], eval_stats.get("cer"), eval_stats["n_used"], eval_stats["n_skipped"],
        )
        if epoch == 0 and train_stats["n_used"] == 0:
            log.warning(
                "Every training utterance was skipped — skip_reasons=%s. If "
                "'input_shorter_than_target' dominates, this encoder's token rate is too low "
                "for these transcripts' label length; if 'no_labelable_units' or "
                "'empty_transcript' dominates, check the manifest's transcript column. "
                "Example (encoder_output_shape, label_length) pairs: %s — if the shape's first "
                "dimension looks fixed/tiny (e.g. always 1) regardless of label_length, the "
                "encoder adapter is likely returning a batch dimension instead of the true "
                "sequence length; if it varies but is still consistently smaller than "
                "label_length, the encoder's segmentation is under-producing tokens relative to "
                "actual syllable count for this language.",
                train_stats["skip_reasons"], train_stats["length_skip_examples"],
            )
        history.append({"epoch": epoch + 1, "train": train_stats, "val": eval_stats})

    out_ckpt = Path(args.out_ckpt or f"models/ctc_probe_{args.encoder}.pth")
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"head_state_dict": head.state_dict(), "char2id": char2id, "blank_id": blank_id, "encoder": args.encoder, "hidden_size": hidden_size},
        out_ckpt,
    )
    log.info("Saved CTC probe head to %s", out_ckpt)

    results_path = Path("results/downstream_eval")
    results_path.mkdir(parents=True, exist_ok=True)
    final_cer = history[-1]["val"].get("cer") if history else None
    (results_path / f"ctc_probe_{args.encoder}.json").write_text(
        json.dumps(
            {
                "encoder": args.encoder,
                "checkpoint": args.checkpoint,
                "vocab_size": len(char2id),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "epochs": args.epochs,
                "final_val_cer": final_cer,
                "history": history,
            },
            indent=2,
        )
    )
    log.info("Wrote results/downstream_eval/ctc_probe_%s.json (final val CER=%s)", args.encoder, final_cer)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a CTC head on a frozen encoder's features and report val CER")
    p_train.add_argument("--manifest", required=True)
    p_train.add_argument("--encoder", choices=["sylber", "hubert", "whisper"], default="sylber")
    p_train.add_argument("--checkpoint", default=None, help="encoder checkpoint override (see encoders.py DEFAULT_CHECKPOINTS)")
    p_train.add_argument("--out-ckpt", default=None, help="default: models/ctc_probe_<encoder>.pth")
    p_train.add_argument("--max-utterances", type=int, default=None, help="cap train split size for a cheap pilot run")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--warmup-ratio", type=float, default=0.05, help="linear warmup fraction of total steps; 0 disables the scheduler")
    p_train.add_argument("--device", default=None)
    p_train.add_argument(
        "--norm-threshold", type=float, default=None,
        help="sylber only: override its fixed voice-activity threshold (default 2.6); see "
        "docs/post-benchmark-roadmap.md's 'Free lever found' section",
    )
    p_train.add_argument(
        "--merge-threshold", type=float, default=None,
        help="sylber only: override its fixed segment-merge cosine-similarity threshold (default 0.8) "
        "— raising it (e.g. 0.98) closes most of the T<L gap that skips every Sylber utterance at the "
        "default; see docs/post-benchmark-roadmap.md's 'Free lever found' section",
    )
    p_train.set_defaults(func=cmd_train)

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.func(args)
