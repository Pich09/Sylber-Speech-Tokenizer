"""Step 5 "Baseline Comparison": aggregate the per-encoder SLM benchmark
results (produced by discretization.py extract/sweep/fit + train_slm.py
encode/train, run once per --encoder) into one comparison table.

Reads, for each encoder:
    data/embeddings/<encoder>/meta.json          (token rate, embedding dim)
    results/downstream_eval/slm_eval_<encoder>.json  (eval_loss, perplexity, vocab_size)

Usage:
    python src/compare_encoders.py --encoders sylber hubert whisper
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_encoder_row(encoder: str, embeddings_dir: Path, results_dir: Path) -> dict | None:
    meta_path = embeddings_dir / encoder / "meta.json"
    eval_path = results_dir / f"slm_eval_{encoder}.json"

    if not meta_path.exists():
        log.warning("Skipping %s: %s not found (run discretization.py extract --encoder %s first)", encoder, meta_path, encoder)
        return None
    if not eval_path.exists():
        log.warning("Skipping %s: %s not found (run train_slm.py train --encoder %s first)", encoder, eval_path, encoder)
        return None

    meta = json.loads(meta_path.read_text())
    eval_metrics = json.loads(eval_path.read_text())

    return {
        "encoder": encoder,
        "token_rate_hz": meta.get("mean_token_rate_hz"),
        "embedding_dim": meta.get("embedding_dim"),
        "n_syllables": meta.get("n_syllables"),
        "vocab_size": eval_metrics.get("vocab_size"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "perplexity": eval_metrics.get("perplexity"),
    }


def cmd_compare(args):
    embeddings_dir = Path(args.embeddings_dir)
    results_dir = Path(args.results_dir)

    rows = [r for r in (load_encoder_row(e, embeddings_dir, results_dir) for e in args.encoders) if r is not None]
    if not rows:
        log.error("No encoder results found under %s / %s; nothing to compare.", embeddings_dir, results_dir)
        return

    df = pd.DataFrame(rows).sort_values("perplexity", na_position="last")

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "encoder_comparison.csv"
    df.to_csv(csv_path, index=False)

    md_path = results_dir / "encoder_comparison.md"
    md_path.write_text(
        "# SLM Encoder Benchmark — Sylber vs. baselines\n\n"
        + df.to_markdown(index=False)
        + "\n\nLower perplexity is better; lower token_rate_hz means less SLM context/compute per second of audio.\n"
    )

    log.info("Comparison:\n%s", df.to_string(index=False))
    log.info("Wrote %s and %s", csv_path, md_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+", choices=["sylber", "hubert", "whisper"], default=["sylber", "hubert", "whisper"])
    parser.add_argument("--embeddings-dir", default="data/embeddings")
    parser.add_argument("--results-dir", default="results/downstream_eval")
    parser.set_defaults(func=cmd_compare)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.func(args)
