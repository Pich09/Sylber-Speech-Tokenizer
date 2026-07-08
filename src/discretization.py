"""Step 4: Discretization — extract syllable-averaged embeddings and build
a k-means token vocabulary.

Usage:
    python src/discretization.py extract --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
        --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth --out data/embeddings/khmer_syllable_embeddings.npy

    python src/discretization.py sweep --embeddings data/embeddings/khmer_syllable_embeddings.npy \
        --k-sweep 5000 10000 20000 40000
    # (reads total duration from the .meta.json written alongside --embeddings by `extract`)

    python src/discretization.py fit --embeddings data/embeddings/khmer_syllable_embeddings.npy \
        --k 10000 --out models/khmer_kmeans_10k.pkl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def extract_syllable_embeddings(manifest_path: str, checkpoint: str) -> tuple[np.ndarray, list[int], float]:
    """Run the (fine-tuned) segmenter over every utterance, mean-pool SSL
    features within each syllable segment, and return:
      - embeddings: (N_syllables, D) array
      - counts_per_utterance: syllables per utterance (for token-rate stats)
      - total_duration_sec
    """
    from segmentation import load_segmenter  # local import: src/ on path

    segmenter = load_segmenter(checkpoint)
    df = pd.read_csv(manifest_path)

    all_embeddings = []
    counts = []
    total_duration = 0.0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting embeddings"):
        out = segmenter(row["path"], in_second=True)
        # sylber's Segmenter exposes both boundaries and the mean-pooled
        # per-segment embedding (`segment_features`) when available.
        if isinstance(out, dict) and "segment_features" in out:
            feats = np.asarray(out["segment_features"])
        else:
            raise RuntimeError(
                "Installed sylber version does not expose 'segment_features'; "
                "mean-pool 'features' over 'segments' boundaries manually."
            )
        all_embeddings.append(feats)
        counts.append(len(feats))
        total_duration += row["duration_sec"]

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, counts, total_duration


def cmd_extract(args):
    embeddings, counts, total_duration = extract_syllable_embeddings(args.manifest, args.checkpoint)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "n_syllables": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "n_utterances": len(counts),
                "total_duration_sec": total_duration,
                "mean_token_rate_hz": sum(counts) / total_duration if total_duration else 0.0,
            },
            indent=2,
        )
    )
    log.info("Saved %s embeddings -> %s (meta: %s)", embeddings.shape, out_path, meta_path)


def cluster_balance(labels: np.ndarray, k: int, top_n: int = 10) -> float:
    """Fraction of all assignments falling in the top-N most frequent clusters."""
    counts = np.bincount(labels, minlength=k)
    top = np.sort(counts)[::-1][:top_n]
    return float(top.sum() / counts.sum())


def sweep_k(embeddings: np.ndarray, k_values: list[int], total_duration_sec: float, random_state: int = 42) -> pd.DataFrame:
    from sklearn.cluster import MiniBatchKMeans

    rows = []
    for k in k_values:
        log.info("Fitting k-means with K=%d ...", k)
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=10000, n_init="auto")
        labels = km.fit_predict(embeddings)

        token_rate_hz = embeddings.shape[0] / total_duration_sec if total_duration_sec else float("nan")
        balance = cluster_balance(labels, k)
        rows.append(
            {
                "k": k,
                "token_rate_hz": token_rate_hz,
                "top10_cluster_share": balance,
                "inertia": km.inertia_,
            }
        )
        log.info("K=%d: token_rate=%.2fHz top10_share=%.1f%% inertia=%.1f", k, token_rate_hz, balance * 100, km.inertia_)

    return pd.DataFrame(rows)


def cmd_sweep(args):
    embeddings = np.load(args.embeddings)
    meta = json.loads(Path(args.embeddings).with_suffix(".meta.json").read_text())
    df = sweep_k(embeddings, args.k_sweep, meta["total_duration_sec"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Sweep results:\n%s", df.to_string(index=False))
    log.info("Wrote sweep results to %s", out_path)
    log.info(
        "Sanity check: expect token_rate_hz in [4,5]; flag any K with top10_cluster_share > 0.5 "
        "(over-clustering / data quality issue)."
    )


def fit_final_kmeans(embeddings: np.ndarray, k: int, random_state: int = 42, minibatch: bool = True, batch_size: int = 10000):
    from sklearn.cluster import KMeans, MiniBatchKMeans

    if minibatch:
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=batch_size, n_init="auto")
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(embeddings)
    return km


def cmd_fit(args):
    embeddings = np.load(args.embeddings)
    km = fit_final_kmeans(embeddings, args.k, minibatch=not args.full_kmeans)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, out_path)
    log.info("Saved k-means model (K=%d) to %s", args.k, out_path)

    # Keep tokenizer_config.yaml in sync. Edited via targeted regex substitution
    # (not a full yaml.safe_load/safe_dump round-trip) because PyYAML's dumper
    # drops comments, and this file is hand-documented.
    config_path = Path("configs/tokenizer_config.yaml")
    if config_path.exists():
        text = config_path.read_text()
        text, n1 = re.subn(r"(?m)^(\s*selected_k:\s*)\d+", rf"\g<1>{args.k}", text, count=1)
        text, n2 = re.subn(
            r"(?m)^(\s*kmeans_model_path:\s*)\S+", rf"\g<1>{out_path}", text, count=1
        )
        if n1 and n2:
            config_path.write_text(text)
            log.info("Updated %s with selected_k=%d", config_path, args.k)
        else:
            log.warning(
                "Could not find selected_k/kmeans_model_path lines in %s; "
                "leaving config unchanged. Update it manually.", config_path,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="extract syllable-mean-pooled embeddings")
    p_extract.add_argument("--manifest", required=True)
    p_extract.add_argument("--checkpoint", default="sylber")
    p_extract.add_argument("--out", default="data/embeddings/khmer_syllable_embeddings.npy")
    p_extract.set_defaults(func=cmd_extract)

    p_sweep = sub.add_parser("sweep", help="sweep K for k-means and report cluster-balance metrics")
    p_sweep.add_argument("--embeddings", required=True)
    p_sweep.add_argument("--k-sweep", type=int, nargs="+", default=[5000, 10000, 20000, 40000])
    p_sweep.add_argument("--out", default="results/kmeans_sweep.csv")
    p_sweep.set_defaults(func=cmd_sweep)

    p_fit = sub.add_parser("fit", help="fit and save the final k-means vocabulary")
    p_fit.add_argument("--embeddings", required=True)
    p_fit.add_argument("--k", type=int, default=10000)
    p_fit.add_argument("--out", default="models/khmer_kmeans_10k.pkl")
    p_fit.add_argument("--full-kmeans", action="store_true", help="use exact KMeans instead of MiniBatchKMeans")
    p_fit.set_defaults(func=cmd_fit)

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.func(args)
