"""Step 4: Discretization — extract syllable-averaged embeddings and build
a k-means token vocabulary.

`extract` streams embeddings to sharded .npy files under --out (a directory)
plus a meta.json, rather than holding the whole corpus (potentially tens of
GB) in RAM at once. `sweep`/`fit` read that directory and fit MiniBatchKMeans
via `partial_fit` over the shards, so K-means itself never needs the full
embedding set resident in memory either.

Usage:
    python src/discretization.py extract --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
        --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth --out data/embeddings/khmer_syllable_embeddings

    python src/discretization.py sweep --embeddings data/embeddings/khmer_syllable_embeddings \
        --k-sweep 5000 10000 20000 40000
    # (reads shard list + total duration from the meta.json written by `extract`)

    python src/discretization.py fit --embeddings data/embeddings/khmer_syllable_embeddings \
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


# Full-corpus scale (~11.8M syllables at 728h / 4.5Hz, per the roadmap's own
# estimate) means holding every embedding in RAM at once (as this used to,
# via a Python list + a single np.concatenate + a single np.save) risks
# OOM on a local machine (e.g. an RTX 4070 box, ~12GB VRAM + finite system
# RAM). Instead, stream embeddings to disk in shards as they're produced.
SHARD_SIZE = 200_000  # syllables per shard file


def iter_syllable_embeddings(manifest_path: str, checkpoint: str):
    """Run the (fine-tuned) segmenter over every utterance and yield
    (feats, duration_sec) per utterance without accumulating in RAM."""
    from segmentation import load_segmenter  # local import: src/ on path

    segmenter = load_segmenter(checkpoint)
    df = pd.read_csv(manifest_path)

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
        yield feats, row["duration_sec"]


def cmd_extract(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[str] = []
    buffer: list[np.ndarray] = []
    buffer_n = 0
    n_syllables = 0
    n_utterances = 0
    total_duration = 0.0
    embedding_dim: int | None = None

    def flush():
        nonlocal buffer, buffer_n
        if not buffer:
            return
        shard = np.concatenate(buffer, axis=0)
        shard_path = out_dir / f"shard_{len(shard_paths):05d}.npy"
        np.save(shard_path, shard)
        shard_paths.append(str(shard_path))
        buffer = []
        buffer_n = 0

    for feats, duration in iter_syllable_embeddings(args.manifest, args.checkpoint):
        if embedding_dim is None and feats.shape[0] > 0:
            embedding_dim = int(feats.shape[1])
        buffer.append(feats)
        buffer_n += len(feats)
        n_syllables += len(feats)
        n_utterances += 1
        total_duration += duration
        if buffer_n >= SHARD_SIZE:
            flush()
    flush()

    meta_path = out_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "shards": shard_paths,
                "n_syllables": n_syllables,
                "embedding_dim": embedding_dim,
                "n_utterances": n_utterances,
                "total_duration_sec": total_duration,
                "mean_token_rate_hz": n_syllables / total_duration if total_duration else 0.0,
            },
            indent=2,
        )
    )
    log.info(
        "Saved %d syllables across %d shards -> %s (meta: %s)",
        n_syllables, len(shard_paths), out_dir, meta_path,
    )


def load_meta(embeddings_arg: str) -> dict:
    """`embeddings_arg` is the --out directory used by `extract`."""
    meta_path = Path(embeddings_arg) / "meta.json"
    return json.loads(meta_path.read_text())


def iter_shards(shard_paths: list[str]):
    for p in shard_paths:
        yield np.load(p)


def _fit_kmeans_streaming(shard_paths: list[str], k: int, random_state: int = 42, batch_size: int = 10000, n_passes: int = 3):
    """Fit MiniBatchKMeans via `partial_fit` over on-disk shards, so the full
    embedding set (potentially tens of GB at full-corpus scale) never has to
    be resident in RAM at once — only one shard, sub-chunked to `batch_size`."""
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=batch_size, n_init=3)
    for p in range(n_passes):
        for shard in iter_shards(shard_paths):
            for i in range(0, len(shard), batch_size):
                km.partial_fit(shard[i : i + batch_size])
        log.info("K=%d: completed pass %d/%d over shards", k, p + 1, n_passes)
    return km


def _kmeans_stream_stats(km, shard_paths: list[str], k: int) -> tuple[float, np.ndarray]:
    """Stream shards through a fitted k-means model to compute inertia and
    per-cluster counts without materializing an (n, k) distance matrix."""
    inertia = 0.0
    counts = np.zeros(k, dtype=np.int64)
    for shard in iter_shards(shard_paths):
        labels = km.predict(shard)
        centers = km.cluster_centers_[labels]
        inertia += float(((shard - centers) ** 2).sum())
        counts += np.bincount(labels, minlength=k)
    return inertia, counts


def sweep_k(shard_paths: list[str], k_values: list[int], total_duration_sec: float, n_syllables: int, random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_values:
        log.info("Fitting k-means with K=%d ...", k)
        km = _fit_kmeans_streaming(shard_paths, k, random_state=random_state)
        inertia, counts = _kmeans_stream_stats(km, shard_paths, k)

        token_rate_hz = n_syllables / total_duration_sec if total_duration_sec else float("nan")
        top10 = np.sort(counts)[::-1][:10]
        balance = float(top10.sum() / counts.sum())
        rows.append(
            {
                "k": k,
                "token_rate_hz": token_rate_hz,
                "top10_cluster_share": balance,
                "inertia": inertia,
            }
        )
        log.info("K=%d: token_rate=%.2fHz top10_share=%.1f%% inertia=%.1f", k, token_rate_hz, balance * 100, inertia)

    return pd.DataFrame(rows)


def cmd_sweep(args):
    meta = load_meta(args.embeddings)
    df = sweep_k(meta["shards"], args.k_sweep, meta["total_duration_sec"], meta["n_syllables"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Sweep results:\n%s", df.to_string(index=False))
    log.info("Wrote sweep results to %s", out_path)
    log.info(
        "Sanity check: expect token_rate_hz in [4,5]; flag any K with top10_cluster_share > 0.5 "
        "(over-clustering / data quality issue)."
    )


def fit_final_kmeans(shard_paths: list[str], k: int, random_state: int = 42, minibatch: bool = True, batch_size: int = 10000, n_passes: int = 3):
    from sklearn.cluster import KMeans

    if minibatch:
        return _fit_kmeans_streaming(shard_paths, k, random_state=random_state, batch_size=batch_size, n_passes=n_passes)

    # Exact KMeans needs everything in RAM; only viable for small embedding sets.
    embeddings = np.concatenate(list(iter_shards(shard_paths)), axis=0)
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(embeddings)
    return km


def cmd_fit(args):
    meta = load_meta(args.embeddings)
    km = fit_final_kmeans(meta["shards"], args.k, minibatch=not args.full_kmeans)

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

    p_extract = sub.add_parser("extract", help="extract syllable-mean-pooled embeddings (streamed to sharded .npy files)")
    p_extract.add_argument("--manifest", required=True)
    p_extract.add_argument("--checkpoint", default="sylber")
    p_extract.add_argument("--out", default="data/embeddings/khmer_syllable_embeddings", help="output directory for shards + meta.json")
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
