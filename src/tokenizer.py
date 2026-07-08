"""End-to-end Khmer syllable tokenizer: raw audio -> discrete token ID sequence.

Chains the (fine-tuned) Sylber segmenter with the fitted k-means vocabulary,
per the Stage A pipeline in docs/audio-tokenizer-comparison.md.

Usage:
    python src/tokenizer.py encode --wav path/to/audio.wav
    python src/tokenizer.py encode --wav path/to/audio.wav --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class KhmerSyllableTokenizer:
    """Frozen Sylber encoder + k-means vocabulary -> token IDs.

    Special token IDs (pad/bos/eos) are computed from the fitted k-means
    model's actual `n_clusters` (see special_tokens.py) rather than trusted
    from config, so they stay correct even if `selected_k` in
    configs/tokenizer_config.yaml drifts out of sync with the k-means model
    actually pointed to by `kmeans_path`.
    """

    def __init__(self, segmenter_checkpoint: str, kmeans_path: str):
        from segmentation import load_segmenter
        from special_tokens import special_token_ids

        self.segmenter = load_segmenter(segmenter_checkpoint)
        self.kmeans = joblib.load(kmeans_path)
        self.vocab_size = self.kmeans.n_clusters
        self.special_tokens = special_token_ids(self.vocab_size)

    @classmethod
    def from_config(cls, config_path: str = "configs/tokenizer_config.yaml", use_khmer_finetuned: bool = True):
        cfg = yaml.safe_load(Path(config_path).read_text())
        seg_cfg = cfg["segmenter"]
        checkpoint = (
            seg_cfg["khmer_finetuned_checkpoint"] if use_khmer_finetuned else seg_cfg["base_checkpoint"]
        )
        if use_khmer_finetuned and not Path(checkpoint).exists():
            log.warning("Khmer-finetuned checkpoint %s not found; falling back to base Sylber.", checkpoint)
            checkpoint = seg_cfg["base_checkpoint"]
        return cls(
            segmenter_checkpoint=checkpoint,
            kmeans_path=cfg["discretization"]["kmeans_model_path"],
        )

    def encode(self, wav_path: str, add_bos_eos: bool = False) -> dict:
        """Return {"token_ids": [...], "segments": [(start, end), ...]}."""
        out = self.segmenter(wav_path, in_second=True)
        if isinstance(out, dict) and "segment_features" in out:
            feats = np.asarray(out["segment_features"])
            segments = out.get("segments", [])
        else:
            raise RuntimeError(
                "Installed sylber version does not expose 'segment_features'; "
                "see src/discretization.py:extract_syllable_embeddings for the manual mean-pool fallback."
            )

        if feats.shape[0] == 0:
            token_ids: list[int] = []
        else:
            token_ids = self.kmeans.predict(feats).tolist()

        if add_bos_eos and "bos" in self.special_tokens and "eos" in self.special_tokens:
            token_ids = [self.special_tokens["bos"]] + token_ids + [self.special_tokens["eos"]]

        return {"token_ids": token_ids, "segments": list(segments)}

    def encode_batch(self, wav_paths: list[str], **kwargs) -> list[dict]:
        return [self.encode(p, **kwargs) for p in wav_paths]


def cmd_encode(args):
    tok = KhmerSyllableTokenizer.from_config(args.config, use_khmer_finetuned=not args.base_checkpoint)
    result = tok.encode(args.wav, add_bos_eos=args.add_bos_eos)
    log.info(
        "Encoded %s -> %d tokens (%.2f Hz)",
        args.wav,
        len(result["token_ids"]),
        len(result["token_ids"]) / max(result["segments"][-1][1] if result["segments"] else 1e-9, 1e-9),
    )
    print(json.dumps(result, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_encode = sub.add_parser("encode", help="encode a wav file into a discrete token sequence")
    p_encode.add_argument("--wav", required=True)
    p_encode.add_argument("--config", default="configs/tokenizer_config.yaml")
    p_encode.add_argument("--base-checkpoint", action="store_true", help="use base pretrained Sylber instead of Khmer fine-tune")
    p_encode.add_argument("--add-bos-eos", action="store_true")
    p_encode.set_defaults(func=cmd_encode)

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.func(args)
