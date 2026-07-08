"""Steps 5/6: Train a Khmer Spoken Language Model on Sylber token sequences.

Reuses a pretrained OPT-125M/Qwen2.5-0.5B checkpoint but replaces its
text vocabulary/embedding + LM head with one sized to the speech-token
vocabulary (k-means K + special tokens), then continues pretraining via
next-token prediction over token ID sequences (no text supervision).

Usage:
    # 1) encode a manifest's audio into token sequences
    python src/train_slm.py encode --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
        --out data/tokens/khmer_tokens.jsonl

    # 2) train
    python src/train_slm.py train --tokens data/tokens/khmer_tokens.jsonl \
        --config configs/tokenizer_config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def cmd_encode(args):
    from tokenizer import KhmerSyllableTokenizer

    tok = KhmerSyllableTokenizer.from_config(args.config, use_khmer_finetuned=not args.base_checkpoint)
    df = pd.read_csv(args.manifest)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_duration = 0.0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="encoding"):
            result = tok.encode(row["path"], add_bos_eos=True)
            f.write(json.dumps({"path": row["path"], "split": row.get("split", "train"), "token_ids": result["token_ids"]}) + "\n")
            total_tokens += len(result["token_ids"])
            total_duration += row["duration_sec"]

    log.info(
        "Encoded %d utterances -> %d tokens (%.2f Hz mean rate) -> %s",
        len(df), total_tokens, total_tokens / total_duration if total_duration else 0.0, out_path,
    )


class TokenSequenceDataset(Dataset):
    """Packs concatenated token-ID sequences into fixed-length blocks for
    causal LM training (standard `run_clm.py`-style block packing)."""

    def __init__(self, jsonl_path: str, split: str, block_size: int, pad_id: int):
        sequences = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("split", "train") == split:
                    sequences.extend(rec["token_ids"])

        n_blocks = max(len(sequences) // block_size, 0)
        usable = n_blocks * block_size
        if usable == 0:
            self.blocks = np.zeros((0, block_size), dtype=np.int64)
        else:
            arr = np.asarray(sequences[:usable], dtype=np.int64)
            self.blocks = arr.reshape(n_blocks, block_size)
        self.pad_id = pad_id

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = torch.from_numpy(self.blocks[idx].copy())
        return {"input_ids": ids, "labels": ids.clone()}


def build_speech_lm(base_model: str, vocab_size: int, max_seq_len: int) -> tuple["torch.nn.Module", int]:
    """Load a pretrained OPT/Qwen checkpoint and resize its embedding + LM
    head to the speech-token vocabulary, discarding the text tokenizer.
    The Transformer backbone weights (attention, MLP) are kept — this is
    the "leverages English text pretrain" transfer described in the doc.

    Returns (model, effective_max_seq_len) — the caller must use the
    returned length when packing blocks, since it may be clamped to the
    base model's max_position_embeddings.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model)
    config = model.config

    if max_seq_len > getattr(config, "max_position_embeddings", max_seq_len):
        log.warning(
            "Requested max_seq_len=%d exceeds base model's max_position_embeddings=%d; "
            "truncating to model limit.", max_seq_len, config.max_position_embeddings,
        )
        max_seq_len = config.max_position_embeddings

    model.resize_token_embeddings(vocab_size)
    return model, max_seq_len


def cmd_train(args):
    from transformers import Trainer, TrainingArguments

    cfg = yaml.safe_load(Path(args.config).read_text())
    slm_cfg = cfg["slm"]
    vocab_size = slm_cfg["vocab_size"] + len(slm_cfg["special_tokens"])
    pad_id = slm_cfg["special_tokens"]["pad"]

    model, block_size = build_speech_lm(slm_cfg["base_model"], vocab_size, slm_cfg["max_seq_len"])

    train_ds = TokenSequenceDataset(args.tokens, "train", block_size, pad_id)
    val_ds = TokenSequenceDataset(args.tokens, "val", block_size, pad_id)
    log.info("train blocks=%d val blocks=%d (block_size=%d)", len(train_ds), len(val_ds), block_size)

    training_args = TrainingArguments(
        output_dir=slm_cfg["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        weight_decay=0.01,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
    )
    trainer.train()
    trainer.save_model(slm_cfg["output_dir"])
    log.info("Saved trained SLM to %s", slm_cfg["output_dir"])

    metrics = trainer.evaluate()
    perplexity = float(np.exp(metrics["eval_loss"])) if "eval_loss" in metrics else None
    results_path = Path("results/downstream_eval")
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "slm_eval.json").write_text(json.dumps({**metrics, "perplexity": perplexity}, indent=2))
    log.info("Eval metrics: %s (perplexity=%s)", metrics, perplexity)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_encode = sub.add_parser("encode", help="encode a manifest's audio to a token-sequence JSONL")
    p_encode.add_argument("--manifest", required=True)
    p_encode.add_argument("--config", default="configs/tokenizer_config.yaml")
    p_encode.add_argument("--out", default="data/tokens/khmer_tokens.jsonl")
    p_encode.add_argument("--base-checkpoint", action="store_true")
    p_encode.set_defaults(func=cmd_encode)

    p_train = sub.add_parser("train", help="continued-pretrain the SLM on token sequences")
    p_train.add_argument("--tokens", required=True)
    p_train.add_argument("--config", default="configs/tokenizer_config.yaml")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=5e-5)
    p_train.set_defaults(func=cmd_train)

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.func(args)
