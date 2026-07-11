"""Package a fine-tuned Sylber checkpoint (from `segmentation.py finetune`)
for reuse, and optionally push it to the Hugging Face Hub.

This is the "fine-tune directly, skip the encoder benchmark" path: run
`segmentation.py finetune --mode full_model`, then use this script to (a)
export a clean local directory (checkpoint + model card) that other code —
e.g. an ASR fine-tune — can load from, and (b) optionally push that
directory to a Hub repo.

Usage:
    # local export only
    python src/publish_checkpoint.py --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth \
        --local-dir models/hf_export/syllber-based-audio-encoder

    # export + push to the Hub (requires `huggingface-cli login` or HF_TOKEN)
    python src/publish_checkpoint.py --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth \
        --repo-id Panhapich/syllber-based-audio-encoder --push
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """\
---
language: km
tags:
- speech
- sylber
- syllable-tokenization
- khmer
---

# {repo_name}

Sylber encoder ([Berkeley-Speech-Group/sylber](https://github.com/Berkeley-Speech-Group/sylber),
base checkpoint `{init_ckpt}`) fine-tuned on Khmer speech.

- **Fine-tune mode**: `{mode}` ({mode_desc})
- **Files**: `sylber_khmer.pth` — a dict with `backbone_state_dict`, `mode`,
  `init_ckpt`, as written by `src/segmentation.py`'s `finetune_segmenter`.
  This is Sylber's backbone (`sylber.Segmenter`'s `.speech_model`, a
  HubertModel) fine-tuned with a boundary-contrastive loss — Sylber has no
  separate learned boundary head, so there's nothing else to load. Load via
  `src/segmentation.py`'s `load_segmenter(<this checkpoint's path>)`, which
  handles restoring `backbone_state_dict` into a fresh `Segmenter` built
  from `init_ckpt` — see that module in
  [Sylber-Speech-Tokenizer](https://github.com/Pich09/Sylber-Speech-Tokenizer)
  for the loading convention this checkpoint follows.

## Intended use

Frozen or further fine-tuned speech encoder for Khmer downstream tasks
(ASR via a CTC/attention decoder head, spoken language modeling via
k-means discretization, etc.) — see the "Encoder Reusability" section of
that repo's `docs/audio-tokenizer-comparison.md`.

## Training data

Fine-tuned on Khmer speech from the DDD-Cambodia corpora
(CC-BY-SA-4.0). **License note**: check CC-BY-SA-4.0 share-alike
compatibility for your intended use of this derived checkpoint before
redistributing.

## Caveats

- No manual linguistic validation of segmentation boundaries is recorded
  for this checkpoint unless noted otherwise by whoever ran the fine-tune
  — verify boundary quality (see Step 2/3 of the source repo's roadmap)
  before relying on this for anything beyond experimentation.
"""

MODE_DESCRIPTIONS = {
    "last_layer": "backbone frozen except the last transformer layer",
    "full_model": "entire backbone fine-tuned",
}


def export_checkpoint(checkpoint_path: str, local_dir: str, repo_id: str | None) -> Path:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "backbone_state_dict" not in ckpt:
        raise ValueError(
            f"{checkpoint_path} doesn't look like a segmentation.py finetune_segmenter checkpoint "
            "(expected a dict with 'backbone_state_dict')."
        )

    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ckpt_path = out_dir / "sylber_khmer.pth"
    torch.save(ckpt, out_ckpt_path)

    mode = ckpt.get("mode", "unknown")
    repo_name = repo_id.split("/")[-1] if repo_id else out_dir.name
    (out_dir / "README.md").write_text(
        MODEL_CARD_TEMPLATE.format(
            repo_name=repo_name,
            init_ckpt=ckpt.get("init_ckpt", "sylber"),
            mode=mode,
            mode_desc=MODE_DESCRIPTIONS.get(mode, "see finetune_segmenter's --mode"),
        )
    )
    (out_dir / "config.json").write_text(
        json.dumps({"mode": mode, "init_ckpt": ckpt.get("init_ckpt", "sylber"), "source": "sylber"}, indent=2)
    )

    log.info("Exported checkpoint + model card to %s", out_dir)
    return out_dir


def push_to_hub(local_dir: Path, repo_id: str, private: bool = True):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")
    log.info("Pushed %s -> https://huggingface.co/%s", local_dir, repo_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a segmentation.py finetune_segmenter --out-ckpt file")
    parser.add_argument("--local-dir", default=None, help="default: models/hf_export/<repo-id's last segment, or 'sylber-khmer'>")
    parser.add_argument("--repo-id", default=None, help="e.g. Panhapich/syllber-based-audio-encoder; required if --push")
    parser.add_argument("--push", action="store_true", help="also push --local-dir to the Hub (needs prior `huggingface-cli login` or HF_TOKEN env var)")
    parser.add_argument("--public", action="store_true", help="create the Hub repo as public (default: private)")
    args = parser.parse_args()

    if args.push and not args.repo_id:
        parser.error("--push requires --repo-id")

    local_dir = args.local_dir or f"models/hf_export/{(args.repo_id.split('/')[-1] if args.repo_id else 'sylber-khmer')}"
    out_dir = export_checkpoint(args.checkpoint, local_dir, args.repo_id)

    if args.push:
        push_to_hub(out_dir, args.repo_id, private=not args.public)
    else:
        log.info("Local export only (pass --push --repo-id %s to publish to the Hub)", args.repo_id or "<repo-id>")


if __name__ == "__main__":
    main()
