# Roadmap: Running the Sylber-for-Khmer Pipeline on a GPU Machine

## Context

The pipeline (`docs/audio-tokenizer-comparison.md` + `src/*.py`) was built and pushed to
`https://github.com/Pich09/Sylber-Speech-Tokenizer` on a GPU-less Windows desktop.
Sylber fine-tuning, full-corpus discretization, and OPT-125M SLM training all need real GPU
compute, so the plan is: clone the repo onto the GPU machine and run everything there. The
Khmer dataset (`DDD-Cambodia/khmer-speech-dataset`) is pulled directly from Hugging Face — no
separate access request needed. This document is a running-order roadmap, not a code change —
almost nothing needs editing, and the few things that do are called out explicitly below.

## Transfer: how to get the code onto the GPU machine

Use the GitHub repo you already pushed to, rather than copying files by hand:

```bash
git clone https://github.com/Pich09/Sylber-Speech-Tokenizer.git
cd Sylber-Speech-Tokenizer
```

Nothing else needs to move — `data/`, `models/`, `results/` artifacts are gitignored on purpose
(see `.gitignore`) because they're either huge (the 495GB corpus) or fully regenerable from the
scripts. They get created fresh on the GPU machine as each step runs.

## Setup on the GPU machine (one change needed here)

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
```

**Change needed:** `requirements.txt` pins `torch>=2.1.0` from generic PyPI, which on some
systems resolves to a CPU-only wheel. Before `pip install -r requirements.txt`, install the
CUDA build matching that machine's driver first, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match to actual CUDA version
pip install -r requirements.txt
```

Then verify: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

**No other setup changes** — `sylber` still installs from the GitHub pin in requirements.txt,
and all script paths in `configs/tokenizer_config.yaml` are relative to the repo root, so they
work unmodified as long as you always run commands from the repo root (matches the README).

## Execution order (what to run, in what sequence)

All commands below assume `cwd` = repo root, venv activated.

1. **Data prep (Step 1)** — download + preprocess directly on the GPU machine (skip
   downloading on the desktop; the corpus is ~495GB, no reason to move it twice):
   ```bash
   python data/preprocessing/prepare_data.py --dataset DDD-Cambodia/khmer-speech-dataset --out data/khmer_asr_cultural_v2
   ```
   Confirm: check the printed split sizes and total hours in the log output, and spot-check a
   couple of `.wav` files in `data/khmer_asr_cultural_v2/wav16k/`.

2. **Zero-shot segmentation eval (Step 2)**:
   ```bash
   python src/segmentation.py eval --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv --n-samples 100 --out results/zero_shot_evaluation.txt
   ```
   **Before running this at scale**, sanity-check the `sylber` package's actual output shape on
   one file — `src/segmentation.py` and `src/discretization.py` assume the installed package
   returns a dict with a `"segment_features"` key (flagged as an assumption in the README).
   Quick check:
   ```bash
   python -c "from sylber import Segmenter; s = Segmenter(model_ckpt='sylber'); print(s('data/khmer_asr_cultural_v2/wav16k/<one_file>.wav', in_second=True).keys())"
   ```
   If the keys differ, that's the one real code change likely needed — adjust the marked spots
   in `segmentation.py`/`discretization.py`/`tokenizer.py` (search for `segment_features`).

   **Decision point** (per the doc): read `results/zero_shot_evaluation.txt`. If manual
   inspection of boundaries is ≥80% accurate, skip straight to step 4. Otherwise do step 3.

3. **Fine-tuning (Step 3, conditional)** — only if zero-shot fell short. This needs a human
   task first: hand-correct a small subset (~500-1000 utterances) of zero-shot segmentations
   into a `boundaries.json` (`{wav_path: [[start,end], ...]}`) — there's no script for this
   annotation step, it's manual linguistic work. Then:
   ```bash
   python src/segmentation.py finetune --manifest <manifest.csv> --boundaries data/annotations/corrected_boundaries.json --mode head_only --out-ckpt models/sylber_checkpoints/sylber_khmer_v1.pth
   ```
   Confirm: loss should trend down across epochs in the log; compare boundary agreement on a
   held-out set against the zero-shot baseline.

4. **Discretization (Step 4)**:
   ```bash
   python src/discretization.py extract --manifest <manifest.csv> --checkpoint <ckpt or "sylber"> --out data/embeddings/khmer_syllable_embeddings.npy
   python src/discretization.py sweep --embeddings data/embeddings/khmer_syllable_embeddings.npy
   python src/discretization.py fit --embeddings data/embeddings/khmer_syllable_embeddings.npy --k 10000 --out models/khmer_kmeans_10k.pkl
   ```
   Confirm: `sweep`'s printed table should show ~4-5Hz token rate and no K with
   `top10_cluster_share > 0.5`; `fit` updates `selected_k`/`kmeans_model_path` in
   `configs/tokenizer_config.yaml` automatically (comments preserved).

5. **SLM training (Steps 5/6)**:
   ```bash
   python src/train_slm.py encode --manifest <manifest.csv> --out data/tokens/khmer_tokens.jsonl
   python src/train_slm.py train --tokens data/tokens/khmer_tokens.jsonl
   ```
   Confirm: `results/downstream_eval/slm_eval.json` gets written with `eval_loss` and
   `perplexity` after training completes.

## Summary: what to change vs. leave alone

| Area | Change needed? |
|---|---|
| Code (`src/*.py`, `configs/*.yaml`) | No changes, as long as `sylber`'s output keys match the `segment_features` assumption (verify with the one-line check in step 2) |
| `requirements.txt` torch line | Install a CUDA-matched torch build *before* `pip install -r requirements.txt` on the GPU machine |
| Paths in config | None — all relative to repo root, already correct |
| Manual work | Boundary annotation file for Step 3 (only if zero-shot eval fails the 80% bar) — this is a human task, not code |

## Verification

Each step above has its own confirm/check listed inline (log output, file existence, printed
metrics). There's no single end-to-end test — verify progressively at each step before moving
to the next, since later steps (discretization, SLM training) are expensive and depend on the
outputs of earlier ones being correct.
