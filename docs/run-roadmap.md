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
   python src/segmentation.py finetune --manifest <manifest.csv> --boundaries data/annotations/corrected_boundaries.json --mode last_layer --out-ckpt models/sylber_checkpoints/sylber_khmer_v1.pth
   ```
   Confirm: loss should trend down across epochs in the log; compare boundary agreement on a
   held-out set against the zero-shot baseline.

4. **Decision point — pick a path.** Once Step 2/3 boundary quality looks reasonable, choose
   between two ways to spend GPU time, and note this is a *different* choice from step 3's
   zero-shot-quality decision — step 3 only fixes *boundary detection*; the choice below is
   about whether to validate the encoder choice before committing further, or commit directly:

   - **Path A — compare Sylber against other encoders first** (cheaper, answers "is Sylber even
     worth it for Khmer" before spending days fine-tuning it). Recommended to run on a small
     manifest subset first (e.g. filter the manifest CSV to ~5-20h of audio) before the full
     corpus — this whole path exists to be a cheap screening step, not a final result.
   - **Path B — fine-tune Sylber directly, skip comparison** (go straight to the goal of a
     reusable fine-tuned Khmer encoder, at the cost of not knowing whether a different encoder
     would've done better).

   Both are documented with full command sequences in the README's
   ["Two paths after Step 2/3"](../README.md#two-paths-after-step-23) and
   ["Benchmarking against other encoders"](../README.md#benchmarking-against-other-encoders)
   sections — this roadmap just calls out the GPU-machine-specific notes for each. **If you've
   already run Path A on a small subset and its comparison looks ambiguous** (e.g. Sylber's
   perplexity far worse than the other encoders), don't decide Path A vs. Path B from that number
   alone — jump to [`docs/post-benchmark-roadmap.md`](post-benchmark-roadmap.md) first, which
   covers why a small-subset discrete-SLM comparison needs much more data to be trustworthy and
   lays out two follow-up options (a CTC probe vs. scaling this comparison up) before committing
   to either path.

   **Path A (benchmark) — discretization + SLM training per encoder:**
   ```bash
   for enc in sylber hubert whisper; do
       python src/discretization.py extract --manifest <manifest.csv> --encoder $enc --out data/embeddings/$enc
       python src/discretization.py sweep --embeddings data/embeddings/$enc
       python src/discretization.py fit --embeddings data/embeddings/$enc --k 10000 --out models/${enc}_kmeans_10k.pkl

       python src/train_slm.py encode --manifest <manifest.csv> --encoder $enc \
           --kmeans-path models/${enc}_kmeans_10k.pkl --out data/tokens/${enc}_tokens.jsonl
       python src/train_slm.py train --tokens data/tokens/${enc}_tokens.jsonl --encoder $enc
   done
   python src/compare_encoders.py --encoders sylber hubert whisper
   ```
   `--out`/`--embeddings` is a directory: `extract` streams embeddings to sharded
   `shard_*.npy` files + `meta.json` under it instead of one giant in-memory array, since the
   full corpus (~11.8M syllables) would otherwise risk OOM on a local RTX 4070 box. `sweep`/`fit`
   fit MiniBatchKMeans via `partial_fit` over those shards. If any encoder's k-means sweep picks
   a K other than 10000, pass `train_slm.py train --vocab-size <that K>` — special (pad/bos/eos)
   token IDs are derived from the actual K, not a fixed constant, so this must match.

   Confirm: `sweep`'s printed table should show ~4-5Hz token rate for sylber (HuBERT/Whisper will
   read much higher, ~50Hz — expected, they have no syllable segmentation) and no K with
   `top10_cluster_share > 0.5`; `fit` updates `selected_k`/`kmeans_model_path` in
   `configs/tokenizer_config.yaml` in place (sylber run only — the other encoders' k-means paths
   are tracked by their own `--out` filenames, not the shared config); `compare_encoders.py`
   writes `results/downstream_eval/encoder_comparison.{csv,md}`.

   **If Path A's perplexity comparison comes back ambiguous** (e.g. Sylber
   looking dramatically worse on a small pilot subset), don't treat that as
   final — see `docs/post-benchmark-roadmap.md` for why a small-subset
   discrete-SLM comparison needs far more data than a CTC-based check to be
   trustworthy, and for the two follow-up options (CTC probe vs. scaling
   this step up) before deciding whether to move to Path B.

   **Path B (direct fine-tune) — full-model fine-tune + publish:**
   ```bash
   python src/segmentation.py finetune --manifest <manifest.csv> \
       --boundaries data/annotations/corrected_boundaries.json --mode full_model \
       --out-ckpt models/sylber_checkpoints/sylber_khmer_v1.pth

   python src/publish_checkpoint.py --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth \
       --repo-id Panhapich/syllber-based-audio-encoder --push
   ```
   `--mode full_model` unfreezes the whole backbone (not just its last transformer layer, as
   `--mode last_layer` does), which is what makes this different from step 3's conditional
   fine-tune even if step 3 was skipped because zero-shot passed — step 3 is about fixing bad
   boundaries, this is about deliberately adapting the encoder to Khmer acoustics for downstream
   reuse. See the README's "Fine-tuning caveat" section: Sylber has no separate learned boundary
   head, so both modes fine-tune the backbone itself against a boundary-contrastive loss, not a
   classifier. `--push` needs `huggingface-cli login` or
   an `HF_TOKEN` env var set on the GPU machine first; it creates the Hub repo **private** by
   default (`--public` to change that). Note DDD-Cambodia is CC-BY-SA-4.0 — check share-alike
   license implications before making a derived checkpoint public.

   Confirm: fine-tune loss trending down in the epoch logs; `publish_checkpoint.py` logs the
   local export dir and, if `--push`, the resulting `https://huggingface.co/<repo-id>` URL.

5. **SLM training on the chosen/final encoder (Steps 5/6, if not already covered by Path A above)**:
   ```bash
   python src/train_slm.py encode --manifest <manifest.csv> --encoder sylber --out data/tokens/sylber_tokens.jsonl
   python src/train_slm.py train --tokens data/tokens/sylber_tokens.jsonl --encoder sylber
   ```
   Confirm: `results/downstream_eval/slm_eval_sylber.json` gets written with `eval_loss` and
   `perplexity` after training completes.

## Summary: what to change vs. leave alone

| Area | Change needed? |
|---|---|
| Code (`src/*.py`, `configs/*.yaml`) | No changes, as long as `sylber`'s output keys match the `segment_features` assumption (verify with the one-line check in step 2) |
| `requirements.txt` torch line | Install a CUDA-matched torch build *before* `pip install -r requirements.txt` on the GPU machine |
| Paths in config | None — all relative to repo root, already correct |
| Manual work | Boundary annotation file for Step 3 (only if zero-shot eval fails the 80% bar); `huggingface-cli login`/`HF_TOKEN` on the GPU machine if using Path B's `--push` |
| Path choice | Decide Path A (compare) vs Path B (fine-tune direct) at step 4 — not code, a call to make based on how much confidence you want before committing GPU time |
| Ambiguous Path A result | See `docs/post-benchmark-roadmap.md` — CTC probe vs. scaled-up SLM comparison, before committing to Path A's verdict or moving to Path B |

## Verification

Each step above has its own confirm/check listed inline (log output, file existence, printed
metrics). There's no single end-to-end test — verify progressively at each step before moving
to the next, since later steps (discretization, SLM training, Path A's benchmark, Path B's
fine-tune) are expensive and depend on the outputs of earlier ones being correct.
