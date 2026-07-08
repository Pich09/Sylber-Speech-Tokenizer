# Sylber-for-Khmer

Syllabic speech tokenization for Khmer, following the roadmap in
[`docs/audio-tokenizer-comparison.md`](docs/audio-tokenizer-comparison.md):
a frozen/fine-tuned [Sylber](https://github.com/Berkeley-Speech-Group/sylber)
encoder (4-5 Hz syllable tokens) feeding an OPT-125M decoder for Khmer
spoken language modeling.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`sylber` is installed straight from GitHub (see `requirements.txt`) since it
isn't on PyPI. GPU with CUDA is strongly recommended for Steps 3/5/6.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Data | `data/preprocessing/prepare_data.py` | Download DDD-Cambodia corpus from HF, resample to 16kHz, VAD-trim, split train/val/test, write manifest CSV |
| 2. Zero-shot eval | `src/segmentation.py eval` | Run pretrained Sylber on a sample of Khmer audio, report token rate + save results |
| 3. Fine-tune | `src/segmentation.py finetune` | Fine-tune the boundary head (`head_only`) or full backbone (`full_model`) on pseudo/corrected Khmer boundaries |
| 4. Discretize | `src/discretization.py extract / sweep / fit` | Extract syllable-mean-pooled embeddings, sweep K, fit final k-means vocabulary |
| — | `src/tokenizer.py encode` | End-to-end: raw wav -> discrete token IDs, via the fitted encoder + k-means |
| 5/6. SLM | `src/train_slm.py encode / train` | Encode a manifest to token sequences, continue-pretrain OPT-125M on them, report perplexity |
| — | `src/encoders.py`, `src/compare_encoders.py` | HuBERT/Whisper baseline encoder adapters + a table comparing perplexity/token-rate across encoders (see "Benchmarking" below) |
| — | `src/publish_checkpoint.py` | Export a fine-tuned Sylber checkpoint locally and/or push it to the Hugging Face Hub for reuse (e.g. in an ASR fine-tune) |

Example end-to-end run:

```bash
python data/preprocessing/prepare_data.py --dataset DDD-Cambodia/khmer-speech-dataset --out data/khmer_asr_cultural_v2

python src/segmentation.py eval --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
    --n-samples 100 --out results/zero_shot_evaluation.txt

# if zero-shot boundary agreement is <80% on manual inspection:
python src/segmentation.py finetune --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
    --boundaries data/annotations/corrected_boundaries.json --mode head_only \
    --out-ckpt models/sylber_checkpoints/sylber_khmer_v1.pth

python src/discretization.py extract --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
    --encoder sylber --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth --out data/embeddings/sylber
python src/discretization.py sweep --embeddings data/embeddings/sylber
python src/discretization.py fit --embeddings data/embeddings/sylber --k 10000 \
    --out models/khmer_kmeans_10k.pkl

python src/train_slm.py encode --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
    --encoder sylber --out data/tokens/sylber_tokens.jsonl
python src/train_slm.py train --tokens data/tokens/sylber_tokens.jsonl --encoder sylber
```

## Two paths after Step 2/3

Once boundary quality looks reasonable (Step 2, and Step 3 if needed), there
are two ways to proceed depending on how much confidence you want before
committing GPU time:

- **Compare first** — run the encoder benchmark below (Sylber vs. HuBERT vs.
  Whisper) on a small pilot subset of the manifest before deciding whether
  Sylber is worth fully fine-tuning on Khmer at all.
- **Fine-tune directly, skip comparison** — go straight to
  `segmentation.py finetune --mode full_model` on the full corpus, then use
  `src/publish_checkpoint.py` to export the result locally and/or push it to
  the Hub for reuse in a separate ASR fine-tune:

  ```bash
  python src/segmentation.py finetune --manifest <manifest.csv> \
      --boundaries data/annotations/corrected_boundaries.json --mode full_model \
      --out-ckpt models/sylber_checkpoints/sylber_khmer_v1.pth

  # local export only:
  python src/publish_checkpoint.py --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth \
      --local-dir models/hf_export/syllber-based-audio-encoder

  # export + push to the Hub (needs `huggingface-cli login` or HF_TOKEN first):
  python src/publish_checkpoint.py --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth \
      --repo-id Panhapich/syllber-based-audio-encoder --push
  ```

  `--push` creates the Hub repo as **private** by default (pass `--public`
  to change that) and uploads the checkpoint plus an auto-generated model
  card. Note the DDD-Cambodia training data is CC-BY-SA-4.0 — check
  share-alike license implications before making a derived checkpoint public.

## Benchmarking against other encoders

`src/encoders.py` adds HuBERT and Whisper (frame-level, no syllable
segmentation) as drop-in alternatives to Sylber for the same discretize ->
SLM pipeline, so they can be compared head-to-head on token rate and
perplexity (Step 5's "Baseline Comparison" in the doc). Repeat the
discretize/encode/train steps above once per encoder:

```bash
for enc in sylber hubert whisper; do
    python src/discretization.py extract --manifest <manifest.csv> --encoder $enc --out data/embeddings/$enc
    python src/discretization.py sweep --embeddings data/embeddings/$enc
    python src/discretization.py fit --embeddings data/embeddings/$enc --k 10000 --out models/${enc}_kmeans_10k.pkl

    python src/train_slm.py encode --manifest <manifest.csv> --encoder $enc --kmeans-path models/${enc}_kmeans_10k.pkl \
        --out data/tokens/${enc}_tokens.jsonl
    python src/train_slm.py train --tokens data/tokens/${enc}_tokens.jsonl --encoder $enc
done

python src/compare_encoders.py --encoders sylber hubert whisper
# -> results/downstream_eval/encoder_comparison.{csv,md}
```

HuBERT/Whisper default to `facebook/hubert-base-ls960` / `openai/whisper-base`
(see `configs/tokenizer_config.yaml: baseline_encoders`); override per-run
with `--checkpoint` on `extract`/`encode`. Note these two baselines have no
notion of syllable boundaries — their embeddings are raw frame-level
features (~50 Hz) fed through the same k-means step, so the comparison is
"syllable tokens vs. discretized frame tokens," matching how the doc's Step 5
baseline (HuBERT at 25-50 Hz) is meant to be read against Sylber's 4-5 Hz.

## Config

All paths/hyperparameters (k-means K, vocab size, SLM base model, splits) live in
[`configs/tokenizer_config.yaml`](configs/tokenizer_config.yaml). `discretization.py fit`
updates `selected_k` / `kmeans_model_path` in place after the sweep.

## Notes / open items

See "Next Steps & Open Questions" in the doc — code-switched (Khmer+English)
data, linguist boundary validation, and CTC/diffusion decoder heads are not
yet implemented; `src/segmentation.py`'s `finetune` command and
`src/tokenizer.py` are structured so a Khmer+English checkpoint can be
swapped in via `configs/tokenizer_config.yaml: segmenter.khmer_english_finetuned_checkpoint`
without other code changes.

The exact I/O contract of the pip-installed `sylber` package (`segment_features`
key, `.model` attribute, `extract_features` method) is assumed based on the
project's public description; if a newer/older `sylber` release exposes a
different API, adjust the marked spots in `src/segmentation.py` and
`src/discretization.py` (search for `segment_features`).
