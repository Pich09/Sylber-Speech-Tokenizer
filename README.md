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
    --checkpoint models/sylber_checkpoints/sylber_khmer_v1.pth --out data/embeddings/khmer_syllable_embeddings.npy
python src/discretization.py sweep --embeddings data/embeddings/khmer_syllable_embeddings.npy
python src/discretization.py fit --embeddings data/embeddings/khmer_syllable_embeddings.npy --k 10000 \
    --out models/khmer_kmeans_10k.pkl

python src/train_slm.py encode --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest.csv \
    --out data/tokens/khmer_tokens.jsonl
python src/train_slm.py train --tokens data/tokens/khmer_tokens.jsonl
```

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
