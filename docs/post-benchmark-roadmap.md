# Roadmap: Resolving the Sylber Benchmark Ambiguity

## Context

`docs/path-a-encoder-comparison.md` ran the discrete-SLM comparison (Sylber
vs. HuBERT vs. Whisper) on a 20-hour Khmer subset and got an ambiguous
result: Sylber's perplexity (1.31e25) was dramatically worse than HuBERT/
Whisper (~5-7e4), but that comparison method itself needs far more data
than 20h to mean anything — Sylber's own paper trains its discrete-token
uLM on a 1,000-hour "limited resource" floor (66,000 hours at full scale).
At 20h, the SLM path is data-starved for any encoder, and Sylber (4.76 Hz,
~10x fewer tokens per hour than the ~50 Hz frame-level baselines) hits that
floor first. So the 20h number doesn't actually tell us whether Sylber is a
good or bad fit for Khmer — it tells us the test was run below its own
minimum viable scale.

This roadmap lays out two ways to get an actual answer, since they resolve
the ambiguity differently and cost very different amounts of GPU time.
They are not mutually exclusive, but **Option A is the recommended
starting point** — see the recommendation at the end before picking.

## Option A — CTC probe (test the real ASR goal directly, cheap, no new data needed)

Your actual end goal (per earlier discussion) is fine-tuning Sylber for
Khmer **ASR**, not winning a discrete-token perplexity benchmark. Sylber
2.0's own paper validates a frozen/lightly-tuned encoder + small supervised
CTC decoder at exactly this data scale (Section 6.3, "Low-Resource ASR":
Korean/Bemba/Quecha, 20-50h each) — so `src/train_ctc.py` is a test that's
actually appropriate for the 20h subset you already have, unlike the SLM
comparison.

**Steps:**

```bash
python src/train_ctc.py train \
    --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest_subset20h.csv \
    --encoder sylber

python src/train_ctc.py train \
    --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest_subset20h.csv \
    --encoder hubert --checkpoint facebook/hubert-base-ls960

python src/train_ctc.py train \
    --manifest data/preprocessing/manifests/khmer-speech-dataset_manifest_subset20h.csv \
    --encoder whisper --checkpoint openai/whisper-base
```

Confirm: check `results/downstream_eval/ctc_probe_{sylber,hubert,whisper}.json`
for `final_val_cer` — lower is better. Also check `n_skipped` in the
history; a high skip count for Sylber (utterances where the ~4.76Hz output
was shorter than the transcript) means the 20h subset has a lot of very
short utterances and the CER comparison is being computed on a smaller
effective sample than the other two encoders — worth noting when reading
the result, not necessarily a blocker.

**Reading the result — this run still uses zero-shot Sylber** (Step 3 of
the original roadmap was skipped when the 20h benchmark ran), so a bad CTC
result doesn't necessarily mean Sylber can't work for Khmer — it may just
mean the *zero-shot* English-pretrained backbone is a poor starting point.
Decision tree:

- **Sylber's CER is competitive with or better than HuBERT/Whisper** → good
  signal. Proceed to Path B (`segmentation.py finetune --mode full_model` +
  `publish_checkpoint.py`) with real confidence, since this is now validated
  on your actual downstream task, not a proxy.
- **Sylber's CER is clearly worse** → don't treat that as final either,
  since it's still zero-shot. Before writing Sylber off:
  1. Fine-tune the backbone first (`segmentation.py finetune --mode
     full_model`) on the 20h (or full) corpus, then re-run the CTC probe —
     a fair comparison needs Sylber to have actually seen Khmer, same as
     HuBERT/Whisper aren't Khmer-adapted either but at least weren't
     penalized by an unadapted segmentation step on top.
  2. Watch for a Sylber 2.0 checkpoint release from
     `Berkeley-Speech-Group/sylber` — confirmed as of now the repo only
     ships the v1 English-only checkpoint; Sylber 2.0's multilingual one
     (explicitly trained across 102 languages) is the version the paper
     shows winning at low-resource ASR, and isn't published there yet.
  3. Hybrid fallback: keep Sylber only for syllable *boundary detection*,
     but pool a stronger multilingual acoustic encoder's (mHuBERT, XLS-R)
     frame features within those boundaries instead of Sylber's own
     embeddings.
  4. Last resort: fall back to a conventional multilingual ASR encoder
     (mHuBERT/XLS-R/Whisper multilingual) fine-tuned directly with CTC —
     proven approach, at the cost of losing Sylber's ~10x token-efficiency
     advantage for a future SLM decoder.

**Cost**: reuses the existing 20h subset and encoder checkpoints already
used for Path A — no new data, no new downloads. This is a few hours of
GPU time per encoder, not days.

## Option B — Scale audio to match token count (fix the SLM comparison itself)

If you still want the discrete-SLM/perplexity comparison to be a
meaningful number (e.g. because the end architecture really is meant to be
Sylber → k-means → OPT-125M, not just Sylber → CTC), the fix is to remove
the data-starvation confound rather than read too much into the 20h result.

**Steps, in order:**

1. **Reduce K for Sylber's sweep at small scale.** The 20h run swept
   `k_sweep: [5000, 10000, 20000, 40000]` and picked 10000 for all three
   encoders — appropriate for HuBERT/Whisper's ~3.6M frames (~360
   examples/cluster at K=10000), but Sylber's 342,505 syllables give only
   ~34 examples/cluster at that K. Add smaller candidates for Sylber
   specifically:
   ```bash
   python src/discretization.py sweep --embeddings data/embeddings/sylber --k-sweep 500 1000 2000 5000
   ```
   `sweep` already accepts `--k-sweep` as a CLI override (defaults to
   `[5000, 10000, 20000, 40000]`, same as the config), so this needs no
   code change — just don't pass it for the HuBERT/Whisper sweeps, which
   should keep using the default/config K range since their token density
   doesn't have the same problem.

2. **Scale hours to match HuBERT/Whisper's effective token count**, not
   just wall-clock audio hours. Sylber needs roughly 10x the hours to reach
   the same total token count (4.76 Hz vs ~50 Hz), so a fair matched-token
   comparison at "20h-of-HuBERT-tokens" scale means running Sylber on
   ~200h. For a comparison that's actually trustworthy per the paper's own
   floor, the real target is closer to 1,000h (Sylber's own "limited
   resource" setting) for all three encoders, not just Sylber:
   ```bash
   for enc in sylber hubert whisper; do
       python src/discretization.py extract --manifest <larger-subset-or-full-manifest.csv> --encoder $enc --out data/embeddings/$enc
       python src/discretization.py sweep --embeddings data/embeddings/$enc
       python src/discretization.py fit --embeddings data/embeddings/$enc --k <selected_k> --out models/${enc}_kmeans.pkl
       python src/train_slm.py encode --manifest <larger-subset-or-full-manifest.csv> --encoder $enc --kmeans-path models/${enc}_kmeans.pkl --out data/tokens/${enc}_tokens.jsonl
       python src/train_slm.py train --tokens data/tokens/${enc}_tokens.jsonl --encoder $enc --vocab-size <selected_k>
   done
   python src/compare_encoders.py --encoders sylber hubert whisper
   ```

3. **Smarter embedding init on vocab resize** (code change, not yet built —
   flag if you want it done before this run). `train_slm.py`'s
   `build_speech_lm` currently lets `resize_token_embeddings` randomly
   initialize all new rows from scratch. At matched-but-still-modest scale,
   initializing each cluster's embedding row from (a projection of) its
   k-means centroid — rather than pure noise — gives every token ID a
   less-arbitrary starting point, reducing how many gradient steps a
   rarely-seen cluster needs before its predictions are sane. This mainly
   helps at in-between scales (hundreds of hours); it stops mattering once
   there's enough data that every cluster gets plenty of updates anyway.

**Confirm**: re-run `compare_encoders.py` at the matched scale and check
whether Sylber's perplexity becomes competitive. If it does, that confirms
the 20h result was a pure data-scale artifact. If it's still far worse even
at matched token counts, that's real evidence of a Sylber-specific
weakness for Khmer at the SLM task specifically (independent of what the
CTC probe says about ASR).

**Cost**: substantially more GPU time and, if going past whatever subset
you already have prepared, more data preprocessing — likely days rather
than hours, especially at the 1,000h target.

## Recommendation

Run **Option A first**. It's cheap (reuses existing artifacts, no new data
or preprocessing), it directly tests the thing you actually care about
(ASR fine-tuning viability), and it's the exact scale the Sylber 2.0 paper
validates — unlike Option B, which requires committing significant new GPU
time before you learn anything. Only invest in Option B if Option A comes
back ambiguous/negative and you specifically still need the SLM/perplexity
number for a downstream decision (e.g. you decide the final architecture
should route through k-means discretization rather than a CTC/attention
decoder).

## Summary table

| | Option A (CTC probe) | Option B (scale-matched SLM) |
|---|---|---|
| Tests | Direct ASR viability (CER) | Discrete-token LM perplexity |
| Data needed | None — reuses 20h subset | ~200h (matched-token) to ~1,000h (paper's floor) |
| GPU time | Hours | Days |
| Validated by | Sylber 2.0 paper, Sec. 6.3 (20-50h) | Sylber paper's uLM section (1K-66K h) |
| Matches stated end goal | Yes (ASR) | Only if SLM architecture is still the target |
