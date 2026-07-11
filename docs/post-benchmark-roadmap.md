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

## Where this fits into `docs/run-roadmap.md`

This doc doesn't replace the original roadmap's numbered steps — it's a
follow-up to how step 4's Path A branch actually turned out, so here's how
the two options map onto that existing sequence:

- **Steps 1-2** (data prep, zero-shot eval) — already done for the 20h
  subset; both options below reuse that same manifest/audio as-is (Option
  B only needs *more* of it, not different prep — see the "Also have a
  folder to save logs" / preprocessing discussion: prep is identical
  across encoders and options, only the encoder-specific extraction step
  downstream of it forks).
- **Step 3** (conditional fine-tune) — was skipped for the 20h run since
  zero-shot passed the 4-5Hz sanity check. Both options' backup ladders
  (see below) can loop back to this step if a frozen/zero-shot Sylber
  turns out to be the actual bottleneck rather than the encoder itself.
- **Step 4** (Path A vs. Path B decision point) — Path A was the branch
  taken; `docs/path-a-encoder-comparison.md` is that run's result, and this
  doc is specifically about what to do with Path A's ambiguous outcome
  before deciding whether to proceed to Path B. It is not a new instance of
  the same decision — it's "Path A came back unclear, now what."
- **Step 5** (SLM training) — **Option B below is a re-run of this exact
  step**, just at larger scale and with a Sylber-specific K. **Option A
  bypasses step 5 entirely**, going from steps 1-3's encoder straight to a
  supervised CTC decoder instead of an unsupervised SLM.

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

### Pilot result (Colab smoke test, ~40min/300-utterance subset)

Ran `notebooks/colab_pilot_run.ipynb`'s deliberately tiny smoke test
(240 train / 30 val utterances, 8 epochs) before committing to a real 20h+
run, and got a concrete, interpretable result — not just "inconclusive at
this scale" like the SLM path was:

- **HuBERT**: `final_val_cer=0.985` — expected-bad at this scale (240
  utterances / 8 epochs is nowhere near enough to train a ~90-class CTC
  head from scratch), not a real signal either way. A firmer HuBERT number
  needs the full 20h+ run, same as any encoder would.
- **Sylber**: every val utterance skipped (`input_shorter_than_target`),
  so `final_val_cer=0.0` is a false zero (no comparisons were made), not
  "perfect." Diagnostic (encoder_output_shape, label_length) examples from
  epoch 1: `(43,768)/70`, `(42,768)/64`, `(26,768)/35`, `(18,768)/29`,
  `(32,768)/47` — the output shape varies correctly per utterance (rules
  out a batch-dimension bug in the encoder adapter), but Sylber's token
  count `T` is consistently **~60-70% of** the grapheme-cluster label
  length `L` (ratio ≈0.66 averaged across the 5 examples). That's a real,
  systematic shortfall, not noise — zero-shot Sylber (English-only
  pretraining) is detecting roughly a third fewer syllable boundaries than
  Khmer transcripts imply.

**This is not a code bug to keep chasing** — `T < L` here reflects actual
segmentation quality, and no amount of CTC-probe-side fixing (grapheme
clustering already applied) can manufacture tokens Sylber didn't produce.
It's concrete evidence for backup-ladder step 1 above: **fine-tune Sylber's
backbone on Khmer boundary supervision (Step 3) before the CTC probe can
produce a usable Sylber number at all**, not just before trusting it. Until that fine-tune
happens, treat Sylber's CTC probe as blocked, and if you want any
usable pilot signal in the meantime, it's HuBERT's (with the caveat that
240/8-epochs is still too small to mean much either).

### Free lever found: `merge_threshold` (before committing to Step 3)

Sylber's segment boundaries come from a fixed, rule-based heuristic
(`sylber.utils.segment_utils.get_segment`), not a learned head: a frame is
"voiced" above `norm_threshold` (default 2.6), and consecutive voiced
frames merge into one segment while their running-average cosine
similarity stays ≥ `merge_threshold` (default 0.8) — both fixed,
English-tuned constants. `src/segmentation.py sweep-thresholds` (new) sweeps
these against real Khmer data with zero training and zero annotation —
just re-running the cheap thresholding step on cached backbone hidden
states — to see how much of the T<L gap is a tunable-threshold artifact
versus something only backbone fine-tuning can fix.

Result, same 30-utterance sample as the pilot above (`norm_threshold` barely
matters; `merge_threshold` is the dominant lever):

| merge_threshold | mean_token_rate_hz | mean T/L | % utterances CTC-viable (T≥L) |
|---|---|---|---|
| 0.80 (Sylber's default) | 4.45 | 0.66 | 0% |
| 0.95 | 5.61 | 0.83 | 6.7% |
| 0.97 | 6.18 | 0.92 | 20% |
| **0.98** | **6.81** | **1.01** | **66.7%** |
| 0.99 | 8.26 | 1.23 | 96.7% |
| 0.995 | 10.43 | 1.55 | 100% |
| 0.999 | 21.54 | 3.22 | 100% |

There's a real tradeoff, not a free lunch: pushing `merge_threshold` high
enough gets 100% CTC-viability, but by ~0.99 the token rate is already
8-10Hz and by 0.999 it's 21Hz+ — no longer "syllabic" at all, just
frame-level fragmentation wearing the segmentation API. That defeats the
actual point of using Sylber (compact ~4-5Hz tokens) even if it makes the
CTC probe's `T≥L` check pass. **`merge_threshold≈0.98` is the practical
sweet spot**: token rate only ~36% above the 4-5Hz reference band (not
2-4x), mean T/L≈1.01 (right at parity), and 2/3 of utterances become
CTC-viable — recovering most of the usable signal without abandoning the
syllable-rate property that makes Sylber worth using over a frame-level
encoder in the first place.

**Confirmed by actually running the CTC probe** (not just the segmentation
proxy stats above) — `python src/train_ctc.py train --encoder sylber
--merge-threshold 0.98 --epochs 8` on the same 240/30 pilot split:

| | default (merge_threshold=0.8) | merge_threshold=0.98 |
|---|---|---|
| train used/skipped | 0/240 | **154/240** |
| val used/skipped | 0/30 | **22/30** |
| train_loss trend | flat 0.0000 (nothing computed) | 6.08 → 5.24 → 4.70 → 4.62 → 4.56 → 4.51 → 4.47 → **4.45** |
| final val_cer | 0.0 (false zero, no comparisons made) | **0.868** |

Sylber's CER (0.868) at `merge_threshold=0.98` is now actually *lower*
(better) than HuBERT's 0.987 at this same tiny pilot scale — though both
numbers are still not trustworthy in absolute terms at 240
utterances/8 epochs, this is the first time Sylber has produced **any**
real, non-degenerate CTC signal at all, closing the loop from a total
blocker to a working (if still small-scale) comparison.

`--norm-threshold`/`--merge-threshold` are now wired through
`train_ctc.py`/`encoders.py load_encoder`/`segmentation.load_segmenter`,
and `python src/segmentation.py sweep-thresholds` runs the free,
zero-training sweep itself (see table above).

**Recommendation**: use `--merge-threshold 0.98` as the default for any
further Sylber CTC-probe runs (a full 20h+ run, not just this 300-utterance
smoke test) before spending the Step 3 annotation effort — then decide
whether the remaining gap (~36% of utterances still skipped) is worth the
Step 3 fine-tune on top. Even after the threshold fix, remember the CER
itself still needs a much larger run to be trustworthy — this only fixes
"can we compute a number at all," not "is the number meaningful yet."

### Fair comparison: restrict CER to a shared subset, not each encoder's own

Comparing `final_val_cer` numbers straight from `train_ctc.py train`'s
per-encoder output is confounded whenever encoders skip different
utterances — an encoder that discards more of the hard examples (like
Sylber, even at `merge_threshold=0.98`, where 8/30 val utterances still
fail `T≥L`) has its CER measured on an easier, pre-filtered subset than an
encoder that skips nothing (HuBERT, 0/30 skipped here). `python
src/train_ctc.py compare --encoders sylber hubert` (new command) fixes
this: it loads both already-trained checkpoints (no retraining), finds the
intersection of val utterances CTC-viable for *every* requested encoder,
and reports CER on exactly that shared subset for each.

Result (same 240/30 pilot, sylber at `merge_threshold=0.98`):

| | own viable subset | CER (own subset) | CER (shared 22/30 subset) |
|---|---|---|---|
| sylber | 22/30 | 0.867 | 0.867 |
| hubert | 30/30 | 0.985 | 0.985 |

In this case HuBERT's CER barely moved when restricted to Sylber's
easier subset (0.985→0.985) — its errors are fairly uniform across the
val set, so the original unrestricted comparison happened not to be badly
biased here. But that's a property of this specific run, not something to
assume in general; **`compare` is what actually confirms it rather than
hoping it's true**, and the result holds either way: Sylber beats HuBERT on
a fair, identical-subset comparison at this pilot scale. Still not a
trustworthy absolute number at only 240 train utterances/8 epochs — same
caveat as before — but the *relative* comparison between the two encoders
is now methodologically sound, not just numerically similar.

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
