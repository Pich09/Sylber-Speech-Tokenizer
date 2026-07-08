# Audio Tokenizer Comparison & Sylber-for-Khmer Implementation Guide

## Introduction: Why Tokenization Design Matters

Speech tokenization is the bridge between continuous audio and discrete language models. The choice of tokenizer fundamentally shapes:

- **Token rate** (tokens/second) — determines context length and compute cost. Higher rates (≫10 Hz) provide fine-grained acoustic detail but blow out context budgets; lower rates (4–5 Hz) compress information but risk losing linguistic nuance.
- **Information type** — acoustic/self-supervised semantics (HuBERT, Sylber) vs. full waveform reconstruction capability (codecs like EnCodec, Mimi).
- **Supervision signal** — self-supervised clustering (discrete units, Sylber) vs. adversarial codec training vs. hybrid discrete-continuous.
- **Downstream task fit** — spoken language modeling benefits from low rates + semantic tokens; codec-based systems are better for TTS or speech-to-speech; real-time dialogue needs streaming-friendly latency profiles.

This document compares five tokenization approaches and outlines how to apply the most promising (Sylber's syllabic tokenization) to Khmer, a low-resource language.

---

## Tokenizer Comparison Table

| **Approach** | **Paper ID** | **Token Rate** | **Granularity** | **Supervision** | **Reconstruction** | **Primary Use** | **Key Trade-off** |
|---|---|---|---|---|---|---|---|
| **Sylber** | 2410.07168, 2601.22306 | 4–5 Hz | Syllable-level | Self-supervised SSL + segmentation | Semantic only | SLM pre-training | Lowest compute cost; zero-shot cross-lingual |
| **HuBERT** | 2106.07447 | 25–50 Hz | Frame-level (10ms windows) | Offline k-means clustering | Semantic only | SLM baseline; ALT tasks | Frame-level detail; high token volume |
| **EnCodec** | 2210.13438 | 75 Hz (24kHz) | Codec frames | Adversarial reconstruction loss | Full waveform | Neural codec; speech compression | Full reconstruction; requires high-quality audio training data |
| **Mimi (Moshi)** | 2410.00037 | 12.5 Hz | RVQ hierarchy | Adversarial codec + dual-stream design | Full waveform + dual streams | Real-time dialogue; codec-LM | Semantic + acoustic streams; real-time latency (160ms theoretical) |
| **Kimi Tokenizer** | 2504.18425 | 12.5 Hz | Hybrid discrete + continuous | LLM-based integration | Semantic + continuous features | Multi-modal audio LLM | Hybrid representation; 13M hours pre-training at scale |

---

## Tokenizer Deep-Dives

### 1. Sylber: Syllabic Embedding Representation of Speech

**Papers:**
- [arXiv 2410.07168](https://arxiv.org/pdf/2410.07168) (ICLR 2025)
- [arXiv 2601.22306](https://arxiv.org/abs/2601.22306) (Sylber 2.0: Universal Syllable Embedding)

**Architecture:**
Sylber reuses HuBERT's CNN feature extractor + Transformer backbone, initializing from SDHuBERT. The key innovation is a **syllable-level segmentation algorithm** that groups frames into phonologically-motivated units, then **averages SSL features within each segment** to produce one token per syllable. This yields ~4–5 Hz token rates — **6–7× sparser than HuBERT**.

**Training Data & Scale:**
- v1: LibriSpeech (960h) + LibriLight (subset)
- v2: Expanded to multilingual data with same phonological principles

**Reported Results:**
The paper "Scaling Spoken Language Models with Syllabic Speech Tokenization" (2509.26634) trains OPT-125M and Qwen2.5-0.5B on 3 datasets (LibriSpeech → LibriSpeech + LibriLight → all plus Spoken TinyStories) using syllabic tokens. Syllabic tokenization **achieves comparable or better performance** on downstream tasks while reducing training time **2.5×** and FLOPs **5–8×** relative to HuBERT baselines.

**Cross-Lingual Generalization:**
Crucially, Sylber trained only on English audiobooks **generalizes zero-shot to Spanish and Mandarin** without fine-tuning, indicating that syllable boundaries encode language-universal phonological properties. This is the key precedent for Khmer adaptation.

**Limitations:**
- Segmentation algorithm tuned for English/Romance/Sino-Tibetan phonotactics; Khmer's complex abugida syllable structure (onset clusters, register distinctions, inherent vowels) may not transfer perfectly.
- No explicit handling of tonal or register distinctions (though Khmer register is implemented via inherent vowel quality, not suprasegmental tone like Mandarin).
- Sylber 2.0 is explicitly designed for multilingual settings; likely the better starting point than v1 for Khmer.

**Relevant Code:**
- [Berkeley-Speech-Group/sylber](https://github.com/Berkeley-Speech-Group/sylber) — pip-installable. Load with `from sylber import Segmenter; Segmenter(model_ckpt="sylber")`.
- Related: [findsylls toolkit](https://arxiv.org/abs/2603.26292) — standardized language-agnostic syllable tokenization (Sylber, VG-HuBERT, others).

---

### 2. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

**Paper:** [arXiv 2106.07447](https://arxiv.org/abs/2106.07447)

**Architecture:**
CNN feature extractor (similar to wav2vec 2.0) feeding a 12-layer Transformer. Training uses a **two-stage offline clustering process**: (1) compute k-means on unlabeled audio features (typically 100 clusters initially), (2) refine via a second clustering iteration, then (3) apply masked prediction loss (BERT-style) to the aligned cluster labels.

**Training Data & Scale:**
LibriSpeech (960h) + proprietary unlabeled data. Achieves 19% relative WER reduction on challenging evaluation subsets vs. wav2vec 2.0 when using a 1B-parameter model.

**Token Rate & Properties:**
Produces **25–50 Hz tokens** depending on stride/dupe-detection post-processing (after deduplication, ~25 Hz in standard practice). Frame-level granularity means fine acoustic detail at the cost of high token volume.

**Downstream Performance:**
State-of-the-art on speech recognition and downstream ALT (audio language understanding) benchmarks at release. Widely adopted as a baseline discrete-unit tokenizer.

**Limitations:**
- High token rate inflates context length in downstream models (10–15× more tokens than Sylber for the same audio).
- Acoustic detail is sometimes a liability for pure language-modeling tasks (syllabic aggregation may generalize better).
- No explicit cross-lingual training; multilingual HuBERT (mHuBERT) exists but requires multilingual pre-training data.

---

### 3. High-Fidelity Neural Audio Compression

**Paper:** [arXiv 2210.13438](https://arxiv.org/pdf/2210.13438)

**Architecture:**
End-to-end encoder-decoder neural codec with **Residual Vector Quantization (RVQ)** bottleneck. Uses a multiscale spectrogram discriminator to enforce perceptual quality; includes a novel **loss balancer** mechanism to decouple hyperparameter tuning from the loss scale.

**Token Rate & Representation:**
Produces **~75 Hz tokens at 24 kHz** (proportionally higher at 48 kHz). Each token is a discrete codebook entry from the RVQ hierarchy.

**Key Innovation:**
The loss balancer enables stable adversarial training across multiple loss terms (reconstruction, adversarial, auxiliary) without manual hyperparameter rebalancing — valuable for codec-LM systems that must integrate codec losses with LLM objectives.

**Performance:**
Achieves state-of-the-art reconstruction fidelity on MUSHRA subjective tests across speech, noisy-reverberant speech, and music. Lightweight Transformer variants can compress an additional 40% with minimal quality loss.

**Limitations:**
- High token rate unsuitable for context-efficient SLM pre-training.
- Requires high-quality audio training data; performs worse on noisy or far-field speech (unlike discrete units, which abstract away acoustic variation).
- Codec training is computationally expensive; typically a pre-trained model is used downstream rather than re-trained per language.

---

### 4. Moshi: Speech-Text Foundation Model for Real-Time Dialogue

**Paper:** [arXiv 2410.00037](https://arxiv.org/pdf/2410.00037)

**Architecture & Token Representation:**
Moshi treats spoken conversation as **speech-to-speech generation** rather than chaining ASR→dialogue→TTS. It models **overlapping speech streams** (simultaneous user and system audio) using a unified discrete token space derived from a **neural audio codec's RVQ residuals**. Key innovation: an **"Inner Monologue" method** that predicts text tokens *before* audio tokens, improving linguistic quality and enabling low-latency streaming.

**Token Rate:**
Encodes audio at **12.5 Hz** (similar to Kimi, but using a codec backbone rather than a hybrid discrete-continuous design).

**Real-Time Performance:**
Achieves 160ms theoretical latency and 200ms practical latency for full-duplex spoken dialogue — enabling interruptions, overlapping speech, and conversational naturalness absent from traditional pipelines.

**Downstream Use:**
Codec-LM foundation model for real-time dialogue; speech-to-speech generation; conversational AI.

**Limitations:**
- RVQ codec-based approach means full waveform reconstruction capability (not just semantics), which requires more parameters and training data than discrete-semantic tokenizers.
- Real-time constraints (dual-stream modeling, parallel user/system) add architectural complexity not needed for non-interactive tasks like SLM pre-training.

---

### 5. Kimi-Audio: Open-Source Foundation Model for Audio Understanding, Generation, and Conversation

**Paper:** [arXiv 2504.18425](https://arxiv.org/pdf/2504.18425) — Technical Report

**Architecture & Token Design:**
Kimi uses a **hybrid discrete-token + continuous-feature tokenizer** that accepts continuous audio features while producing discrete tokens fed to an LLM backbone. The system combines:
- A dedicated audio encoder that extracts task-specific features (e.g., semantic for understanding, acoustic for generation).
- A **12.5 Hz discrete tokenizer** derived from the audio features.
- A **chunk-wise streaming detokenizer** using flow matching for efficient generation.

**Training Data & Scale:**
Curated **13+ million hours** of pre-training audio across speech, music, and sound domains. Continual pre-training on combined audio-text data with specialized tasks (speech recognition, audio understanding, dialogue).

**Downstream Performance:**
State-of-the-art on speech recognition, audio understanding, audio Q&A, and speech conversation benchmarks. Provides released code, model checkpoints, and evaluation toolkits.

**Design Rationale:**
The hybrid discrete-continuous approach allows Kimi to leverage LLM pre-training while maintaining the acoustic fidelity of continuous features. This is scalable to very large audio corpora (13M hours) without the per-language codec training cost of Moshi or EnCodec.

**Limitations:**
- Architectural complexity of hybrid tokenization; training and inference both require careful orchestration of discrete and continuous streams.
- Very large-scale pre-training (13M hours) is not typical for low-resource languages; unclear how the approach degrades on small corpora like Khmer.

---

## Takeaways & Sylber Selection for Khmer

### Why Sylber?

For a Khmer syllable tokenizer, **Sylber is the best starting point**:

1. **Lowest Token Rate** — 4–5 Hz means minimal context overhead in downstream models, critical for low-compute environments and small Khmer datasets.

2. **Proven Cross-Lingual Transfer** — Zero-shot generalization to Spanish and Mandarin (despite English-only training) shows that syllable boundaries capture language-universal phonological structure. Khmer is linguistically distant from English but shares universal syllabic structure.

3. **No Codec Training Required** — Unlike Moshi, Kimi, or EnCodec, Sylber doesn't require learning an adversarial codec from high-quality audio. This is a huge advantage for a low-resource language where we have ~106 hours of speech vs. millions of hours of codec training data in high-resource settings.

4. **Semantic-Only Representation** — Syllabic tokens encode *linguistic meaning* (phonotactics, syllable structure) not *acoustic detail* (speaker identity, background noise). This abstraction is more robust for low-resource, noisy Khmer speech.

5. **Established Pre-Trained Weights** — Sylber 2.0 is released with pre-trained models; we can use zero-shot or fine-tune directly.

### Limitations to Watch

- **Abugida Phonotactics** — Khmer is an abugida script; syllables have complex onset consonant clusters (up to 3 consonants) and register distinctions implemented via inherent vowel quality. Sylber's English-trained segmentation may misalign cluster boundaries or confuse register distinctions.
- **Untested on Southeast Asian Languages** — Sylber is tested on Spanish, Mandarin, and English. No published results on Khmer, Thai, Lao, or other Indic-script languages.
- **No Inherent Vowel Modeling** — In Khmer, vowels are often implicit in the abugida (default vowel is /ə/ or /a/; other vowels require diacritics). Sylber may not explicitly model this, requiring manual annotation or fine-tuning to capture properly.

---

## Part 2: Sylber-for-Khmer Implementation Roadmap

This section outlines concrete steps to apply Sylber to Khmer, from baseline zero-shot evaluation to fine-tuning and validation.

### Final Architecture: Sylber Encoder + OPT-125M Decoder

This roadmap implements a two-stage architecture where Sylber acts as a speech encoder and OPT-125M acts as a language model decoder for **speech understanding and token prediction** (not audio-to-text transcription).

```
STAGE A: SPEECH ENCODER (Sylber)
  Raw Audio (Khmer, optionally extended to Khmer+English code-switched)
      ↓
  [HuBERT CNN + Transformer backbone] → self-supervised speech features
      ↓
  [Sylber Segmentation Algorithm] → detects syllable boundaries
      ↓
  [Syllable-level averaging] → continuous embeddings (~4–5 Hz)
      ↓
  [K-means discretization] → discrete token vocabulary (token IDs)
  
  Adaptation phases:
    A1: Fine-tune on Khmer-only (~728h, DDD-Cambodia dataset)
    A2: Further fine-tune on Khmer+English code-switched audio (if data available)
    A3: Re-run k-means on combined embeddings (post-A2) for unified vocabulary

STAGE B: LANGUAGE MODEL DECODER (OPT-125M)
  Discrete tokens (output of Stage A)
      ↓
  [OPT-125M, pretrained on English text]
      ↓
  Continued pretraining via next-token prediction over speech tokens
      ↓
  **FINAL OUTPUT: Token ID (an integer, e.g., 18)**
  
  **Output is a token ID in speech-token space, not text or audio.** OPT-125M learns
  linguistic patterns from Khmer speech tokens and produces a probability distribution
  over the vocabulary (K=10,000 token IDs); you select the highest-probability token.
  Example: Given syllables [5, 23, 41, 12], model predicts token 18 with 87% confidence.
  
  **To get audio or text from tokens:** See "Encoder Reusability: Alternative Decoder
  Heads" section below — add a vocoder to convert tokens → audio, or a CTC/ASR decoder
  to convert tokens → text. Those are separate extensions not in this roadmap.
```

**Key properties:**
- **Encoder (A)** can be frozen and reused with multiple decoder heads (CTC, diffusion, etc.)
- **Decoder (B)** is optional for speech-understanding use cases; required for SLM evaluation and downstream tasks
- **No random initialization**: Sylber leverages HuBERT pretrain; OPT-125M leverages English text pretrain

---

### Overview: Training vs. Fine-Tuning vs. Clustering

**This roadmap does not train any model from random initialization.** All steps leverage pretrained checkpoints:

| Step | Activity | Model State |
|---|---|---|
| Step 2 (Zero-shot) | Run Sylber segmenter, no updates | **Pretrained Sylber checkpoint** (unchanged) |
| Step 3/3b (Adaptation) | Fine-tune segmenter boundaries on Khmer (A1), then Khmer+code-switch (A2) | **Pretrained Sylber** → **fine-tuned on Khmer** → **further fine-tuned on code-switched** |
| Step 4 (Discretization) | K-means clustering on embeddings | No learning/backprop (unsupervised clustering) |
| Step 5/6 (SLM) | Train language model on token sequences | **Pretrained OPT-125M/Qwen2.5-0.5B** → **continued pretraining on Khmer tokens** |

**Why not train the segmenter from scratch on Khmer?**
Even with 728 hours of data, Khmer corpus is small relative to Sylber's original English training scale (LibriLight, ~50k hours). Syllable segmentation learned from English generalizes reasonably to typologically distant languages (Spanish, Mandarin, and—we conjecture—Khmer). Fine-tuning a model with already-learned priors is expected to outperform training from zero on a single-language, single-domain corpus. Training from scratch can be an optional **ablation study** if compute permits, not the default path.

---

### Step 1: Gather & Curate Khmer Audio Data

**Primary Corpus:**
- **[DDD-Cambodia/khmer-speech-dataset](https://huggingface.co/datasets/DDD-Cambodia/khmer-speech-dataset)** (Hugging Face)
  - **~728 hours** of speech-text pairs
  - **450,396 utterances**, 12 native speakers (5 female, 7 male)
  - **Dataset Size**: 495.9 GB total
  - **Audio Format**: 16kHz WAV files, ~8 seconds per utterance on average
  - **Coverage**: 61 topics, 1,749 subtopics on Cambodian cultural content
  - **License**: Creative Commons Attribution Share Alike 4.0 (CC-BY-SA-4.0)
  - **Access**: [Hugging Face Datasets](https://huggingface.co/datasets/DDD-Cambodia/khmer-speech-dataset)
  - **Note**: This dataset represents a substantial upgrade from earlier Khmer corpora, providing 7–18× more data than prior collections. The scale is now closer to supporting full model fine-tuning rather than head-only adaptation.

**Secondary/Evaluation Corpora:**
- **DDD-Cambodia/khm-asr-cultural** (Hugging Face) — 134.6 hours, 56,726 utterances, 8 speakers. Same publisher/lineage as the primary dataset; check for overlap before treating as additive.
- **OpenSLR 42** — 3.97 hours (male only). Out-of-domain evaluation set since it comes from different recording conditions than the DDD-Cambodia series.

**Open Tasks (worth investigating):**
- Search FLEURS, VoxLingua107, Common Voice archives for additional Khmer audio.
- Self-supervised pre-training doesn't require transcripts; untranscribed Khmer speech (e.g., from YouTube, podcasts) can be included if licensed appropriately.
- Estimate total available hours (target: 150–300h if possible to boost generalization).

**Data Preprocessing:**
- Resample all audio to 16 kHz (Sylber standard).
- Remove silence/padding (standard VAD).
- Split into train/val/test (e.g., 80/10/10).
- Optional: Normalize loudness (loudness normalization helps with encoder generalization).

---

### Step 2: Baseline Zero-Shot Evaluation

**Hypothesis:**
Sylber 2.0 trained on multilingual data will segment Khmer audio zero-shot with reasonable accuracy, given demonstrated transfer to typologically distant languages (Spanish, Mandarin).

**Procedure:**
1. Load the pre-trained Sylber 2.0 segmenter:
   ```python
   from sylber import Segmenter
   segmenter = Segmenter(model_ckpt="sylber_v2")  # or check for latest ckpt name
   ```

2. Run on a **diverse sample** of Khmer audio (50–100 utterances across age/gender/topic):
   ```python
   for wav_file in khmer_samples:
       segments = segmenter(wav_file, in_second=True)
       # segments = [(start, end), (start, end), ...]
   ```

3. **Manual evaluation** against Khmer phonotactics:
   - Do segment boundaries align with expected syllable onsets?
   - Are onset consonant clusters (e.g., /kr/, /kl/, /pr/) kept together or split?
   - Are register/vowel distinctions respected (do inherent vs. explicit vowels segment consistently)?

4. **Quantitative proxy** (if you have a small hand-annotated reference):
   - Compute boundary precision/recall vs. ground truth.
   - Target: >80% boundary agreement for a pass.

5. **Document findings** (e.g., in a `results/zero_shot_evaluation.txt`):
   - Which phenomena work well? Which fail?
   - Are failures systematic (e.g., all cluster-onsets mis-segmented) or random?

**Decision Point:**
- **If zero-shot is ≥80% accurate on manual inspection**: Proceed to Step 4 (discretization). Fine-tuning may not be necessary.
- **If zero-shot is <80%**: Proceed to Step 3 (adaptation/fine-tuning).

---

### Step 3: Adaptation/Fine-Tuning (if needed)

**Rationale:**
If zero-shot segmentation fails on Khmer's specific phonotactics, fine-tune the segmentation module on Khmer audio rather than training from scratch. Sylber's original training is on ~100h of LibriLight audio; with ~728h of Khmer data now available, this is significantly above the original training scale, opening two adaptation strategies:
- **Conservative (head-only)**: Freeze the encoder, fine-tune only the segmentation algorithm. Data-efficient, safe, preserves pretrained features.
- **Aggressive (full model)**: Unfreeze and fine-tune the entire model (CNN + Transformer + segmentation). With 728h, this becomes viable and may better capture Khmer's abugida phonotactics. Compare both on a held-out validation set.

**Fine-Tuning Strategy:**
1. **Prepare pseudo-labels** (self-supervised, no transcripts needed):
   - Use zero-shot Sylber to segment all Khmer audio.
   - Manually correct a small subset (~500–1000 utterances, ~10–20 hours) to establish ground truth.
   - Use these corrected segments as training targets.

2. **Choose a fine-tuning strategy** (see rationale above):
   - **Head-only (conservative)**: Keep the CNN + Transformer backbone frozen (initialized from pretrain); only train the segmentation algorithm. More sample-efficient; good if zero-shot is close to passing.
   - **Full-model (aggressive)**: Unfreeze the backbone and train all layers end-to-end. With 728h, this is data-feasible; likely yields better Khmer-specific phonology. Run both variants and compare on validation data.

3. **Training details**:
   - Optimizer: Adam, lr=1e-4 to 1e-5 (conservative, since we're not fully retraining).
   - Loss: Binary cross-entropy or similar on boundary detection.
   - Batch size: 8–16 (limited by GPU memory if using one).
   - Epochs: 5–10 (stop early based on validation set).

4. **Validation**:
   - Evaluate fine-tuned segmenter on held-out Khmer set (10–20 hours).
   - Compare precision/recall vs. zero-shot.
   - Target: ≥90% boundary agreement.

**Output:**
- Fine-tuned checkpoint: `checkpoints/sylber_khmer_v1.pth` (or similar).
- Test scores: Save to `results/fine_tune_evaluation.txt`.

---

### Step 3b: Code-Switch Adaptation (Stage A2)

**Objective:**
Extend the fine-tuned Khmer Sylber encoder to handle Khmer–English code-switching, enabling a unified encoder that segments and tokenizes mixed-language utterances correctly.

**Data Requirement (Open Task):**
- **Khmer–English code-switched speech corpus** — currently unresourced. Candidates:
  - Search for existing corpora (e.g., research groups working on multilingual ASR, code-switching linguistics).
  - If no public corpus exists, consider collecting ~10–50 hours of code-switched utterances from bilingual Khmer-English speakers (annotation: mark language boundaries, validate segmentations).
  - Minimum feasible scale: ~10 hours (can fine-tune from the Khmer-adapted checkpoint A1).
  - Ideal scale: ~50–100 hours (allows broader generalization to diverse code-switching patterns).

**Fine-Tuning Strategy (Continuation from Step 3):**
1. Load the Step 3 checkpoint (Khmer-adapted Sylber).
2. Fine-tune the segmenter on code-switched audio using the same procedure as Step 3 (head-only or full-model, validated on held-out code-switched test set).
3. Target: ≥85–90% boundary agreement on code-switched utterances, validated against manual inspection of a small reference set.

**Updated K-means Vocabulary (Step 4 extension):**
Once A2 fine-tuning is complete, re-run k-means discretization (Step 4) on the combined embedding space:
- Extract syllable embeddings from both Khmer-only and code-switched audio.
- Cluster across both modalities into a single vocabulary (K=10000–20000, same sweep as Step 4).
- This unified vocabulary will encode both Khmer and English phonetic clusters.
- Document: the resulting model is now `sylber_khmer_english_v1.pth` (Stage A, full pipeline complete).

**Output:**
- Fine-tuned checkpoint: `checkpoints/sylber_khmer_english_v1.pth`.
- Code-switched evaluation: `results/code_switch_evaluation.txt`.
- Unified k-means model: `models/khmer_english_kmeans_20k.pkl`.

**Status:** This step is **conditional on acquiring code-switched speech data**. If corpus-building is not feasible, proceed directly from Step 3 to Step 4 using Khmer-only data, and mark code-switching as a future extension.

---

### Sylber Checkpoint Reusability Scope

Once Sylber is fine-tuned on Khmer audio (Steps 3/3b), it becomes specialized for Khmer phonotactics. This is powerful for Khmer and Khmer-English tasks, but **does not** generalize to unrelated languages as well as the original pretrained checkpoint does. Understanding this distinction is critical for reusing the encoder:

| Checkpoint | Generalization | Best Use Case | Limitations |
|---|---|---|---|
| **Base Pretrained Sylber** (English-trained) | ✅ Zero-shot cross-lingual (Spanish, Mandarin demonstrated) | Encoding audio in *new* languages where no fine-tuned checkpoint exists; establishing a baseline for languages with diverse phonotactics. | Segmentation optimized for English/Romance/Sino-Tibetan; may miss language-specific patterns. |
| **Khmer-fine-tuned Sylber** (post Step 3) | ⚠️ Specialized for Khmer phonotactics | Encoding Khmer-only audio; expected to outperform base model on Khmer speech. | Segmentation head has adapted to Khmer's abugida structure + register distinctions; unlikely to generalize well to typologically distant languages (French, Arabic, etc.). Don't feed unrelated-language audio here. |
| **Khmer+English-fine-tuned Sylber** (post Step 3b, if done) | ✅ Handles Khmer + English code-switching | Encoding Khmer, English, or mixed Khmer-English utterances within the same utterance. | Still specialized for this language pair; not expected to handle other languages well. |

**Practical Guidance:**
- **Keep both checkpoints.** Save the base pretrained Sylber checkpoint separately from the Khmer-adapted one — they serve different reuse purposes and neither should overwrite the other.
- **For Khmer tasks:** Use the Khmer-fine-tuned checkpoint (better performance on Khmer phonotactics).
- **For other new languages:** Use the base pretrained checkpoint (better zero-shot transfer) unless you have reason to believe it won't work, then consider fine-tuning a fresh copy on that language's audio.
- **For code-switching:** Use the Khmer+English-fine-tuned checkpoint (if Step 3b is completed).

---

### Step 4: Discretization — Build Token Vocabulary

**Procedure:**
1. **Extract syllable-averaged embeddings**:
   - For each segment (syllable) boundary from Step 2 or 3, extract the SSL feature within that segment.
   - Average the features across the segment (temporal mean pooling).
   - Result: one embedding vector per syllable (~4–5 Hz token rate).

2. **K-means clustering**:
   - Cluster the embeddings into **K discrete tokens** using k-means.
   - Sweep K over the range **[5000, 10000, 20000, 40000]** (following the scaling paper).
   - For each K, compute:
     - **Token rate** (tokens/second): should be ~4–5 Hz (if not, check segmentation quality).
     - **Cluster balance**: % of tokens in top-10 clusters. Imbalanced clusters (e.g., >50% in one cluster) indicate over-clustering or a data quality issue.

3. **Select vocabulary size**:
   - Choose K that balances expressiveness vs. cluster health.
   - Recommendation: Start with **K=10000** (middle of the sweep). Adjust based on cluster balance metrics.

4. **Save tokenizer**:
   - Save the k-means model: `models/khmer_kmeans_10k.pkl` (or use joblib/pickle).
   - Document: `configs/tokenizer_config.yaml` with K, vocab size, mean token rate.

**Sanity Check:**
- Token rate should be 4–5 Hz. If >10 Hz, segmentation is too fine. If <2 Hz, too coarse.
- **Token count estimate** (sanity check on scale): 728h * 3600s/h * 4.5 tokens/sec ≈ 11.8M tokens. This is substantial but manageable for SLM training (OPT-125M training on ~10–20M tokens is typical).
- Visualize a sample of segmentations + token assignments (e.g., a spectogram with boundaries and token IDs overlaid). Include in `results/tokenization_examples/`.

---

### Step 5: Evaluation — Validate Tokenizer Quality

**Challenge:**
No established Khmer syllable-segmentation benchmark exists. Evaluation requires a proxy task.

**Option A: Hand-Annotated Reference (Small-scale)**
- Annotate syllable boundaries for 50–100 Khmer utterances (8–10 hours total).
- Compute precision/recall of automatic segmentation vs. reference.
- Pros: Direct measurement of phonotactic correctness.
- Cons: Expensive in time; requires Khmer linguistic expertise.

**Option B: Downstream Proxy Task (Scalable)**
Train a small **Khmer Spoken Language Model (SLM)** on the tokenized data and measure perplexity or generation quality (following the "Scaling Spoken Language Models with Syllabic Speech Tokenization" recipe):

- **Model**: OPT-125M or Qwen2.5-0.5B (mimic the scaling paper).
- **Training**: 
  - Encode all ~728h of Khmer audio (primary dataset) into discrete tokens using the trained tokenizer. This yields ~11.8M tokens at 4–5 Hz token rate.
  - Train the language model on token sequences (next-token prediction, autoregressive).
  - Optionally, finetune on a small Khmer text corpus (e.g., Khmer Wikipedia) if available, to ground linguistic understanding.
  
- **Metrics**:
  - Perplexity on a held-out Khmer audio test set.
  - (Stretch) Language modeling quality: e.g., consistency of predictions (do repeated contexts yield similar continuations?).
  - (Stretch) Downstream task: e.g., Khmer ASR reranking (use LM to rerank ASR hypotheses).

- **Baseline Comparison** (if feasible):
  - Train the same LM on HuBERT tokens (25 Hz) from the same Khmer audio.
  - Compare perplexities: Sylber tokens should match or beat HuBERT tokens with far fewer tokens.

**Documentation:**
- Save results to `results/downstream_eval/` with perplexity curves, token counts, and WER/quality metrics.

---

### Step 6: Stretch Goal — End-to-End Khmer Spoken Language Model

**Objective:**
Validate the entire pipeline (segmentation → discretization → LM training) with a complete spoken language model trained on Khmer tokens.

**Setup:**
1. **Encode all Khmer audio** into token sequences using the fine-tuned tokenizer + k-means vocabulary.
2. **Train a small Khmer SLM** via continued pretraining:
   - Model: OPT-125M or Qwen2.5-0.5B, pre-trained on English (transfer learning).
   - Data: ~728h Khmer audio (primary dataset), converted to ~11.8M tokens at 4–5 Hz token rate.
   - Continued pretraining for ~10–50 epochs until convergence (exact schedule depends on compute and learning rate).
3. **Evaluation**:
   - Perplexity on held-out test set.
   - Qualitative: Sample generations (e.g., "given the first 5 syllables of a Khmer utterance, does the model predict sensible continuations?").
   - Quantitative (if transcripts available): Measure coherence of predicted token sequences against ground-truth transcripts.

**Expected Outcome:**
If Sylber's syllable tokenization generalizes to Khmer, the model should learn reasonable linguistic structure (syllable-level patterns, typical Khmer phonotactic sequences) and outperform a baseline HuBERT-token model in efficiency (fewer tokens for the same quality).

**Deliverables:**
- Trained model checkpoint: `models/khmer_slm_opt125m.pth`.
- Evaluation report: `results/end_to_end_eval.md` (perplexity, token efficiency, sample outputs).

---

## Project Structure & File Organization

```
Speech Tokenizer/
├── docs/
│   └── audio-tokenizer-comparison.md          (this file)
├── data/
│   ├── khmer_asr_cultural_v2/                 (primary corpus, ~106h)
│   ├── openslr_42/                            (supplementary, ~4h)
│   └── preprocessing/
│       └── prepare_data.py                    (resample, VAD, split)
├── models/
│   ├── sylber_checkpoints/                    (downloaded Sylber v2.0)
│   ├── khmer_kmeans_10k.pkl                   (k-means vocabulary)
│   └── khmer_slm_opt125m.pth                  (trained SLM, Step 6)
├── configs/
│   └── tokenizer_config.yaml                  (k, vocab size, token rate)
├── src/
│   ├── segmentation.py                        (zero-shot + fine-tuning)
│   ├── discretization.py                      (k-means, vocabulary building)
│   ├── tokenizer.py                           (end-to-end tokenizer class)
│   └── train_slm.py                           (Step 6, optional)
├── results/
│   ├── zero_shot_evaluation.txt               (Step 2 findings)
│   ├── fine_tune_evaluation.txt               (Step 3, if applicable)
│   ├── tokenization_examples/                 (spectograms, boundaries, token IDs)
│   ├── downstream_eval/                       (Step 5, perplexity, metrics)
│   └── end_to_end_eval.md                     (Step 6, if done)
└── README.md                                  (project overview)
```

---

## Encoder Reusability: Alternative Decoder Heads

Once the Sylber encoder (Stage A) is trained and discretized, it can be frozen and paired with different decoder heads for alternative tasks. Below is a comparison of feasibility and data requirements:

| Decoder Head | Feasibility | Data Requirement | Use Case | Notes |
|---|---|---|---|---|
| **AR Decoder (OPT-125M/Qwen2.5-0.5B)** | ✅ Planned | Unlabeled audio only | Speech understanding, next-token prediction (Stage B, current roadmap) | OPT-125M learns language patterns from Khmer speech tokens. Output is speech-token space, not text. Requires ~10–50 epochs of training on ~11.8M tokens. |
| **CTC Decoder** | ✅ Feasible | **Transcribed audio (audio-text pairs)** | Automatic Speech Recognition (ASR) / speech-to-text | Feed continuous syllable embeddings (pre-k-means) into CTC loss against Khmer character/phoneme sequences. Requires aligned audio + transcripts (new data sourcing task, not yet in roadmap). Would enable Khmer speech→text transcription. |
| **Diffusion Decoder** | ⚠️ Experimental | Unlabeled audio only | Generative speech modeling, TTS-style synthesis, token generation | Two sub-variants: (1) **Continuous diffusion** conditioned on Sylber embeddings to generate waveforms (diffusion vocoder, similar to diffusion TTS models), or (2) **Discrete diffusion** over k-means tokens as an alternative to AR generation. No established recipe in the literature yet — would be a novel extension requiring careful design and validation. Higher risk but potentially powerful for speech generation. |

**Key Property:**
The encoder (Stage A) can remain frozen and reused across all decoder heads once fine-tuned and discretized. This mirrors how HuBERT features are reused across many downstream tasks in speech processing. Each decoder head requires its own training, but the encoder's syllable boundaries and embeddings are shared, saving compute and ensuring consistency.

**Practical Implications:**
- **AR decoder (current plan):** Use unlabeled audio. Fast iteration, lower compute barrier.
- **CTC decoder:** Requires finding or collecting Khmer transcribed audio (~50–500 hours for good performance). Unlocks ASR applications.
- **Diffusion decoder:** More speculative; useful if you want to generate or synthesize Khmer speech conditioned on semantic understanding. Requires research to establish a working architecture.

---

## Next Steps & Open Questions

1. **Data Access** — Confirm access to Khmer ASR Cultural Dataset V2. Register with Mozilla Data Collective if needed. Download and validate ~728h Khmer-only audio before proceeding to Step 2.

2. **Linguistic Validation** — Recruit a Khmer linguist to spot-check zero-shot segmentations against Khmer phonotactics (abugida onset clusters, register distinctions). This is critical for Step 2 decision-making.

3. **Code-Switched Data (Step 3b)** — Identify or collect Khmer–English code-switched speech (~10–100 hours). This is currently the main blocker for Step 3b (Stage A2). Without this corpus, the pipeline can proceed through Step 6 with Khmer-only data; code-switching becomes a future extension.

4. **Compute Requirements** — Assess available GPU/TPU:
   - Sylber fine-tuning (Step 3): ~1–2 GPU-days on 16GB VRAM.
   - SLM training (Step 6 / Stage B): depends on model size (OPT-125M: ~1–2 GPU-weeks for 10–50 epochs on ~11.8M tokens).

5. **CTC Data (Alternative: Encoder Reusability)** — If you want to build a Khmer ASR system later, source transcribed Khmer audio (~50–500 hours with aligned text). This is a future alternative to the AR decoder, not a blocker for the current roadmap.

6. **Khmer Text Corpus** — For Step 6 (SLM training), a small Khmer text corpus (Wikipedia, news, etc.) is useful but not required. Prioritize audio tokens; text is supplementary.

7. **Diffusion Decoder Research** — If interested in speech generation or synthesis, conduct a literature review on discrete/continuous diffusion models for speech. This is experimental and not part of the current plan.

8. **Benchmarking** — Identify an existing Khmer ASR system or task to use as a downstream benchmark for validating tokenizer quality (Step 5, Option B).

---

## References

1. Scaling Spoken Language Models with Syllabic Speech Tokenization. 2509.26634.
2. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. 2106.07447.
3. High Fidelity Neural Audio Compression. 2210.13438.
4. Moshi: a speech-text foundation model for real-time dialogue. 2410.00037.
5. Kimi-Audio Technical Report. 2504.18425.
6. Sylber: Syllabic Embedding Representation of Speech from Raw Audio. 2410.07168 (ICLR 2025).
7. Sylber 2.0: A Universal Syllable Embedding. 2601.22306.
8. findsylls: A Language-Agnostic Toolkit for Syllable-Level Speech Tokenization and Embedding. 2603.26292.
9. Berkeley-Speech-Group/sylber. [GitHub](https://github.com/Berkeley-Speech-Group/sylber).
10. Khmer ASR Cultural Dataset (V2). [Mozilla Data Collective](https://datacollective.mozillafoundation.org/datasets/cml9h5vgc01bxmn075sjeftek).
