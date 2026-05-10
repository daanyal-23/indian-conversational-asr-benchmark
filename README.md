# ASR Shootout — Indian Conversational Speech Benchmark
### Vahan AI Intern Assignment

Benchmarks three ASR systems on 22 Bangalore locality name recordings across 5 real-world noise conditions, with script-normalized WER metrics for fair multilingual evaluation.

---

## Results at a Glance

| Model | Avg WER (normalized) | Avg CER | Entity Accuracy | Avg Latency |
|---|---|---|---|---|
| **Deepgram nova-2** | **0.73** | **0.31** | **13.6% (3/22)** | ~4.7s |
| Whisper large-v3 | 0.78 | 0.28 | 13.6% (3/22) | ~24.4s (CPU) |
| collabora/whisper-base-hindi | 1.08 | 0.54 | 0% (0/22) | ~5.9s (CPU) |

> All WER/CER computed after Devanagari → Roman transliteration for fair multilingual comparison. See `report.md` for full analysis.

---

## Repo Structure

```
asr-benchmark/
├── audio files/             ← 22 source audio clips (.m4a / .ogg)
├── charts/                  ← 5 output charts (auto-generated)
│   ├── 1_wer_by_model.png
│   ├── 2_entity_acc_by_model.png
│   ├── 3_wer_by_condition.png
│   ├── 4_entity_heatmap.png
│   └── 5_wer_vs_entity.png
├── ground_truth.csv         ← Roman-script ground truth for all 22 clips
├── benchmark_results.csv    ← Full per-file benchmark output
├── run_benchmark.py         ← Main pipeline script
├── requirements.txt         ← Python dependencies
├── report.md                ← Full 3-page analysis report
├── .env.example             ← API key template (copy to .env)
└── README.md
```

---

## Setup & Run

### 1. Prerequisites

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install **ffmpeg** (required for audio conversion):
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

### 2. Set your Deepgram API key

```bash
cp .env.example .env
# Then edit .env and paste your key
```

Get a free API key at [deepgram.com](https://deepgram.com) — the free tier is sufficient for this dataset.

### 3. Run the full pipeline

```bash
python run_benchmark.py
```

This will:
- Convert all audio to 16kHz WAV
- Transcribe each clip with Deepgram nova-2, Whisper large-v3, and whisper-base-hindi
- Transliterate Devanagari output → Roman (ITRANS) before metric computation
- Compute WER, CER, Entity Accuracy, and latency per file per model
- Export `benchmark_results.csv` and generate all 5 charts in `charts/`

First run takes ~10–15 minutes — Whisper large-v3 (~2.9GB) downloads and caches on first use.

---

## Dataset

22 audio recordings across 5 conditions designed to simulate real-world Vahan call scenarios:

| Condition | Files | Description |
|---|---|---|
| `quiet_*` | 6 | Clean room — controlled baseline |
| `bg_*` | 6 | Street / traffic background noise |
| `call_*` | 4 | Compressed OGG — simulates VOIP phone call quality |
| `fast_*` | 4 | Rushed / fast speech |
| `whisper_*` | 2 | Whispered / low-energy speech |

All clips feature conversational Hindi/Hinglish sentences embedding one Bangalore locality name.

---

## Models

| Model | Type | Notes |
|---|---|---|
| **Deepgram nova-2** | Cloud API | Baseline; real-time multilingual STT |
| **OpenAI Whisper large-v3** | Open-source, local (CPU) | Industry-standard multilingual benchmark |
| **collabora/whisper-base-hindi** | Open-source, local (CPU) | Whisper-base fine-tuned on Hindi via AI4Bharat's Shrutilipi dataset |

> `ai4bharat/indicwhisper` and `ai4bharat/whisper-medium-hi` were both unavailable on HuggingFace (gated or 404) at time of testing. `collabora/whisper-base-hindi` was used as the closest publicly accessible Hindi-specialized alternative.

---

## Key Design Decision: Script Normalization

Both Deepgram and Whisper return **Devanagari** for Hindi speech while ground truth is in **Roman script** (Hinglish). Without normalization, phonetically correct transcriptions score 100% WER. This pipeline fixes that by transliterating all model output to ITRANS Roman before computing any metric:

```python
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

transliterate(devanagari_text, sanscript.DEVANAGARI, sanscript.ITRANS)
```

This reduced average WER by ~0.25 points across models — not cosmetic, but a correction for a real measurement artifact.

---

## Key Takeaways

This benchmark highlights a critical real-world ASR evaluation problem:

> **Low WER does not imply usable downstream performance.**

All three models frequently produce phonetically plausible but semantically wrong locality names — especially for Kannada-origin entities embedded inside Hindi/Hinglish utterances:

| Ground Truth | Deepgram | Whisper | Hindi-Whisper |
|---|---|---|---|
| Doddanekundi | दो धन्य कुंडली | दोधन नेकुंडी | दो दन्य कुंडी |
| Thanisandra | सनी संतरा | धनि संदरा | ठीक अंदर आ |
| Banashankari | वन शंकरी | वन शंकरी | वन शंकरी |

Key conclusions:
- Entity extraction robustness matters more than aggregate WER for this use case
- Script normalization is essential for fair multilingual evaluation
- Hindi-specific fine-tuning on a smaller model *hurts* performance on Kannada-origin locality names — model scale and multilingual pretraining win
- A fuzzy-match NER post-processing layer is the highest-ROI near-term fix

---

## Output Files

| File | Description |
|---|---|
| `benchmark_results.csv` | Per-file WER, CER, Entity Accuracy, latency — all 3 models |
| `charts/1_wer_by_model.png` | Average WER per model |
| `charts/2_entity_acc_by_model.png` | Locality name accuracy per model |
| `charts/3_wer_by_condition.png` | WER broken down by noise condition |
| `charts/4_entity_heatmap.png` | Entity accuracy heatmap (condition × model) |
| `charts/5_wer_vs_entity.png` | WER vs Entity Accuracy scatter — key insight chart |

---

## Configuration

Top-level constants in `run_benchmark.py` you may want to change:

```python
WHISPER_MODEL  = "large-v3"                    # change to "medium" if RAM is limited
INDIC_MODEL_ID = "collabora/whisper-base-hindi" # swap for any HF ASR model
```

---

## Author

**Syed Daanyal**
Vahan AI Intern Assignment — ASR Benchmarking for Indian Conversational Speech