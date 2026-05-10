# ASR Shootout — Indian Conversational Speech Benchmark
### Vahan AI Intern Assignment

Benchmarks three ASR systems on 22 Bangalore locality name recordings across 5 real-world noise conditions, with script-normalized WER metrics for fair multilingual evaluation.

---

## Results at a Glance

| Model | Avg WER (normalized) | Avg CER | Entity Accuracy | Avg Latency |
|---|---|---|---|---|
| **Deepgram nova-2** | **0.68** | **0.26** | **18.2% (4/22)** | ~3.7s |
| Whisper large-v3 | 0.78 | 0.28 | 13.6% (3/22) | ~24.8s (CPU) |
| IndicWhisper | — | — | — (failed to load) | — |

> All WER/CER computed after Devanagari → Roman transliteration. See `report.md` for full analysis.

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
├── ground_truth_fixed.csv   ← Roman-script ground truth for all 22 clips
├── benchmark_results.csv              ← Full per-file benchmark output
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

Get a free API key at [deepgram.com](https://deepgram.com) , the free tier is more than sufficient for this dataset.

### 3. Run the full pipeline

```bash
python run_benchmark.py
```

This will:
- Convert all audio to 16kHz WAV
- Transcribe each clip with Deepgram nova-2 and Whisper large-v3
- Transliterate Devanagari output → Roman (ITRANS) before metric computation
- Compute WER, CER, Entity Accuracy, and latency per file per model
- Export `benchmark_results.csv` and generate all 5 charts in `charts/`

First run takes ~10 minutes — Whisper large-v3 (~2.9GB) downloads and caches on first use.

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
| **OpenAI Whisper large-v3** | Open-source, local (CPU) | Industry-standard benchmark |
| **AI4Bharat IndicWhisper** | Open-source, local | Could not be evaluated — model identifier unavailable on HuggingFace at time of testing |

---

## Key Design Decision: Script Normalization

Both Deepgram and Whisper return **Devanagari** for Hindi speech while ground truth is in **Roman script** (Hinglish). Without normalization, phonetically correct transcriptions score 100% WER. This pipeline fixes that by transliterating all model output to ITRANS Roman before computing any metric:

```python
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

transliterate(devanagari_text, sanscript.DEVANAGARI, sanscript.ITRANS)
```

This reduced average WER by ~0.25 points across both models.

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

In `run_benchmark.py`, top-level constants you may want to change:

```python
WHISPER_MODEL  = "large-v3"   # change to "medium" if RAM is limited
INDIC_MODEL_ID = "ai4bharat/indicwhisper"  # update when correct ID is available
```
