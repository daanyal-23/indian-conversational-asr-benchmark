"""
ASR Benchmarking Pipeline — Vahan AI Intern Assignment
=======================================================
Models  : Deepgram nova-2  |  OpenAI Whisper large-v3  |  AI4Bharat IndicWhisper
Metrics : WER, CER, Entity Accuracy, Latency (API models)
Output  : results.csv  +  charts/  folder

"""

import os
import csv
import time
import json
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── third-party (installed via requirements.txt) ──────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import jiwer
import Levenshtein                     # python-Levenshtein
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from pydub import AudioSegment         # audio conversion
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
import torch
import whisper

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
AUDIO_DIR       = Path("audio files")
GROUND_TRUTH    = Path("ground_truth.csv")
OUTPUT_CSV      = Path("benchmark_results.csv")
CHARTS_DIR      = Path("charts")
CONVERTED_DIR   = Path("_converted_wav")   # temp 16kHz WAVs

DEEPGRAM_KEY    = os.environ.get("DEEPGRAM_API_KEY", "")
WHISPER_MODEL   = "large-v3"               # change to "medium" if RAM is tight
INDIC_MODEL_ID  = "collabora/whisper-base-hindi" # HuggingFace model ID

# Locality keyword for entity-accuracy check (mapped from filename stem)
LOCALITY_MAP = {
    "hsr_layout"       : "hsr layout",
    "whitefield"       : "whitefield",
    "koramangala"      : "koramangala",
    "rajajinagar"      : "rajajinagar",
    "hebbal"           : "hebbal",
    "bellandur"        : "bellandur",
    "electronic_city"  : "electronic city",
    "marathahalli"     : "marathahalli",
    "silk_board"       : "silk board",
    "yelahanka"        : "yelahanka",
    "kr_puram_station" : "kr puram",
    "peenya"           : "peenya",
    "btm_layout"       : "btm layout",
    "sarjapur_road"    : "sarjapur",
    "bommanahalli"     : "bommanahalli",
    "yeshwanthpur"     : "yeshwanthpur",
    "thanisandra"      : "thanisandra",
    "doddanekundi"     : "doddanekundi",
    "banashankari"     : "banashankari",
    "jayanagar"        : "jayanagar",
    "majestic"         : "majestic",
    "indiranagar"      : "indiranagar",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_ground_truth(csv_path: Path) -> dict:
    """Returns {filename_stem: ground_truth_text}"""
    gt = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["filename"].strip().lower()] = row["ground_truth"].strip()
    return gt


def get_condition(stem: str) -> str:
    """Extract recording condition from filename prefix."""
    prefix = stem.split("_")[0]
    mapping = {
        "quiet"   : "quiet",
        "bg"      : "background noise",
        "call"    : "phone call",
        "fast"    : "fast speech",
        "whisper" : "whispered",
    }
    return mapping.get(prefix, "unknown")


def get_locality_key(stem: str) -> str:
    """Strip condition prefix+number → locality key."""
    parts = stem.split("_", 2)
    # handle stems like 'quiet_04-rajajinagar'
    if len(parts) >= 3:
        return parts[2].replace("-", "_")
    elif len(parts) == 2:
        return parts[1].replace("-", "_")
    return stem


def convert_to_wav(src: Path, dst_dir: Path) -> Path:
    """Convert any audio format to 16kHz mono WAV (required by Whisper)."""
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / (src.stem + ".wav")
    if not dst.exists():
        audio = AudioSegment.from_file(str(src))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(dst), format="wav")
    return dst


def normalize_text(text: str) -> str:
    """Lowercase + strip punctuation for fair WER/CER comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def transliterate_to_roman(text: str) -> str:
    """
    Transliterate Devanagari → ITRANS Roman for fair WER computation.
    Models output Devanagari for Hindi speech; ground truth is Roman (Hinglish).
    Without this, phonetically correct transcriptions score 100% WER.
    """
    import re
    if re.search(r"[\u0900-\u097F]", text):
        def replace_deva(match):
            return transliterate(match.group(), sanscript.DEVANAGARI, sanscript.ITRANS)
        text = re.sub(r"[\u0900-\u097F]+", replace_deva, text)
    return text


def normalize_for_wer(text: str) -> str:
    """Transliterate Devanagari to Roman, then normalize."""
    return normalize_text(transliterate_to_roman(text))


def compute_metrics(reference: str, hypothesis: str, locality_key: str) -> dict:
    """Compute WER, CER, and Entity Accuracy for one prediction."""
    ref_norm = normalize_for_wer(reference)
    hyp_norm = normalize_for_wer(hypothesis) if hypothesis else ""

    if not hyp_norm:
        return {"wer": 1.0, "cer": 1.0, "entity_acc": 0, "entity_dist": 999}

    wer = jiwer.wer(ref_norm, hyp_norm)
    cer = jiwer.cer(ref_norm, hyp_norm)

    # Entity accuracy: is the locality name in the transcript?
    locality = LOCALITY_MAP.get(locality_key, locality_key.replace("_", " "))
    entity_acc = int(locality in hyp_norm)

    # Levenshtein distance between locality and nearest word-group in hypothesis
    # (catches near-misses like 'marathalli' for 'marathahalli')
    words = hyp_norm.split()
    loc_words = locality.split()
    n = len(loc_words)
    min_dist = min(
        Levenshtein.distance(" ".join(words[i:i+n]), locality)
        for i in range(max(1, len(words)-n+1))
    ) if words else 999

    return {
        "wer"         : round(wer, 4),
        "cer"         : round(cer, 4),
        "entity_acc"  : entity_acc,
        "entity_dist" : min_dist,
    }

# ── MODEL 1 : DEEPGRAM ────────────────────────────────────────────────────────

def transcribe_deepgram(wav_path: Path) -> tuple[str, float]:
    """Returns (transcript, latency_seconds). Returns ('', -1) on error."""
    if not DEEPGRAM_KEY:
        print("  ⚠️  DEEPGRAM_API_KEY not set — skipping Deepgram")
        return "", -1
    try:
        client = DeepgramClient(DEEPGRAM_KEY)
        options = PrerecordedOptions(
            model="nova-2",
            language="hi",          # Hindi; falls back to auto-detect
            smart_format=True,
        )
        with open(wav_path, "rb") as f:
            audio_data = {"buffer": f.read()}
        t0 = time.time()
        response = client.listen.prerecorded.v("1").transcribe_file(audio_data, options)
        latency = round(time.time() - t0, 3)
        transcript = (
            response["results"]["channels"][0]["alternatives"][0]["transcript"]
        )
        return transcript, latency
    except Exception as e:
        print(f"  Deepgram error on {wav_path.name}: {e}")
        return "", -1

# ── MODEL 2 : WHISPER ─────────────────────────────────────────────────────────

_whisper_model = None

def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        print(f"\n⏳ Loading Whisper {WHISPER_MODEL} (first load takes ~30s)...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print("✅ Whisper loaded.")
    return _whisper_model


def transcribe_whisper(wav_path: Path) -> tuple[str, float]:
    model = load_whisper()
    t0 = time.time()
    result = model.transcribe(
        str(wav_path),
        language="hi",        # hint: Hindi/Hinglish
        task="transcribe",
    )
    latency = round(time.time() - t0, 3)
    return result["text"].strip(), latency

# ── MODEL 3 : INDICWHISPER ────────────────────────────────────────────────────

_indic_pipe = None

def load_indic_whisper():
    global _indic_pipe
    if _indic_pipe is None:
        print("\n⏳ Loading IndicWhisper (first load takes ~60s)...")
        try:
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            _indic_pipe = pipeline(
                "automatic-speech-recognition",
                model=INDIC_MODEL_ID,
                device=device,
                chunk_length_s=30,
            )
            print("✅ IndicWhisper loaded.")
        except Exception as e:
            print(f"  ⚠️  IndicWhisper failed to load: {e}")
            print("      Run: pip install transformers accelerate")
            _indic_pipe = None
    return _indic_pipe


def transcribe_indic(wav_path: Path) -> tuple[str, float]:
    pipe = load_indic_whisper()
    if pipe is None:
        return "", -1
    t0 = time.time()
    try:
        result = pipe(str(wav_path))
        latency = round(time.time() - t0, 3)
        return result["text"].strip(), latency
    except Exception as e:
        print(f"  IndicWhisper error on {wav_path.name}: {e}")
        return "", -1

# ── CHARTS ────────────────────────────────────────────────────────────────────

def generate_charts(df: pd.DataFrame):
    CHARTS_DIR.mkdir(exist_ok=True)
    models = ["deepgram", "whisper", "indicwhisper"]
    colors = {"deepgram": "#378ADD", "whisper": "#1D9E75", "indicwhisper": "#D85A30"}
    conditions = df["condition"].unique()

    # ── Chart 1: WER by model (bar) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_wer = df.groupby("model")["wer"].mean().reindex(models)
    bars = ax.bar(avg_wer.index, avg_wer.values,
                  color=[colors[m] for m in avg_wer.index], width=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=11)
    ax.set_ylabel("Word Error Rate (lower is better)")
    ax.set_title("Average WER by Model", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "1_wer_by_model.png", dpi=150)
    plt.close()

    # ── Chart 2: Entity Accuracy by model ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    avg_ent = df.groupby("model")["entity_acc"].mean().reindex(models)
    bars = ax.bar(avg_ent.index, avg_ent.values,
                  color=[colors[m] for m in avg_ent.index], width=0.5)
    ax.bar_label(bars, fmt="%.0f%%",
                 labels=[f"{v*100:.0f}%" for v in avg_ent.values],
                 padding=4, fontsize=11)
    ax.set_ylabel("Entity Accuracy (higher is better)")
    ax.set_title("Locality Name Accuracy by Model", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "2_entity_acc_by_model.png", dpi=150)
    plt.close()

    # ── Chart 3: WER by condition × model (grouped bar) ──────────────────────
    pivot = df.pivot_table(index="condition", columns="model", values="wer", aggfunc="mean")
    pivot = pivot.reindex(columns=models)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax,
               color=[colors[m] for m in models],
               width=0.7, edgecolor="white")
    ax.set_ylabel("WER (lower is better)")
    ax.set_title("WER by Recording Condition", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(title="Model")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "3_wer_by_condition.png", dpi=150)
    plt.close()

    # ── Chart 4: Entity Accuracy heatmap (model × condition) ─────────────────
    pivot_ent = df.pivot_table(
        index="condition", columns="model", values="entity_acc", aggfunc="mean"
    ).reindex(columns=models)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot_ent, annot=True, fmt=".0%", cmap="YlGn",
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title("Entity Accuracy Heatmap (Condition × Model)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "4_entity_heatmap.png", dpi=150)
    plt.close()

    # ── Chart 5: WER vs Entity Accuracy scatter ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        sub = df[df["model"] == m]
        ax.scatter(sub["wer"], sub["entity_acc"] + np.random.uniform(-0.02, 0.02, len(sub)),
                   label=m, color=colors[m], alpha=0.7, s=60)
    ax.set_xlabel("WER (lower is better)")
    ax.set_ylabel("Entity Accuracy (1 = correct)")
    ax.set_title("WER vs Entity Accuracy — Key Insight", fontweight="bold")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "5_wer_vs_entity.png", dpi=150)
    plt.close()

    print(f"\n📊 Charts saved to ./{CHARTS_DIR}/")

# ── FAILURE ANALYSIS ──────────────────────────────────────────────────────────

def print_failure_analysis(df: pd.DataFrame):
    """Print worst-performing examples per model for the report."""
    print("\n" + "="*70)
    print("FAILURE ANALYSIS — Worst Locality Predictions")
    print("="*70)
    for model in df["model"].unique():
        sub = df[(df["model"] == model) & (df["entity_acc"] == 0)].copy()
        sub = sub.sort_values("entity_dist", ascending=False).head(5)
        print(f"\n▶  {model.upper()} — missed locality names:")
        if sub.empty:
            print("   (No failures — all localities correctly identified!)")
            continue
        for _, row in sub.iterrows():
            print(f"   File      : {row['filename']}")
            print(f"   Reference : {row['reference']}")
            print(f"   Predicted : {row['hypothesis']}")
            print(f"   WER={row['wer']:.2f}  CER={row['cer']:.2f}  "
                  f"Edit-dist={row['entity_dist']}")
            print()

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  ASR BENCHMARK — Vahan Intern Assignment")
    print("="*70)

    # Load ground truth
    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {GROUND_TRUTH}")
    gt = load_ground_truth(GROUND_TRUTH)
    print(f"\n✅ Loaded {len(gt)} ground-truth entries")

    # Collect audio files
    audio_files = sorted(AUDIO_DIR.glob("*"))
    audio_files = [f for f in audio_files if f.suffix in {".m4a",".ogg",".wav",".mp3"}]
    print(f"✅ Found {len(audio_files)} audio files in '{AUDIO_DIR}/'")

    if not audio_files:
        raise RuntimeError(f"No audio files found in '{AUDIO_DIR}/'")

    records = []

    for audio_path in audio_files:
        stem = audio_path.stem.lower()
        if stem not in gt:
            print(f"  ⚠️  No ground truth for '{stem}' — skipping")
            continue

        reference    = gt[stem]
        condition    = get_condition(stem)
        locality_key = get_locality_key(stem)
        wav_path     = convert_to_wav(audio_path, CONVERTED_DIR)

        print(f"\n🎙  {stem}  [{condition}]")

        for model_name, transcribe_fn in [
            ("deepgram",     transcribe_deepgram),
            ("whisper",      transcribe_whisper),
            ("indicwhisper", transcribe_indic),
        ]:
            hypothesis, latency = transcribe_fn(wav_path)
            metrics = compute_metrics(reference, hypothesis, locality_key)

            flag = "✅" if metrics["entity_acc"] else "❌"
            print(f"  {model_name:<14} WER={metrics['wer']:.2f}  "
                  f"CER={metrics['cer']:.2f}  Entity={flag}  "
                  f"({'latency: '+str(latency)+'s' if latency > 0 else 'local'})")

            records.append({
                "filename"    : stem,
                "condition"   : condition,
                "locality_key": locality_key,
                "model"       : model_name,
                "reference"   : reference,
                "hypothesis"  : hypothesis,
                "latency_s"   : latency,
                **metrics,
            })

    # ── Build results DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Results saved to {OUTPUT_CSV}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY — Aggregate Metrics per Model")
    print("="*70)
    summary = df.groupby("model").agg(
        avg_wer        = ("wer",        "mean"),
        avg_cer        = ("cer",        "mean"),
        entity_acc_pct = ("entity_acc", lambda x: f"{x.mean()*100:.1f}%"),
        avg_latency    = ("latency_s",  lambda x: f"{x[x>0].mean():.2f}s" if (x>0).any() else "local"),
    ).reindex(["deepgram","whisper","indicwhisper"])
    print(summary.to_string())

    # ── Charts ────────────────────────────────────────────────────────────────
    generate_charts(df)

    # ── Failure analysis ─────────────────────────────────────────────────────
    print_failure_analysis(df)

    print("\n" + "="*70)
    print("DONE. Files generated:")
    print(f"  📄 {OUTPUT_CSV}")
    print(f"  📊 {CHARTS_DIR}/1_wer_by_model.png")
    print(f"  📊 {CHARTS_DIR}/2_entity_acc_by_model.png")
    print(f"  📊 {CHARTS_DIR}/3_wer_by_condition.png")
    print(f"  📊 {CHARTS_DIR}/4_entity_heatmap.png")
    print(f"  📊 {CHARTS_DIR}/5_wer_vs_entity.png")
    print("="*70)


if __name__ == "__main__":
    main()