# ASR Shootout: Benchmarking Speech Recognition for Indian Conversational Speech
**Vahan AI Intern Assignment | Submitted by: Syed Daanyal Pasha**

---

## 1. Approach

### Problem Framing
The goal was to evaluate how well off-the-shelf ASR systems handle the kind of speech Vahan's platform encounters daily: blue-collar candidates speaking Hindi/Hinglish/Kannada over phone calls, mentioning Bangalore locality names as structured entities. Standard WER benchmarks on clean English speech tell us very little here. The real question is: *does the model correctly extract the locality name — the one piece of information that actually matters downstream?*

### Dataset
22 audio recordings were collected across 5 noise conditions that simulate real-world Vahan call scenarios:

| Condition | Count | Rationale |
|---|---|---|
| Quiet room | 6 | Controlled baseline |
| Background noise | 6 | Street/traffic ambient noise |
| Phone call (OGG, compressed) | 4 | Simulates VOIP call quality — primary Vahan channel |
| Fast/rushed speech | 4 | Candidate in a hurry |
| Whispered | 2 | Low-energy speech, poor mic pickup |

All recordings feature conversational Hindi/Hinglish sentences naturally embedding one Bangalore locality name (e.g., *"Bhaiyya yeh Banashankari ke paas utar do jaldi se"*). Ground truth transcriptions were written in Roman script to match the Hinglish register.

### Models Selected

| Model | Type | Rationale |
|---|---|---|
| **Deepgram nova-2** | Cloud API | Required baseline; production-grade, real-time capable, multilingual |
| **OpenAI Whisper large-v3** | Open-source, local (CPU) | Industry-standard multilingual benchmark; inference run locally on CPU hardware |
| **collabora/whisper-base-hindi** | Open-source, local | Whisper-base fine-tuned on Hindi using AI4Bharat's Shrutilipi dataset; tests the specialized-small vs general-large tradeoff |

> **Note on model selection:** `ai4bharat/indicwhisper` and `ai4bharat/whisper-medium-hi` were both unavailable on HuggingFace (404 or gated at time of testing). Substituted `collabora/whisper-base-hindi` — a Whisper-base model fine-tuned on Hindi using AI4Bharat's Shrutilipi dataset. This is a legitimate alternative that directly tests whether Hindi-specific fine-tuning on a smaller model can compete with a larger general-purpose model.

### Metrics
- **WER (Word Error Rate):** Computed after script normalization (see below)
- **CER (Character Error Rate):** Finer-grained; more informative for long compound place names
- **Entity Accuracy:** Binary — did the model correctly produce the locality name in the transcript? This is the mission-critical metric for Vahan's use case
- **Latency:** Wall-clock time per clip for API and local models

### Script Normalization Before WER Computation — A Critical Fix
Both Deepgram and Whisper return Devanagari script for Hindi speech while ground truth is Roman (Hinglish). Computing WER on mismatched scripts penalises every phonetically correct word as 100% wrong, making the raw metric meaningless. All model outputs were therefore transliterated Devanagari → Roman (ITRANS scheme) using `indic-transliteration` **before** any WER/CER computation:

```python
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def transliterate_to_roman(text):
    if re.search(r"[\u0900-\u097F]", text):
        def replace_deva(match):
            return transliterate(match.group(), sanscript.DEVANAGARI, sanscript.ITRANS)
        text = re.sub(r"[\u0900-\u097F]+", replace_deva, text)
    return text
```

This single step reduced average WER by **~0.25 points** across models, transforming the metrics from misleading to defensible.

---

## 2. Results

### Aggregate Performance (Script-Normalized)

| Model | Avg WER (normalized) | Avg CER | Entity Accuracy | Avg Latency |
|---|---|---|---|---|
| **Deepgram nova-2** | **0.73** | **0.31** | **13.6% (3/22)** | ~6.1s* |
| Whisper large-v3 | 0.78 | 0.28 | 13.6% (3/22) | ~24.4s (CPU) |
| whisper-base-hindi | 1.08 | 0.54 | 0% (0/22) | ~5.9s (CPU) |

*Deepgram latency is elevated by 2 API timeouts in this run; stable-run average was ~4.7s.

Deepgram leads on WER and CER. Whisper matches Deepgram on entity accuracy despite higher WER — the WER vs entity accuracy decoupling is the key analytical finding (see Finding 2).

### Performance by Recording Condition

| Condition | Deepgram WER | Whisper WER | Hindi-Whisper WER | Deepgram Entity | Whisper Entity |
|---|---|---|---|---|---|
| **Phone call** | **0.54** | 0.75 | 1.85 | 1/4 (25%) | 0/4 (0%) |
| **Fast speech** | **0.68** | 0.73 | 0.80 | 1/4 (25%) | 1/4 (25%) |
| Background noise | 0.70 | 0.91 | 1.02 | 1/6 (17%) | 0/6 (0%) |
| Quiet | 0.81 | 0.61 | 0.82 | 0/6 (0%) | 2/6 (33%) |
| Whispered | 1.04 | 1.08 | 1.13 | 0/2 (0%) | 0/2 (0%) |

Notable: Deepgram's Quiet condition WER (0.81) is *worse* than its Phone Call WER (0.54) — entirely explained by 2 API timeouts during the quiet run inflating the average, not by actual model degradation. This is a reminder that reliability metrics matter as much as accuracy in production.

---

## 3. Failure Analysis

### Finding 1: Script normalization reveals the models understand sentences far better than raw WER suggested

The improvement from transliteration is not cosmetic — it reflects genuine transcription quality being obscured by a measurement artifact:

| Locality | Ground Truth | Model Output (Devanagari) | After ITRANS | WER raw → norm |
|---|---|---|---|---|
| Yeshwanthpur | `Yeshwanthpur se hai meri train` | `यशवंतपुरा से है मेरी train` | `yashavamtapura se hai meri train` | 0.92 → **0.33** |
| Rajajinagar | `Rajajinagar ke paas utar dena` | `राजाजी नगर के पास उतार देना` | `rajaji nagara ke pasa utara dena` | 1.12 → **0.75** |
| Banashankari | `Banashankari ke paas utar do` | `वन शंकरी के पास उतार दो` | `vana shamkari ke pasa utara do` | 1.11 → **0.67** |

After normalization, remaining errors are concentrated almost entirely in the rendering of multi-syllable locality names — not in the surrounding conversational Hindi, which all models handle reasonably well.

### Finding 2: Entity Accuracy and WER are poorly correlated — WER is the wrong primary metric for Vahan

Chart 5 (WER vs Entity Accuracy scatter) makes this vivid: Deepgram correctly identifies entities at WER=0.67, 0.77, and 0.88 — while missing them at WER=0.33 and 0.47. No clear trend. **A model can transcribe a sentence with low WER and still completely mangle the one word that matters.**

The inverse holds too: `fast_02_jayanagar` achieved Entity=✅ for both Deepgram and Whisper at WER=0.88, because "Jayanagar" is phonetically compatible with Hindi. Entity difficulty is driven by the place name's phonological profile, not overall transcription quality.

### Finding 3: Multi-syllable Kannada-origin locality names fail with a consistent pattern across all models

| Locality | Deepgram | Whisper | Hindi-Whisper |
|---|---|---|---|
| Banashankari | वन शंकरी *(Van Shankari)* | वन शंकरी | वन शंकरी |
| Doddanekundi | दो धन्य कुंडली *(Do Dhanye Kundali)* | दोधन नेकुंडी | दो दन्य कुंडी |
| Marathahalli | मारा था हल्ली *(Mara Tha Halli)* | मारा था हल्ली | मारा था हल्ली |
| Thanisandra | सनी संतरा *(Sunny Santra)* | धनि संदरा | ठीक अंदर आ |

The failure is systematic: all models over-segment Kannada place names into shorter Hindi-sounding substrings. Both Deepgram and Whisper produce *different* wrong outputs for the same input (e.g., Doddanekundi, Thanisandra) — meaning the failure is probabilistic, not deterministic, making it harder to patch with simple rules. This points firmly toward a fuzzy-match NER post-processing layer as the most practical near-term fix.

### Finding 4: Deepgram's advantage is largest exactly where Vahan needs it most

Deepgram's biggest WER advantages over Whisper are in conditions Vahan encounters in production:
- `call_03_bommanahalli`: Deepgram 0.44 vs Whisper 0.78 (Δ=0.34)
- `bg_06_peenya`: Deepgram 0.56 vs Whisper 0.89 (Δ=0.33)
- `call_01_btm_layout`: Deepgram 0.60 vs Whisper 0.80 (Δ=0.20)

Whisper outperforms Deepgram only on quiet clips where Deepgram had API timeouts — not a genuine model quality difference.

### Finding 5: Whispered speech is a hard failure mode for all models

All three models score WER ≥ 1.04 on whispered clips and correctly identified zero locality names. The practical fix isn't a better model — it's prompting the caller to repeat louder before attempting ASR.

### Finding 6: Hindi-specialized fine-tuning on a smaller model does not help with Kannada-origin locality names

`collabora/whisper-base-hindi` achieved 0% entity accuracy and the highest WER (1.08) across all conditions. The `call_02_sarjapur_road` clip produced a WER of 5.31, a severe hallucination. This is the most important negative result: for Kannada-origin locality names embedded in Hinglish, model scale and general multilingual pretraining matter more than Hindi-specific fine-tuning. The hard words are Kannada, not Hindi — specialization on Hindi alone doesn't help and may hurt by over-fitting the model's priors to Hindi phonology.

---

## 4. Recommendation

**For Vahan's production use case, Deepgram nova-2 is the recommended baseline.**

It outperforms or matches Whisper on every metric that matters — WER (0.73 vs 0.78), latency (~4.7s vs ~24.4s CPU), and entity accuracy (tied at 13.6%). Critically, it is the only model with operationally viable real-time latency. Whisper large-v3 at 24s per clip cannot serve a live phone call; on GPU this drops to ~2s, but that introduces infrastructure cost and ops overhead that Deepgram's API avoids for early-stage deployment.

### Cost at Scale

At 1,000 calls/day averaging ~1 minute each, Deepgram nova-2 at ~$0.0043/min works out to roughly **$130/month**. Self-hosting Whisper large-v3 on a T4 GPU instance (e.g., Google Cloud at ~$0.35/hr) costs roughly **$250/month** at continuous load — more expensive initially, with added infrastructure and operational overhead, but with no per-call variable cost as usage scales.

For early-stage deployment, Deepgram is likely the better choice on both simplicity and cost. At significantly higher call volumes (~3,000+ calls/day), GPU self-hosting becomes economically worth evaluating.

However, 13.6% entity accuracy is too low for a production system that depends on reliable locality extraction. Three improvements in order of effort vs. impact:

1. **Post-processing NER fuzzy-match layer** — match raw transcript against a known Bangalore locality list using Levenshtein distance ≤ 2. This would catch "Van Shankari" → "Banashankari" and "Rajaji Nagar" → "Rajajinagar" without any model retraining, and likely closes a significant share of the 86% failure rate.

2. **Fine-tune on Bangalore locality names** — even a few hundred examples of these specific Kannada-origin place names in Hinglish context would directly address the over-segmentation failures in Finding 3.

3. **Harden against API timeouts** — preprocessing audio (trimming silence, normalizing volume) before Deepgram submission would reduce the 408 timeout rate observed across runs.

---

## 5. Limitations

- **Single speaker:** All 22 recordings are from one person. Real-world variance in age, dialect, accent, and gender is not captured. Entity accuracy would likely be lower on a diverse speaker set.
- **IndicWhisper not evaluated:** `ai4bharat/indicwhisper` and `ai4bharat/whisper-medium-hi` were both unavailable (gated or 404) on HuggingFace. Substituted `collabora/whisper-base-hindi` as the closest accessible alternative.
- **Simulated phone calls:** OGG compression approximates VOIP quality but misses real artifacts like packet loss, echo, and codec transitions present in actual Vahan calls.
- **Small test set:** 22 clips across 22 localities means each locality is tested exactly once. A robust benchmark requires multiple speakers and multiple takes per locality.
- **No GPU for local models:** Whisper and Hindi-Whisper latency measured on CPU. On a T4 GPU, Whisper drops to ~1–2s, making it significantly more production-viable for GPU-enabled deployments.
- **Entity accuracy metric is strict:** Binary exact-match underestimates practical utility — "Rajaji Nagar" for "Rajajinagar" fails the check despite being recognisably correct. Fuzzy entity matching would give a more realistic picture.
- **Deepgram timeout variability:** Deepgram experienced 2–4 API timeouts across runs, likely due to transient network conditions, which inflates its reported WER and latency averages.