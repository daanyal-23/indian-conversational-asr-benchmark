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
| **AI4Bharat IndicWhisper** | Open-source, local | Fine-tuned on Indian languages; the expected challenger |

> **Note on IndicWhisper:** IndicWhisper could not be evaluated due to unavailable/publicly inaccessible model identifiers at the time of experimentation (`ai4bharat/indicwhisper` returned a 404 on HuggingFace). Its results are excluded from quantitative analysis. It remains an important candidate for future evaluation once the correct versioned release is identified.

### Metrics
- **WER (Word Error Rate):** Computed after script normalization (see below)
- **CER (Character Error Rate):** Finer-grained; more informative for long compound place names
- **Entity Accuracy:** Binary — did the model correctly produce the locality name in the transcript? This is the mission-critical metric for Vahan's use case
- **Latency:** Wall-clock time per clip for API models

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

This single step reduced average WER by **0.25 points for Deepgram** and **0.27 points for Whisper**, transforming the metrics from misleading to defensible.

---

## 2. Results

### Aggregate Performance (Script-Normalized)

| Model | Avg WER (raw) | Avg WER (normalized) | Avg CER | Entity Accuracy | Avg Latency |
|---|---|---|---|---|---|
| **Deepgram nova-2** | 0.93 | **0.68** | **0.26** | **18.2% (4/22)** | ~3.7s |
| Whisper large-v3 | 1.05 | **0.78** | 0.28 | 13.6% (3/22) | ~24.8s *(CPU, no GPU)* |
| IndicWhisper | — | — | — | — (failed to load) | — |

Transliteration normalization meaningfully changed the picture: Deepgram's WER dropped from 0.93 → 0.68 and Whisper's from 1.05 → 0.78. Deepgram leads on every dimension — WER, CER, entity accuracy, and latency.

### Performance by Recording Condition

| Condition | Deepgram WER | Whisper WER | Deepgram Entity | Whisper Entity |
|---|---|---|---|---|
| **Phone call** | **0.54** | 0.75 | 1/4 (25%) | 0/4 (0%) |
| **Quiet** | **0.58** | 0.61 | 2/6 (33%) | 2/6 (33%) |
| Fast speech | 0.68 | 0.73 | 1/4 (25%) | 1/4 (25%) |
| Background noise | 0.75 | 0.91 | 0/6 (0%) | 0/6 (0%) |
| Whispered | 1.04 | 1.08 | 0/2 (0%) | 0/2 (0%) |

Two standout results: Deepgram achieves its best performance under **phone call conditions** (WER=0.54) , exactly where Vahan needs it most. And `quiet_02_whitefield` achieved **WER=0.00** for both models , a perfect transcription when conditions are clean and the place name is phonetically simple.

---

## 3. Failure Analysis

### Finding 1: Script normalization reveals the models understand sentences far better than raw WER suggested

The improvement from transliteration is not cosmetic , but rather it reflects genuine transcription quality being obscured by a measurement artifact:

| Locality | Ground Truth | Model Output (Devanagari) | After ITRANS | WER raw → norm |
|---|---|---|---|---|
| Yeshwanthpur | `Yeshwanthpur se hai meri train` | `यशवंतपुरा से है मेरी train` | `yashavamtapura se hai meri train` | 0.92 → **0.33** |
| Rajajinagar | `Rajajinagar ke paas utar dena` | `राजाजी नगर के पास उतार देना` | `rajaji nagara ke pasa utara dena` | 1.12 → **0.75** |
| Banashankari | `Banashankari ke paas utar do` | `वन शंकरी के पास उतार दो` | `vana shamkari ke pasa utara do` | 1.11 → **0.67** |

After normalization, the remaining errors are concentrated almost entirely in the rendering of multi-syllable locality names — not in the surrounding conversational Hindi, which both models handle well.

### Finding 2: Entity Accuracy and WER are poorly correlated — WER is the wrong primary metric for Vahan

Although absolute entity accuracy appears low (18.2% for Deepgram), most failures were phonetically close approximations rather than complete misses. Chart 5 (WER vs Entity Accuracy scatter) makes this vivid: Deepgram correctly identifies entities at WER=0.62, 0.77, and 0.88 — while missing them at WER=0.33 and 0.44. The absence of any clear trend in the scatter confirms WER is not a reliable proxy for the metric Vahan actually cares about. **A model can transcribe a sentence with low WER and still completely mangle the one word that matters.**

The inverse is also true: `fast_02_jayanagar` achieved Entity=✅ for both models at WER=0.88, because "Jayanagar" is phonetically simple and phonologically compatible with Hindi. Entity difficulty is driven by the place name's phonological profile, not the sentence's overall transcription quality.

### Finding 3: Multi-syllable Kannada-origin locality names fail with a consistent pattern across both models

| Locality | Ground Truth | Deepgram | Whisper |
|---|---|---|---|
| Banashankari | Banashankari | वन शंकरी *(Van Shankari)* | वन शंकरी *(Van Shankari)* |
| Doddanekundi | Doddanekundi | दो धन्य कुंडली *(Do Dhanye Kundali)* | दोधन नेकुंडी *(Dodhan Nekundi)* |
| Marathahalli | Marathahalli | मारा था हल्ली *(Mara Tha Halli)* | मारा था हल्ली |
| Thanisandra | Thanisandra | सनी संतरा *(Sunny Santra)* | धनि संदरा *(Dhani Sandra)* |
| Rajajinagar | Rajajinagar | राजाजी नगर *(Rajaji Nagar)* | राजाजी नगर |

The failure pattern is systematic: both models over-segment Kannada place names into shorter, Hindi-sounding substrings. Notably, both models produce **different wrong outputs** for the same input (see Doddanekundi, Thanisandra) , meaning the failure is probabilistic, not deterministic, making it harder to patch with simple rules. This points firmly toward a fuzzy-match NER post-processing layer as the most practical near-term fix.

### Finding 4: Deepgram's advantage is largest exactly where Vahan needs it most

Deepgram's biggest WER advantages over Whisper are all in conditions Vahan encounters in production:
- `call_03_bommanahalli`: Deepgram 0.44 vs Whisper 0.78 (Δ=0.34)
- `bg_06_peenya`: Deepgram 0.56 vs Whisper 0.89 (Δ=0.33)
- `call_01_btm_layout`: Deepgram 0.60 vs Whisper 0.80 (Δ=0.20)

Whisper only beats Deepgram on two clips: `quiet_06_bellandur` (Δ=0.14) and `bg_03_silk_board` , where Deepgram timed out entirely with a 408 error. The timeout on `bg_03_silk_board` is a real reliability signal: Deepgram's API had **one inference timeout event** out of 22 clips (4.5%), which matters for a live-call environment that cannot silently drop calls.

### Finding 5: Whispered speech is a hard failure mode for both models

Both models score WER ≥ 1.04 on whispered clips and correctly identified zero locality names. At this performance level, whispered speech requires a different approach , i.e: either a dedicated whisper-detection model that routes to a specialized ASR, or prompting the caller to repeat louder before processing.

---

## 4. Recommendation

**For Vahan's production use case, Deepgram nova-2 is the recommended baseline.**

It outperforms Whisper large-v3 on every measurable dimension: WER 0.68 vs 0.78, entity accuracy 18.2% vs 13.6%, and latency 3.7s vs 24.8s on CPU. Critically, it is the only model with operationally viable real-time latency — Whisper at 24s per clip cannot serve a live phone call. On GPU, Whisper drops to ~2s, but that introduces infrastructure cost Deepgram's API avoids.

However, 18.2% entity accuracy remains too low for a real-world deployment that depends on reliable locality extraction. Three improvements in order of effort vs. impact:

1. **Post-processing NER fuzzy-match layer** — match raw transcript against a known Bangalore locality list using Levenshtein distance ≤ 2. This would catch "Van Shankari" → "Banashankari" and "Rajaji Nagar" → "Rajajinagar" without any model retraining, likely improving usable locality recovery significantly.

2. **Fine-tune on Bangalore locality names** — even a few hundred examples of these specific place names in Hinglish would eliminate the systematic over-segmentation errors identified in Finding 3.

3. **Handle the Deepgram timeout** — the 408 on `bg_03_silk_board` suggests audio preprocessing (trimming silence, normalizing volume) before API submission would improve reliability.

---

## 5. Limitations

- **Single speaker:** All 22 recordings are from one person. Real-world variance in age, dialect, accent, and gender is not captured. Entity accuracy would likely be lower on a diverse speaker set.
- **IndicWhisper not evaluated:** Could not be loaded due to inaccessible model identifiers. Remains an important candidate as it was specifically designed to handle Indian place names and could outperform both evaluated models on entity accuracy.
- **Simulated phone calls:** OGG compression approximates VOIP quality but misses real artifacts like packet loss, echo, and codec transitions present in actual Vahan calls.
- **Small test set:** 22 clips across 22 localities means each locality is tested exactly once. A robust benchmark requires multiple speakers and multiple takes per locality to separate model performance from recording variance.
- **No GPU for Whisper:** Latency of ~25s/clip was measured on CPU. On a T4 GPU this drops to ~1–2s, making Whisper significantly more production-viable than these numbers alone suggest for GPU-enabled deployments.
- **Entity accuracy metric is strict:** The current binary exact-match check underestimates practical utility — "Rajaji Nagar" for "Rajajinagar" would fail the check despite being recognisably correct. A fuzzy entity match would give a more realistic picture of usable accuracy.
