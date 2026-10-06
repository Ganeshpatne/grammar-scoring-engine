# 🎙️ Grammar Scoring Engine AI

Automated machine learning pipeline & interactive Web UI dashboard predicting continuous grammar scores (**0.0 – 5.0**) from spoken audio recordings (45–60s WAV format) for the **SHL Hiring Assessment 2026 Kaggle Competition**.

---

## 📌 Executive Summary & Architecture Overview

Evaluating spoken grammar directly from raw audio wave signals is fundamentally incomplete because grammar rules (subject-verb agreement, syntactic depth, clause structures, vocabulary rarity) live in the **linguistic domain**, while delivery speed, hesitations, and pauses live in the **acoustic domain**.

This project implements a **Dual-Modal Hybrid Architecture (ASR + NLP + Acoustic Prosody)**:

```
                                  ┌──> Acoustic Features (MFCCs, Spectral Centroid, Pauses) ──┐
Audio WAV (45-60s) ──> Preproc ──┤                                                            ├──> 5-Fold Stratified CV Ensemble ──> Score (0-5)
                                  └──> Whisper ASR ──> Text Features (WPM, TTR, Fillers) ────┘
```

---

## 📊 Evaluation Metrics & Validation Performance

- **Evaluation Metrics**: Root Mean Squared Error (RMSE) + Pearson Correlation Coefficient (\(r\)).
- **Validation Framework**: 5-Fold Stratified Cross-Validation (targets binned into 10 quantile strata).

| Metric | Score |
| :--- | :--- |
| **Mandatory Training RMSE Score** | **`0.3046`** |
| **Validation Out-of-Fold (OOF) RMSE** | **`0.7780`** |
| **Validation Pearson Correlation (\(r\))** | **`0.7786`** |

---

## ⚙️ Key Feature Engineering Parameters

### 1. Acoustic & Prosodic Features (`src/audio_processor.py`)
- **MFCCs**: 20 Mel-Frequency Cepstral Coefficients (mean, std, and delta across frames).
- **Fluency Metrics**:
  - `duration`: Total audio length in seconds.
  - `active_speech_ratio`: Non-silent frame duration / total duration.
  - `pause_count`: Number of silent intervals (> 300ms contiguous silence below energy threshold).
  - `pauses_per_min`: Normalized pause frequency per minute.
- **Spectral Dynamics**:
  - `spectral_centroid_mean` / `std`: Timbral brightness & vocal tract resonance.
  - `zero_crossing_rate_mean` / `std`: High-frequency noise and fricative consonant ratio.
  - `rms_energy_mean` / `max` / `std`: Vocal volume dynamics.

### 2. Linguistic & Grammatical Features (`src/nlp_processor.py`)
- **Whisper ASR Speech-to-Text**: Automated transcription using OpenAI Whisper.
- **Speech Rate (WPM)**: Total word count normalized by active speaking time.
- **Type-Token Ratio (TTR)**: Unique words / total words (lexical diversity).
- **Disfluency Density**: Count and ratio of filler words (`"um"`, `"uh"`, `"er"`, `"like"`, `"you know"`).
- **Syntactic Indicators**: Punctuation density (commas, periods), sentence count, and repeated word counts.

### 3. Model Architecture & Ensemble (`src/pipeline.py`)
- **LightGBM Regressor (35%)**: `n_estimators=300, learning_rate=0.03`.
- **XGBoost Regressor (30%)**: `n_estimators=300, learning_rate=0.03, max_depth=4`.
- **CatBoost Regressor (20%)**: `n_estimators=300, learning_rate=0.03, depth=4`.
- **Ridge Regression (15%)**: `alpha=10.0` on scaled features.

---

## 🚀 Installation & Setup Instructions

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/grammar-scoring-engine.git
cd grammar-scoring-engine
pip install -r requirements.txt
```

### 2. Run Main Evaluation Pipeline
```bash
python run_acoustic_baseline.py
```

### 3. Launch Web UI Dashboard
```bash
python server.py
```
Open **`http://localhost:8000`** in your web browser.

---

## 🖥️ Web UI Dashboard Features
- **Live File Upload**: Drag and drop any `.wav` audio file.
- **Real-Time WebSockets**: Live progress events pushed step-by-step (`Step 1: Audio Loading` -> `Step 2: Acoustic Extraction` -> `Step 3: Whisper ASR` -> `Step 4: Scoring`).
- **Minimalist Aesthetic**: 680px maximum container width, 0.5px borders, flat controls, Tabler outline icons, native to Claude.ai design.
