# 🏗️ BM-Vibration Project Architecture

## 📋 Overview

The BM-Vibration project is a power tool vibration classification system using Movesense accelerometer data. The architecture consists of **two parallel branches**:

1. **Signal Processing Pipeline** — Production ML pipeline for vibration classification
2. **Data Analysis Branch** — Exploratory dataset analysis and visualization

---

## 🗂️ High-Level Structure

```
BM-Vibration/
│
├── signal/                    # 🔧 Production Pipeline (ML Classification)
│   ├── 00_research/           # Research notes
│   ├── 01_data_collection/    # Raw training data (JSON)
│   │   └── raw/
│   │       ├── tool_drill/    # Drill vibration recordings
│   │       ├── tool_grinder/  # Grinder vibration recordings
│   │       ├── noise_walking/ # Walking noise recordings
│   │       └── noise_stairs/  # Stair climbing recordings
│   ├── 02_preprocessing/      # Filtering and segmentation
│   ├── 03_classifiers/        # Classification models
│   │   ├── on_off/            # ON/OFF vibration detector
│   │   └── tool_type/         # Tool type identifier (WIP)
│   ├── 04_validation/         # Validation suite
│   └── main_pipline.py        # 🎯 PIPELINE ORCHESTRATOR
│
├── data_analysis/             # 🔬 Exploratory Analysis Branch
│   ├── [27 dataset folders]/  # Per-dataset analysis results
│   └── automated_labeling_spectrogram.py
│
├── utils/                     # 🛠️ Shared Utilities
│   ├── loader_vizualizer_FFT_Welch.py  # JSON loader + FFT/Welch
│   └── data_loader.py         # UCI HAR format loader
│
├── data/                      # 📦 Source Movesense data (.json, .csv)
├── reports/                   # 📊 Final reports
└── docs/                      # 📚 Documentation
```

---

## 🔄 Pipeline Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          BM-VIBRATION ARCHITECTURE                                   │
│                                                                                      │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║                    BRANCH 1: SIGNAL PROCESSING PIPELINE                        ║  │
│  ║                         (Production ML Workflow)                               ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                      │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐                   │
│   │     DATA     │    │   PREPROCESSING  │    │   CLASSIFIERS   │                   │
│   │  COLLECTION  │───▶│     (02_)        │───▶│      (03_)      │                   │
│   │    (01_)     │    │                  │    │                 │                   │
│   └──────────────┘    └──────────────────┘    └─────────────────┘                   │
│         │                     │                       │                              │
│         │                     │                       │                              │
│         ▼                     ▼                       ▼                              │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐                   │
│   │ Movesense    │    │ Highpass Filter  │    │ Multi-Feature   │                   │
│   │ JSON Files   │    │ (0.5 Hz cutoff)  │    │ ON/OFF Model    │                   │
│   │              │    │                  │    │                 │                   │
│   │ • tool_drill │    │ Bandpass Filter  │    │ • RMS           │                   │
│   │ • tool_grind │    │ (40-400 Hz)      │    │ • Centroid      │                   │
│   │ • noise_walk │    │                  │    │ • HF Ratio      │                   │
│   │ • noise_stair│    │ Segmentation     │    │ • Flatness      │                   │
│   └──────────────┘    │ (64 samples,50%) │    │ • Crest Factor  │                   │
│                       └──────────────────┘    └─────────────────┘                   │
│                                │                       │                            │
│                                │                       │                            │
│                                ▼                       ▼                            │
│                        ┌───────────────────────────────────────┐                    │
│                        │           VALIDATION (04_)            │                    │
│                        │    comprehensive_validation.py        │                    │
│                        │                                       │                    │
│                        │  • Confusion Matrix                   │                    │
│                        │  • Classification Report              │                    │
│                        │  • Per-Dataset Accuracy               │                    │
│                        │  • Feature Distribution Analysis      │                    │
│                        │  • Error Analysis                     │                    │
│                        └───────────────────────────────────────┘                    │
│                                         │                                           │
│                                         ▼                                           │
│                               ┌─────────────────┐                                   │
│                               │    REPORTS/     │                                   │
│                               │  Final Results  │                                   │
│                               └─────────────────┘                                   │
│                                                                                      │
│  ════════════════════════════════════════════════════════════════════════════════   │
│                                                                                      │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║                    BRANCH 2: DATA ANALYSIS (Exploration)                       ║  │
│  ║                         (Research & Visualization)                             ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                      │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐                   │
│   │   data/*.json│    │     UTILS/       │    │  data_analysis/ │                   │
│   │  (Raw JSON)  │───▶│                  │───▶│   Per-Dataset   │                   │
│   │              │    │ loader_FFT_Welch │    │    Analysis     │                   │
│   └──────────────┘    │ data_loader      │    └─────────────────┘                   │
│                       └──────────────────┘             │                            │
│                                                        │                            │
│                                ┌───────────────────────┼───────────────────────┐    │
│                                │                       │                       │    │
│                                ▼                       ▼                       ▼    │
│                        ┌──────────────┐       ┌──────────────┐       ┌────────────┐ │
│                        │  EMD/CEEMDAN │       │ Spectrogram  │       │  Bishop    │ │
│                        │   Analysis   │       │  Generation  │       │   Plots    │ │
│                        │              │       │              │       │            │ │
│                        │ emd_analysis │       │ automated_   │       │ (Combined  │ │
│                        │     .py      │       │ labeling_    │       │  Visual)   │ │
│                        └──────────────┘       │ spectrogram  │       └────────────┘ │
│                                               └──────────────┘                      │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Signal Processing Pipeline (Detailed)

### Main Orchestrator: `main_pipline.py`

```python
# Execution Modes:
python signal/main_pipline.py                     # Full pipeline
python signal/main_pipline.py --visualize         # Pipeline + visualizations
python signal/main_pipline.py --visualize-only    # Only classification viz
python signal/main_pipline.py --skip-training     # Use existing model
python signal/main_pipline.py --validation-only   # Validation only
python signal/main_pipline.py --calibrate-tool grinder  # Calibrate new tool
```

### Pipeline Steps:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN PIPELINE FLOW                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: run_preprocessing() — scan_data_folders()                   │    │
│  │                                                                      │    │
│  │   01_data_collection/raw/                                           │    │
│  │   ├── tool_drill/         ─┐                                        │    │
│  │   │   └── *.json           │  ON (Tool Active)                      │    │
│  │   ├── tool_grinder/       ─┘                                        │    │
│  │   │   └── *.json                                                    │    │
│  │   ├── noise_walking/      ─┐                                        │    │
│  │   │   └── *.json           │  OFF (Non-Active)                      │    │
│  │   └── noise_stairs/       ─┘                                        │    │
│  │       └── *.json                                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: run_training() — Multi-Feature Calibration                  │    │
│  │                                                                      │    │
│  │   ┌──────────────────────────────────────────────────────┐          │    │
│  │   │       load_raw_data_from_json(json_path)             │          │    │
│  │   │       → Parse Movesense JSON, calculate Fs           │          │    │
│  │   │       → Extract ArrayAcc samples with timestamps     │          │    │
│  │   └──────────────────────────────────────────────────────┘          │    │
│  │                              │                                       │    │
│  │                              ▼                                       │    │
│  │   ┌──────────────────────────────────────────────────────┐          │    │
│  │   │       filter_triaxial_data(df, fs, apply_bandpass)   │          │    │
│  │   │       • Butterworth HPF @ 0.5 Hz (Order 4)           │          │    │
│  │   │       • Butterworth BPF @ 40-400 Hz (Order 4)        │          │    │
│  │   └──────────────────────────────────────────────────────┘          │    │
│  │                              │                                       │    │
│  │                              ▼                                       │    │
│  │   ┌──────────────────────────────────────────────────────┐          │    │
│  │   │       create_overlapping_windows(df_filtered)        │          │    │
│  │   │       • Window: 64 samples                           │          │    │
│  │   │       • Overlap: 50% (Slide Step: 32)                │          │    │
│  │   └──────────────────────────────────────────────────────┘          │    │
│  │                              │                                       │    │
│  │                              ▼                                       │    │
│  │   ┌──────────────────────────────────────────────────────┐          │    │
│  │   │       calibrate_threshold(on_windows, off_windows)   │          │    │
│  │   │       → Extract 7 spectral features per window       │          │    │
│  │   │       → Calculate optimal thresholds                 │          │    │
│  │   │       → Save to thresholds.json                      │          │    │
│  │   └──────────────────────────────────────────────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: run_validation()                                            │    │
│  │                                                                      │    │
│  │   ComprehensiveValidator                                            │    │
│  │   ├── validate_all_datasets()     → y_true, y_pred, y_probs        │    │
│  │   ├── generate_confusion_matrix() → confusion_matrix.png           │    │
│  │   ├── generate_classification_report() → report.txt                │    │
│  │   ├── plot_per_dataset_performance() → accuracy chart              │    │
│  │   ├── plot_magnitude_distribution() → feature analysis             │    │
│  │   ├── analyze_errors()            → error breakdown                 │    │
│  │   └── save_detailed_results()     → CSV exports                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4: generate_final_report()                                     │    │
│  │                                                                      │    │
│  │   04_validation/results/                                            │    │
│  │   ├── visualizations/                                               │    │
│  │   │   └── *_classification.png    (Bishop-style plots)              │    │
│  │   ├── confusion_matrix.png                                          │    │
│  │   ├── classification_report.txt                                     │    │
│  │   ├── per_dataset_accuracy.png                                      │    │
│  │   ├── magnitude_analysis.png                                        │    │
│  │   ├── dataset_results.csv                                           │    │
│  │   ├── window_results.csv                                            │    │
│  │   └── validation_summary.json                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Multi-Feature ON/OFF Classifier

### Feature Extraction (`RobustFeatureExtractor`)

```
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-FEATURE EXTRACTION                            │
│                                                                  │
│   INPUT: Window (64 samples × 3 axes)                           │
│                        │                                         │
│                        ▼                                         │
│   ┌────────────────────────────────────────────┐                │
│   │  magnitude = √(x² + y² + z²) per sample    │                │
│   └────────────────────────────────────────────┘                │
│                        │                                         │
│          ┌─────────────┴─────────────┐                          │
│          ▼                           ▼                          │
│   ┌──────────────┐           ┌───────────────┐                  │
│   │ TIME DOMAIN  │           │ FREQ DOMAIN   │                  │
│   │              │           │ (FFT)         │                  │
│   │ • RMS        │           │               │                  │
│   │ • Crest      │           │ • Centroid    │                  │
│   │   Factor     │           │ • Bandwidth   │                  │
│   └──────────────┘           │ • Flatness    │                  │
│                              │ • HF Ratio    │                  │
│                              │ • Peak Prom.  │                  │
│                              └───────────────┘                  │
│                        │                                         │
│                        ▼                                         │
│   ┌────────────────────────────────────────────┐                │
│   │     7 FEATURES PER WINDOW                  │                │
│   │                                            │                │
│   │  1. RMS (Root Mean Square)                 │                │
│   │  2. Spectral Centroid (Hz)                 │                │
│   │  3. Spectral Bandwidth (Hz)                │                │
│   │  4. Spectral Flatness (0-1)                │                │
│   │  5. Crest Factor (Peak/RMS)                │                │
│   │  6. HF Energy Ratio (>50Hz / Total)        │                │
│   │  7. Peak Prominence (max_peak / mean)      │                │
│   └────────────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Classification Logic (`OnOffClassifier`)

```
┌─────────────────────────────────────────────────────────────────┐
│                     ON/OFF CLASSIFIER                            │
│                                                                  │
│   INPUT: 7 Features from window                                 │
│                        │                                         │
│                        ▼                                         │
│   ┌────────────────────────────────────────────┐                │
│   │            THRESHOLD RULES                  │                │
│   │                                            │                │
│   │  rms_ok      = RMS > rms_threshold (2.0)   │                │
│   │  centroid_ok = Centroid > centroid_min (40 Hz)              │
│   │  hf_ratio_ok = HF_Ratio > hf_ratio_min (0.25)               │
│   │  flatness_ok = Flatness < flatness_max (0.5)                │
│   └────────────────────────────────────────────┘                │
│                        │                                         │
│                        ▼                                         │
│   ┌────────────────────────────────────────────┐                │
│   │           CLASSIFICATION RULE              │                │
│   │                                            │                │
│   │  is_active = rms_ok AND centroid_ok        │                │
│   │              AND hf_ratio_ok               │                │
│   └────────────────────────────────────────────┘                │
│                        │                                         │
│                        ▼                                         │
│   ┌────────────────────────────────────────────┐                │
│   │  IF is_active: VIBRATION_ON (True)         │                │
│   │  ELSE: VIBRATION_OFF (False)               │                │
│   └────────────────────────────────────────────┘                │
│                        │                                         │
│   OUTPUT: Boolean + Optional Warnings                           │
│           • LOW_FREQ_TOOL: High RMS, low centroid               │
│           • IDLE_TOOL: Low RMS, high frequency content          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Default Thresholds (`thresholds.json`)

```json
{
  "rms_threshold": 2.0,
  "centroid_min": 40.0,
  "hf_ratio_min": 0.25,
  "flatness_max": 0.5,
  "crest_factor_min": 2.0,
  "peak_prominence_min": 3.0,
  "consecutive_samples": 5,
  "tool_profiles": {}
}
```

---

## 🔬 Data Analysis Branch (Detailed)

### Analysis Workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DATA ANALYSIS WORKFLOW                                  │
│                                                                              │
│  ┌───────────────────────────┐                                              │
│  │      INPUT: data/*.json   │                                              │
│  │   (Raw Movesense JSON)    │                                              │
│  └─────────────┬─────────────┘                                              │
│                │                                                             │
│                ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │               utils/loader_vizualizer_FFT_Welch.py                   │    │
│  │                                                                      │    │
│  │   load_movesense_json()    ─────▶  DataFrame (timestamp, x, y, z)   │    │
│  │   check_signal_quality()   ─────▶  Effective Fs, DC offset          │    │
│  │   compute_fft()            ─────▶  FFT spectrum                     │    │
│  │   compute_welch_psd()      ─────▶  Welch PSD                        │    │
│  │   plot_time_series()       ─────▶  Time domain visualization        │    │
│  │   plot_spectral_analysis() ─────▶  Frequency domain analysis        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                │                                                             │
│                ├────────────────────────┬─────────────────────────┐         │
│                ▼                        ▼                         ▼         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   emd_analysis.py   │  │ automated_labeling_ │  │  Manual Analysis    │  │
│  │                     │  │  spectrogram.py     │  │  (Notebooks)        │  │
│  │  • CEEMDAN decomp.  │  │                     │  │                     │  │
│  │  • IMF extraction   │  │  • Bandpass filter  │  │  • filtering_exp.   │  │
│  │  • Freq analysis    │  │  • ST-RMS energy    │  │  • gravity_removal  │  │
│  │                     │  │  • K-means segment  │  │                     │  │
│  └──────────┬──────────┘  │  • Bishop plots     │  └─────────────────────┘  │
│             │             │  • Auto-labeling    │                           │
│             │             └──────────┬──────────┘                           │
│             │                        │                                       │
│             ▼                        ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    data_analysis/[dataset]/                          │    │
│  │                                                                      │    │
│  │   ├── processed_*.csv                    (Parsed CSV)               │    │
│  │   ├── Figure_1_Time_Series.png           (Waveform)                 │    │
│  │   ├── Figure_2_Spectral_Analysis.png     (FFT/Welch)                │    │
│  │   ├── *_BishopPlot.png                   (Combined visualization)   │    │
│  │   │                                                                  │    │
│  │   ├── EMD_results/                                                   │    │
│  │   │   ├── *_CEEMDAN_Analysis.png         (IMF decomposition)        │    │
│  │   │   └── *_IMFs.csv                     (IMF data export)          │    │
│  │   │                                                                  │    │
│  │   └── Detailed_Analysis/                                             │    │
│  │       ├── CEEMDAN_ACTIVE_Tool.png        (Active segment CEEMDAN)   │    │
│  │       ├── CEEMDAN_REST_Idle.png          (Rest segment CEEMDAN)     │    │
│  │       └── Spectrogram_ACTIVE.png         (Active-only spectrogram)  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Module Dependencies Map

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                            IMPORT DEPENDENCY GRAPH                                    │
│                                                                                       │
│                              ┌─────────────────────┐                                 │
│                              │   main_pipline.py   │                                 │
│                              │      (signal/)      │                                 │
│                              └──────────┬──────────┘                                 │
│                                         │                                            │
│              ┌──────────────────────────┼──────────────────────────┐                │
│              │                          │                          │                │
│              ▼                          ▼                          ▼                │
│   ┌──────────────────┐     ┌──────────────────────┐    ┌───────────────────┐        │
│   │ preprocess_      │     │    onoff_model.py    │    │ comprehensive_    │        │
│   │ pipeline.py      │     │   (03_classifiers)   │    │ validation.py     │        │
│   └────────┬─────────┘     └──────────┬───────────┘    └─────────┬─────────┘        │
│            │                          │                          │                 │
│   ┌────────┴────────┐        ┌────────┴────────┐        ┌────────┴────────┐        │
│   │                 │        │                 │        │                 │        │
│   ▼                 ▼        ▼                 ▼        ▼                 ▼        │
│  ┌────────────┐  ┌────────┐ ┌────────────┐ ┌────────┐ ┌────────────┐ ┌────────────┐│
│  │ highpass_  │  │segment │ │scipy.fft   │ │scipy.  │ │ sklearn    │ │ seaborn    ││
│  │ filter.py  │  │ation.py│ │scipy.signal│ │ stats  │ │ metrics    │ │ matplotlib ││
│  └────────────┘  └────────┘ └────────────┘ └────────┘ └────────────┘ └────────────┘│
│        │              │                                                            │
│        └───────┬──────┘                                                            │
│                │                                                                   │
│                ▼                                                                   │
│        ┌──────────────┐                                                           │
│        │   scipy.     │                                                           │
│        │   signal     │                                                           │
│        │   (butter,   │                                                           │
│        │   filtfilt)  │                                                           │
│        └──────────────┘                                                           │
│                                                                                    │
│  ═══════════════════════════════════════════════════════════════════════════════  │
│                                                                                    │
│                          DATA ANALYSIS BRANCH                                      │
│                                                                                    │
│   ┌──────────────────────────────────────────────────────────────────────────┐    │
│   │                                                                          │    │
│   │  ┌────────────────────┐           ┌──────────────────────────┐          │    │
│   │  │  automated_        │ ────────▶ │ loader_vizualizer_       │          │    │
│   │  │  labeling_         │           │ FFT_Welch.py             │          │    │
│   │  │  spectrogram.py    │           │        (utils/)          │          │    │
│   │  └────────────────────┘           └──────────────────────────┘          │    │
│   │           │                                   │                          │    │
│   │           │                                   │                          │    │
│   │           ▼                                   ▼                          │    │
│   │  ┌────────────────────┐           ┌──────────────────────────┐          │    │
│   │  │     PyEMD          │           │     scipy.signal         │          │    │
│   │  │    (CEEMDAN)       │           │     scipy.fft            │          │    │
│   │  └────────────────────┘           └──────────────────────────┘          │    │
│   │           │                                   │                          │    │
│   │  ┌────────┴────────────────────────────────────┤                        │    │
│   │  │                                             │                        │    │
│   │  ▼                                             ▼                        │    │
│   │  ┌────────────────────┐           ┌──────────────────────────┐          │    │
│   │  │   emd_analysis.py  │           │ sklearn.cluster.KMeans   │          │    │
│   │  │  (02_preprocessing)│           │ (for auto-segmentation)  │          │    │
│   │  └────────────────────┘           └──────────────────────────┘          │    │
│   └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Key Files Reference

| Path                                                     | Purpose                          | Used By     |
| -------------------------------------------------------- | -------------------------------- | ----------- |
| `signal/main_pipline.py`                                 | Main orchestrator                | Entry point |
| `signal/02_preprocessing/scripts/highpass_filter.py`     | HPF (0.5Hz) + BPF (40-400Hz)     | Pipeline    |
| `signal/02_preprocessing/scripts/segmentation.py`        | Sliding window (64, 50%)         | Pipeline    |
| `signal/02_preprocessing/scripts/preprocess_pipeline.py` | Load JSON → Filter → Segment     | Pipeline    |
| `signal/02_preprocessing/scripts/emd_analysis.py`        | CEEMDAN decomposition            | Analysis    |
| `signal/03_classifiers/on_off/onoff_model.py`            | Multi-feature ON/OFF classifier  | Pipeline    |
| `signal/03_classifiers/on_off/thresholds.json`           | Calibrated thresholds            | Classifier  |
| `signal/03_classifiers/tool_type/fft_feature_extract.py` | FFT features for tool ID         | Future      |
| `signal/04_validation/comprehensive_validation.py`       | Full validation suite            | Pipeline    |
| `utils/loader_vizualizer_FFT_Welch.py`                   | JSON parser + FFT/Welch          | Analysis    |
| `data_analysis/automated_labeling_spectrogram.py`        | Auto-segmentation + Bishop plots | Analysis    |

---

## 📐 Filter Parameters

### Highpass Filter

- **Type:** Butterworth IIR
- **Cutoff:** 0.5 Hz
- **Order:** 4
- **Purpose:** Remove gravity and low-frequency drift

### Bandpass Filter (Optional)

- **Type:** Butterworth IIR
- **Band:** 40-400 Hz
- **Order:** 4
- **Purpose:** Isolate tool vibration frequencies
- **Note:** Automatically skipped if Fs < 80 Hz (Nyquist limit)

### Segmentation Parameters

- **Window Size:** 64 samples
- **Overlap:** 50%
- **Slide Step:** 32 samples

---

## 🚀 Usage Examples

### Run Full Pipeline:

```bash
cd BM-Vibration/signal
python main_pipline.py
```

### Pipeline with Visualization:

```bash
python main_pipline.py --visualize
```

### Visualization Only:

```bash
python main_pipline.py --visualize-only
```

### Skip Training (use existing model):

```bash
python main_pipline.py --skip-training
```

### Validation Only:

```bash
python main_pipline.py --validation-only
```

### Calibrate New Tool:

```bash
python main_pipline.py --calibrate-tool grinder
```

This mode:

1. Scans `01_data_collection/raw/tool_{name}/` for JSON/CSV files
2. Extracts feature statistics from all windows
3. Creates tool profile in `thresholds.json`

### Run Dataset Analysis:

```bash
cd BM-Vibration/data_analysis
python automated_labeling_spectrogram.py
# Select dataset number or 'all'
```

### Run EMD Analysis:

```bash
cd BM-Vibration/signal/02_preprocessing/scripts
python emd_analysis.py
```

---

## 📈 Data Flow Summary

```
                    RAW MOVESENSE JSON
                           │
                           ▼
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
   ┌──────────────┐                 ┌──────────────┐
   │   PIPELINE   │                 │   ANALYSIS   │
   │   (signal/)  │                 │(data_analysis)│
   └──────────────┘                 └──────────────┘
          │                                 │
          ▼                                 ▼
   ┌──────────────┐                 ┌──────────────┐
   │  Highpass    │                 │ FFT, Welch   │
   │  + Bandpass  │                 │ Spectrogram  │
   │  Filters     │                 │              │
   └──────────────┘                 └──────────────┘
          │                                 │
          ▼                                 ▼
   ┌──────────────┐                 ┌──────────────┐
   │ Segmentation │                 │   CEEMDAN    │
   │ (64 samples) │                 │   (IMFs)     │
   └──────────────┘                 └──────────────┘
          │                                 │
          ▼                                 ▼
   ┌──────────────┐                 ┌──────────────┐
   │  7-Feature   │                 │ K-Means      │
   │  Extraction  │                 │ Labeling     │
   └──────────────┘                 └──────────────┘
          │                                 │
          ▼                                 ▼
   ┌──────────────┐                 ┌──────────────┐
   │ ON/OFF       │                 │ Bishop Plots │
   │ Classification│                │ Visualizations│
   └──────────────┘                 └──────────────┘
          │                                 │
          ▼                                 ▼
   ┌──────────────┐                 ┌──────────────┐
   │ Validation   │                 │ Per-Dataset  │
   │ Metrics      │                 │ Analysis     │
   └──────────────┘                 └──────────────┘
          │                                 │
          └─────────────┬───────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   REPORTS/   │
                 │  Final Docs  │
                 └──────────────┘
```

---

## 📊 Dataset Mapping

### Training Data (01_data_collection/raw/)

| Folder           | Category | Description                  |
| ---------------- | -------- | ---------------------------- |
| `tool_drill/`    | ON       | Drill vibration recordings   |
| `tool_grinder/`  | ON       | Grinder vibration recordings |
| `noise_walking/` | OFF      | Walking motion data          |
| `noise_stairs/`  | OFF      | Stair climbing data          |

### Analysis Datasets (data_analysis/)

27 analysis folders with naming convention:

- `{number}. {timestamp}_{device_id}_acc_stream/`
- `{number}. {descriptive_name}/`

Each folder contains:

- Processed CSV file
- Time series plot (Figure 1)
- Spectral analysis plot (Figure 2)
- EMD results (optional)
- Bishop plot (optional)
- Detailed analysis (optional)

---

_Last updated: December 4, 2025_
