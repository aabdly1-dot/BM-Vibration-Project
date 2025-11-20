# Signal Processing & Classification Repository

This repository contains the signal processing pipeline, data collection structure, and machine learning models for the vibration analysis project.

## Repository Structure

```text
signal/
│
├── 00_research/
│   ├── notes_transfer_function.md
│   ├── summary_previous_projects.md
│   └── README.md
│
├── 01_data_collection/
│   ├── raw/
│   │   ├── tool_drill/
│   │   ├── tool_grinder/
│   │   ├── tool_saw/
│   │   ├── noise_walking/
│   │   ├── noise_stairs/
│   │   └── noise_arm_swing/
│   ├── metadata/
│   │   └── dataset_info.json
│   └── README.md
│
├── 02_preprocessing/
│   ├── notebooks/
│   │   ├── filtering_experiments.ipynb
│   │   └── gravity_removal.ipynb
│   ├── scripts/
│   │   ├── highpass_filter.py
│   │   ├── preprocess_pipeline.py
│   │   └── segmentation.py
│   └── README.md
│
├── 03_classifiers/
│   ├── on_off/
│   │   ├── train_onoff.ipynb
│   │   ├── onoff_model.py
│   │   ├── thresholds.json
│   │   └── README.md
│   ├── tool_type/
│   │   ├── train_tooltype.ipynb
│   │   ├── fft_feature_extract.py
│   │   ├── tool_model.pkl
│   │   └── notes.txt
│   └── README.md
│
├── 04_validation/
│   ├── results/
│   │   ├── drill_validation.csv
│   │   ├── grinder_validation.csv
│   │   ├── saw_validation.csv
│   │   └── summary_results.xlsx
│   ├── test_protocol.md
│   ├── validation_script.py
│   └── README.md
│
├── interface/
│   ├── api_contract.md
│   ├── example_output.json
│   └── minimal_c_version.c
│
└── README.md
