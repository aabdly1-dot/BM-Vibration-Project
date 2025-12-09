"""
MAIN PIPELINE - Complete Vibration Analysis Workflow
Orchestrates: Data Loading -> Preprocessing -> Training -> Validation -> Reporting

Includes calibration mode for adding new tools to the classification system.
"""

import os
import sys
import argparse
from datetime import datetime
import glob

# Add all module paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '02_preprocessing', 'scripts'))
sys.path.append(os.path.join(script_dir, '03_classifiers', 'on_off'))
sys.path.append(os.path.join(script_dir, '04_validation'))

def print_banner(text):
    """Print styled banner"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80)

def print_section(text):
    """Print section header"""
    print("\n" + "-"*80)
    print(f">>> {text}")
    print("-"*80)

def scan_data_folders():
    """
    Scan the raw data folder structure and return paths to JSON files organized by category.
    
    Returns:
    - dict: {category: [list of JSON file paths]}
    """
    raw_path = os.path.join(script_dir, '01_data_collection', 'raw')
    
    # Define expected folder structure
    folder_categories = {
        'tool_drill': 'ON',
        'tool_grinder': 'ON',
        'noise_walking': 'OFF',
        'noise_stairs': 'OFF'
    }
    
    data_files = {
        'ON': [],   # Tool active (drill, grinder, etc.)
        'OFF': []   # Non-active (walking, stairs, idle)
    }
    
    print(f"[SCAN] Scanning data directory: {raw_path}")
    
    for folder_name, category in folder_categories.items():
        folder_path = os.path.join(raw_path, folder_name)
        if os.path.isdir(folder_path):
            # Find all JSON files in the folder
            json_files = glob.glob(os.path.join(folder_path, '*.json'))
            if json_files:
                data_files[category].extend(json_files)
                print(f"   [OK] {folder_name}/: {len(json_files)} JSON file(s)")
                for jf in json_files:
                    print(f"      • {os.path.basename(jf)}")
            else:
                print(f"   [WARN]  {folder_name}/: No JSON files found")
        else:
            print(f"   [ERROR] {folder_name}/: Folder not found")
    
    return data_files


def run_preprocessing():
    """Step 1: Preprocess raw data - scan for JSON files in folder structure"""
    print_section("STEP 1: DATA PREPROCESSING")
    print("Scanning for Movesense JSON data files...")
    
    from highpass_filter import BANDPASS_LOW, BANDPASS_HIGH
    
    print(f"Bandpass filter: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz\n")
    
    # Scan folders for JSON files
    data_files = scan_data_folders()
    
    total_on = len(data_files['ON'])
    total_off = len(data_files['OFF'])
    total = total_on + total_off
    
    print(f"\n[STATS] Summary:")
    print(f"   • ON (Tool Active) files: {total_on}")
    print(f"   • OFF (Non-Active) files: {total_off}")
    print(f"   • Total: {total}")
    
    if total > 0:
        print("\n[OK] Data files found and ready!")
        return True, data_files
    else:
        print("\n[ERROR] No data files found!")
        return False, data_files

def run_training(data_files=None):
    """Step 2: Train ON/OFF classifier with multi-feature analysis using JSON data"""
    print_section("STEP 2: TRAINING ON/OFF CLASSIFIER (Multi-Feature)")
    
    import numpy as np
    from preprocess_pipeline import load_raw_data_from_json
    from highpass_filter import filter_triaxial_data
    from segmentation import create_overlapping_windows
    from onoff_model import OnOffClassifier
    
    # If no data_files provided, scan for them
    if data_files is None:
        _, data_files = scan_data_folders() if 'scan_data_folders' in dir() else (False, {'ON': [], 'OFF': []})
    
    # Load ON data (tool_drill, tool_grinder)
    print("Loading ON (Tool Active) data from JSON files...")
    on_windows = []
    effective_fs = 833.0  # Default
    
    if not data_files['ON']:
        print("  [WARN]  No ON data files found!")
    else:
        for json_path in data_files['ON']:
            print(f"  • {os.path.basename(json_path)}")
            try:
                df, fs = load_raw_data_from_json(json_path)
                if df is None:
                    print(f"    [ERROR] Failed to load")
                    continue
                effective_fs = fs
                df_filtered = filter_triaxial_data(df, fs=fs, apply_bandpass=True)
                windows = create_overlapping_windows(df_filtered)
                on_windows.append(windows)
                print(f"    → {len(windows)} windows (Fs={fs:.1f}Hz)")
            except Exception as e:
                print(f"    [ERROR] Error: {e}")
    
    if len(on_windows) == 0:
        print("\n[ERROR] No ON data loaded! Cannot train.")
        return False
    
    on_windows = np.vstack(on_windows)
    print(f"  Total ON windows: {len(on_windows)}")
    
    # Load OFF data (noise_walking, noise_stairs)
    print("\nLoading OFF (Non-Active) data from JSON files...")
    off_windows = []
    
    if not data_files['OFF']:
        print("  [WARN]  No OFF data files found!")
    else:
        for json_path in data_files['OFF']:
            print(f"  • {os.path.basename(json_path)}")
            try:
                df, fs = load_raw_data_from_json(json_path)
                if df is None:
                    print(f"    [ERROR] Failed to load")
                    continue
                df_filtered = filter_triaxial_data(df, fs=fs, apply_bandpass=True)
                windows = create_overlapping_windows(df_filtered)
                off_windows.append(windows)
                print(f"    → {len(windows)} windows (Fs={fs:.1f}Hz)")
            except Exception as e:
                print(f"    [ERROR] Error: {e}")
    
    if len(off_windows) == 0:
        print("\n[ERROR] No OFF data loaded! Cannot train.")
        return False
    
    off_windows = np.vstack(off_windows)
    print(f"  Total OFF windows: {len(off_windows)}")
    
    # Use existing thresholds (no calibration, no overwriting)
    print("\n[INFO] Using existing thresholds from thresholds.json (no calibration)")
    classifier = OnOffClassifier(fs=effective_fs)  # Loads from thresholds.json
    
    # Show current thresholds
    print(f"  Current thresholds:")
    print(f"    • RMS threshold: {classifier.thresholds['rms_threshold']}")
    print(f"    • Centroid min: {classifier.thresholds['centroid_min']} Hz")
    print(f"    • Flatness max: {classifier.thresholds['flatness_max']}")
    
    # Validate on loaded data (without saving)
    on_correct = sum(classifier.predict_batch(on_windows))
    off_correct = sum(1 for p in classifier.predict_batch(off_windows) if not p)
    
    on_acc = 100 * on_correct / len(on_windows) if len(on_windows) > 0 else 0
    off_acc = 100 * off_correct / len(off_windows) if len(off_windows) > 0 else 0
    
    print(f"\n[STATS] Validation with current thresholds:")
    print(f"    • ON accuracy: {on_acc:.1f}% ({on_correct}/{len(on_windows)})")
    print(f"    • OFF accuracy: {off_acc:.1f}% ({off_correct}/{len(off_windows)})")
    print(f"    • Overall: {(on_acc + off_acc) / 2:.1f}%")
    
    print(f"\n[OK] Thresholds validated (NOT overwritten)")
    
    return True

def run_validation():
    """Step 3: Comprehensive validation"""
    print_section("STEP 3: COMPREHENSIVE VALIDATION")
    print("Running validation suite...")
    
    # Import and run comprehensive validation
    import comprehensive_validation
    
    validator = comprehensive_validation.ComprehensiveValidator()
    
    # Run all validation steps
    y_true, y_pred, y_probs = validator.validate_all_datasets()
    metrics = validator.generate_confusion_matrix(y_true, y_pred)
    validator.generate_classification_report(y_true, y_pred)
    validator.plot_per_dataset_performance()
    validator.plot_magnitude_distribution()
    validator.analyze_errors()
    validator.save_detailed_results()
    validator.generate_summary_report(metrics)
    
    print("\n[OK] Validation complete!")
    return metrics


def visualize_classification(data_files=None, save_plots=True):
    """
    Visualize automatic labeling results based on threshold rules.
    
    Creates Bishop-style plots showing:
    - Waveform (magnitude over time)
    - Classification labels (ACTIVE/REST per window)
    - Feature values that determined the classification
    
    Parameters:
    - data_files (dict): Dictionary with 'ON' and 'OFF' file lists
    - save_plots (bool): Whether to save plots to results folder
    """
    print_section("VISUALIZATION: Automatic Labeling Results")
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from preprocess_pipeline import load_raw_data_from_json
    from highpass_filter import filter_triaxial_data
    from segmentation import create_overlapping_windows, WINDOW_SIZE, SLIDE_STEP
    from onoff_model import OnOffClassifier, RobustFeatureExtractor
    
    # Create results directory
    results_dir = os.path.join(script_dir, '04_validation', 'results', 'visualizations')
    os.makedirs(results_dir, exist_ok=True)
    
    # If no data_files provided, scan for them
    if data_files is None:
        data_files = scan_data_folders()
    
    # Initialize classifier
    classifier = OnOffClassifier()
    
    # Process each file
    all_files = data_files.get('ON', []) + data_files.get('OFF', [])
    
    if not all_files:
        print("[ERROR] No data files found for visualization!")
        return
    
    for json_path in all_files:
        filename = os.path.basename(json_path)
        folder_name = os.path.basename(os.path.dirname(json_path))
        expected_label = "ON" if "tool_" in folder_name else "OFF"
        
        print(f"\n[STATS] Processing: {folder_name}/{filename}")
        print(f"   Expected: {expected_label}")
        
        try:
            # Load and process data
            df, fs = load_raw_data_from_json(json_path)
            if df is None:
                print(f"   [ERROR] Failed to load")
                continue
            
            classifier.fs = fs
            classifier.extractor.fs = fs
            
            # Apply filtering
            df_filtered = filter_triaxial_data(df, fs=fs, apply_bandpass=True)
            
            # Create windows
            windows = create_overlapping_windows(df_filtered)
            
            if len(windows) == 0:
                print(f"   [ERROR] No windows created")
                continue
            
            # Classify each window and extract features
            predictions, all_features = classifier.predict_batch(windows, verbose=True)
            
            # Calculate time axis
            n_samples = len(df)
            time_seconds = np.arange(n_samples) / fs
            
            # Calculate window center times
            window_times = []
            for i in range(len(windows)):
                start_sample = i * SLIDE_STEP
                center_sample = start_sample + WINDOW_SIZE // 2
                window_times.append(center_sample / fs)
            window_times = np.array(window_times)
            
            # Calculate magnitude
            magnitude = np.sqrt(
                df_filtered['accel_x_filtered'].values**2 + 
                df_filtered['accel_y_filtered'].values**2 + 
                df_filtered['accel_z_filtered'].values**2
            )
            
            # Extract feature arrays
            rms_values = [f['rms'] for f in all_features]
            centroid_values = [f['spectral_centroid'] for f in all_features]
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            fig.suptitle(f'Automatic Classification: {folder_name}/{filename}\n'
                        f'Fs={fs:.1f}Hz | {len(windows)} windows | Expected: {expected_label}',
                        fontsize=12, fontweight='bold')
            
            # --- Subplot 1: Waveform ---
            ax1 = axes[0]
            ax1.plot(time_seconds, magnitude, linewidth=0.5, color='steelblue', alpha=0.8)
            ax1.set_ylabel('Magnitude (m/s²)')
            ax1.set_title('Filtered Signal Magnitude')
            ax1.grid(True, alpha=0.3)
            
            # --- Subplot 2: Classification Labels ---
            ax2 = axes[1]
            
            # Create color array for windows
            colors = ['#2ecc71' if p else '#e74c3c' for p in predictions]  # Green=ACTIVE, Red=REST
            
            # Plot as colored bars
            for i, (t, pred) in enumerate(zip(window_times, predictions)):
                color = '#2ecc71' if pred else '#e74c3c'
                width = (WINDOW_SIZE / fs) * 0.9
                ax2.barh(0.5, width, left=t - width/2, height=0.8, color=color, alpha=0.7)
            
            ax2.set_ylim(0, 1)
            ax2.set_yticks([0.5])
            ax2.set_yticklabels(['State'])
            ax2.set_title('Classification: ACTIVE (green) / REST (red)')
            
            # Add legend
            legend_elements = [
                Patch(facecolor='#2ecc71', alpha=0.7, label='ACTIVE (ON)'),
                Patch(facecolor='#e74c3c', alpha=0.7, label='REST (OFF)')
            ]
            ax2.legend(handles=legend_elements, loc='upper right')
            
            # --- Subplot 3: Feature Values (RMS and Centroid) ---
            ax3 = axes[2]
            
            ax3_rms = ax3
            ax3_rms.plot(window_times, rms_values, 'b-o', markersize=3, label='RMS', linewidth=1)
            ax3_rms.axhline(y=classifier.thresholds['rms_threshold'], color='b', linestyle='--', 
                           alpha=0.5, label=f"RMS thresh ({classifier.thresholds['rms_threshold']:.1f})")
            ax3_rms.set_ylabel('RMS (m/s²)', color='blue')
            ax3_rms.tick_params(axis='y', labelcolor='blue')
            
            ax3_cent = ax3.twinx()
            ax3_cent.plot(window_times, centroid_values, 'r-s', markersize=3, label='Centroid', linewidth=1)
            ax3_cent.axhline(y=classifier.thresholds['centroid_min'], color='r', linestyle='--', 
                            alpha=0.5, label=f"Centroid thresh ({classifier.thresholds['centroid_min']:.0f}Hz)")
            ax3_cent.set_ylabel('Spectral Centroid (Hz)', color='red')
            ax3_cent.tick_params(axis='y', labelcolor='red')
            
            ax3.set_title('Feature Values vs Thresholds')
            ax3.grid(True, alpha=0.3)
            
            # Combined legend
            lines1, labels1 = ax3_rms.get_legend_handles_labels()
            lines2, labels2 = ax3_cent.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
            
            # --- Subplot 4 removed (HF ratio) ---
            ax4 = axes[2]
            ax4.set_xlabel('Time (seconds)')
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            if save_plots:
                safe_filename = filename.replace('.json', '').replace(' ', '_')
                output_path = os.path.join(results_dir, f'{folder_name}_{safe_filename}_classification.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"   [SAVED] Saved: {output_path}")
            
            plt.show()
            
            # Print classification summary
            n_active = sum(predictions)
            n_rest = len(predictions) - n_active
            print(f"   [RESULT] Results: {n_active} ACTIVE ({100*n_active/len(predictions):.1f}%) | "
                  f"{n_rest} REST ({100*n_rest/len(predictions):.1f}%)")
            
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[OK] Visualization complete! Plots saved to: {results_dir}")

def run_calibration(tool_name):
    """
    Calibrate thresholds for a new tool type.
    
    This mode allows users to add new tools to the classification system by
    recording ~10 seconds of data with the new tool and computing feature statistics.
    Supports both JSON (Movesense) and CSV files.
    
    Parameters:
    - tool_name (str): Name of the new tool (e.g., 'grinder', 'saw', 'impact_driver').
    """
    print_section(f"CALIBRATION MODE: {tool_name.upper()}")
    
    import numpy as np
    import json as json_lib
    from preprocess_pipeline import load_raw_data_from_json, load_raw_data_from_csv
    from highpass_filter import filter_triaxial_data
    from segmentation import create_overlapping_windows
    from onoff_model import calibrate_tool_profile, THRESHOLDS_FILE
    
    raw_path = os.path.join(script_dir, '01_data_collection', 'raw')
    
    print(f"\n[INFO] Calibration Instructions:")
    print(f"1. Record ~10 seconds of {tool_name} in use (actively touching workpiece)")
    print(f"2. Save as JSON or CSV in: {raw_path}/tool_{tool_name}/")
    print(f"   Or name the file: calibration_{tool_name}.json/.csv")
    
    # Search in tool-specific subfolders
    tool_folders = [
        os.path.join(raw_path, f'tool_{tool_name}'),
        os.path.join(raw_path, tool_name),
    ]
    
    json_files = []
    csv_files = []
    
    # Check direct calibration files
    for ext in ['.json', '.csv']:
        calibration_file = os.path.join(raw_path, f'calibration_{tool_name}{ext}')
        if os.path.exists(calibration_file):
            if ext == '.json':
                json_files.append(calibration_file)
            else:
                csv_files.append(calibration_file)
    
    # Check tool folders for JSON and CSV files
    for folder in tool_folders:
        if os.path.isdir(folder):
            json_files.extend(glob.glob(os.path.join(folder, '*.json')))
            csv_files.extend(glob.glob(os.path.join(folder, '*.csv')))
    
    total_files = len(json_files) + len(csv_files)
    
    if total_files == 0:
        print(f"\n[ERROR] No calibration data found!")
        print(f"   Expected JSON/CSV files in: {tool_folders}")
        return False
    
    print(f"\n[DIR] Found calibration files:")
    if json_files:
        print(f"   JSON files ({len(json_files)}):")
        for f in json_files:
            print(f"      • {os.path.basename(f)}")
    if csv_files:
        print(f"   CSV files ({len(csv_files)}):")
        for f in csv_files:
            print(f"      • {os.path.basename(f)}")
    
    # Load and process all calibration data
    all_windows = []
    effective_fs = 833.0
    
    # Process JSON files first (preferred for Movesense data)
    for json_file in json_files:
        print(f"\n[LOAD] Processing: {os.path.basename(json_file)}")
        try:
            df, fs = load_raw_data_from_json(json_file)
            if df is None:
                continue
            effective_fs = fs
            df_filtered = filter_triaxial_data(df, fs=fs, apply_bandpass=True)
            windows = create_overlapping_windows(df_filtered)
            all_windows.append(windows)
            print(f"   → {len(windows)} windows extracted (Fs={fs:.1f}Hz)")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
    
    # Process CSV files
    for csv_file in csv_files:
        print(f"\n[LOAD] Processing: {os.path.basename(csv_file)}")
        try:
            df, fs = load_raw_data_from_csv(csv_file)
            effective_fs = fs
            df_filtered = filter_triaxial_data(df, fs=fs, apply_bandpass=True)
            windows = create_overlapping_windows(df_filtered)
            all_windows.append(windows)
            print(f"   → {len(windows)} windows extracted (Fs={fs:.1f}Hz)")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
    
    if len(all_windows) == 0:
        print("\n[ERROR] No valid windows extracted from calibration data!")
        return False
    
    all_windows = np.vstack(all_windows)
    print(f"\n[STATS] Total windows for calibration: {len(all_windows)}")
    
    # Generate tool profile
    profile = calibrate_tool_profile(all_windows, tool_name, fs=effective_fs)
    
    # Load existing thresholds and add tool profile
    if os.path.exists(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE, 'r') as f:
            thresholds = json_lib.load(f)
    else:
        thresholds = {
            'rms_threshold': 2.0,
            'centroid_min': 40.0,
            'flatness_max': 0.5,
            'crest_factor_min': 2.0,
            'peak_prominence_min': 3.0,
            'consecutive_samples': 5,
            'tool_profiles': {}
        }
    
    # Add/update tool profile
    thresholds['tool_profiles'][tool_name] = profile
    
    # Save updated thresholds
    with open(THRESHOLDS_FILE, 'w') as f:
        json_lib.dump(thresholds, f, indent=2)
    
    print(f"\n[OK] Tool profile '{tool_name}' saved to thresholds.json")
    print(f"   Key characteristics:")
    print(f"   • RMS: {profile['features']['rms']['mean']:.2f} ± {profile['features']['rms']['std']:.2f}")
    print(f"   • Spectral Centroid: {profile['features']['spectral_centroid']['mean']:.1f} Hz")
    print(f"   • HF Energy Ratio: {profile['features']['hf_energy_ratio']['mean']:.3f}")
    
    return True


def classify_feature_csv(csv_path, save_results=True, visualize=True):
    """
    Classify pre-extracted feature dataset using threshold rules.
    
    This function handles CSV files that already contain computed features
    (RMS, Spectral Centroid, etc.) and applies the threshold-based classification
    rules to determine ACTIVE/NON-ACTIVE status.
    
    Parameters:
    - csv_path (str): Path to the CSV file with pre-extracted features.
    - save_results (bool): Whether to save classified CSV and metrics.
    - visualize (bool): Whether to show visualization plots.
    
    Expected CSV columns:
    - rms, spectral_centroid, spectral_bandwidth, spectral_flatness
    - crest_factor, hf_energy_ratio, peak_prominence
    - window_id, label (ground truth tool name)
    
    Label mapping (ground truth):
    - ACTIVE (ON): plsh, drill, pmf, psm (all tools)
    - NON-ACTIVE (OFF): walking
    """
    print_section("CSV FEATURE DATASET CLASSIFICATION")
    
    import pandas as pd
    import numpy as np
    from onoff_model import OnOffClassifier
    
    # Label mapping: tools → ACTIVE, walking → NON-ACTIVE
    LABEL_MAP = {
        'plsh': True,    # Tool: ACTIVE
        'drill': True,   # Tool: ACTIVE
        'pmf': True,     # Tool: ACTIVE
        'psm': True,     # Tool: ACTIVE
        'walking': False # Walking: NON-ACTIVE
    }
    
    # Load CSV
    print(f"[LOAD] Loading CSV: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} windows")
    print(f"   Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['rms', 'spectral_centroid', 'label']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns: {missing_cols}")
        return False
    
    # Show label distribution
    print(f"\n[INFO] Label distribution:")
    for label, count in df['label'].value_counts().items():
        gt = "ACTIVE" if LABEL_MAP.get(label, False) else "NON-ACTIVE"
        print(f"   * {label}: {count} windows -> {gt}")
    
    # Initialize classifier (loads thresholds from file)
    classifier = OnOffClassifier()
    print(f"\n[INFO] Using thresholds:")
    print(f"   * RMS threshold: {classifier.thresholds['rms_threshold']}")
    print(f"   * Centroid min: {classifier.thresholds['centroid_min']} Hz")
    print(f"   * Flatness max: {classifier.thresholds['flatness_max']}")
    
    # Classify each window using pre-extracted features
    print(f"\n[CLASSIFY] Applying threshold rules to {len(df)} windows...")
    predictions = []
    
    for idx, row in df.iterrows():
        features = {
            'rms': row['rms'],
            'spectral_centroid': row['spectral_centroid'],
            'spectral_flatness': row.get('spectral_flatness', 0.5),
            'crest_factor': row.get('crest_factor', 1.0),
            'peak_prominence': row.get('peak_prominence', 1.0)
        }
        is_active = classifier.predict_from_features(features, verbose=False)
        predictions.append(is_active)
    
    df['predicted_active'] = predictions
    df['predicted_label'] = df['predicted_active'].apply(lambda x: 'ACTIVE' if x else 'NON-ACTIVE')
    
    # Add ground truth column
    df['ground_truth_active'] = df['label'].apply(lambda x: LABEL_MAP.get(x, False))
    df['ground_truth_label'] = df['ground_truth_active'].apply(lambda x: 'ACTIVE' if x else 'NON-ACTIVE')
    
    # Calculate metrics
    print(f"\n[METRICS] Classification Results:")
    
    y_true = df['ground_truth_active'].values
    y_pred = df['predicted_active'].values
    
    # Confusion matrix components
    TP = np.sum((y_true == True) & (y_pred == True))
    TN = np.sum((y_true == False) & (y_pred == False))
    FP = np.sum((y_true == False) & (y_pred == True))
    FN = np.sum((y_true == True) & (y_pred == False))
    
    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    print(f"   Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    ACTIVE  NON-ACTIVE")
    print(f"   Actual ACTIVE    {TP:5d}   {FN:5d}")
    print(f"   Actual NON-ACTIVE{FP:5d}   {TN:5d}")
    print(f"")
    print(f"   Accuracy:    {accuracy*100:.2f}%")
    print(f"   Precision:   {precision*100:.2f}%")
    print(f"   Recall:      {recall*100:.2f}%")
    print(f"   F1-Score:    {f1*100:.2f}%")
    print(f"   Specificity: {specificity*100:.2f}%")
    
    # Per-tool accuracy
    print(f"\n[STATS] Per-Tool Classification:")
    for label in df['label'].unique():
        tool_df = df[df['label'] == label]
        tool_gt = LABEL_MAP.get(label, False)
        tool_correct = sum(tool_df['predicted_active'] == tool_gt)
        tool_acc = 100 * tool_correct / len(tool_df) if len(tool_df) > 0 else 0
        expected = "ACTIVE" if tool_gt else "NON-ACTIVE"
        print(f"   * {label}: {tool_acc:.1f}% correct ({tool_correct}/{len(tool_df)}) - Expected: {expected}")
    
    # Save results
    if save_results:
        results_dir = os.path.join(script_dir, '04_validation', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save classified CSV
        output_csv = os.path.join(results_dir, 'tools_dataset_classified.csv')
        df.to_csv(output_csv, index=False)
        print(f"\n[SAVED] Classified CSV: {output_csv}")
        
        # Save metrics to JSON
        import json
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'confusion_matrix': {'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)},
            'total_windows': len(df),
            'thresholds_used': classifier.thresholds
        }
        metrics_file = os.path.join(results_dir, 'tools_dataset_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAVED] Metrics JSON: {metrics_file}")
    
    # Visualize
    if visualize:
        visualize_csv_classification(df, classifier, csv_path, save_results)
    
    print(f"\n[OK] CSV classification complete!")
    return True


def visualize_csv_classification(df, classifier, csv_path, save_plots=True):
    """
    Visualize classification results from a pre-extracted feature CSV.
    
    Creates a 3-subplot figure showing:
    - RMS values per window with threshold line
    - Classification bars (predicted vs ground truth)
    - Spectral Centroid values with threshold line
    - HF Energy Ratio values with threshold line
    
    Parameters:
    - df (pd.DataFrame): DataFrame with features and predictions.
    - classifier (OnOffClassifier): Classifier with thresholds.
    - csv_path (str): Original CSV path (for title).
    - save_plots (bool): Whether to save the plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    
    print(f"\n[PLOT] Generating visualization...")
    
    # Group data by tool label for clearer visualization
    labels = df['label'].unique()
    label_boundaries = []
    current_idx = 0
    
    for label in labels:
        count = len(df[df['label'] == label])
        label_boundaries.append((current_idx, current_idx + count, label))
        current_idx += count
    
    # Window indices
    window_ids = np.arange(len(df))
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'CSV Feature Classification: {os.path.basename(csv_path)}\n'
                f'{len(df)} windows | Thresholds: RMS>{classifier.thresholds["rms_threshold"]:.1f}, '
                f'Centroid>{classifier.thresholds["centroid_min"]:.0f}Hz, '
                f'Flatness<{classifier.thresholds["flatness_max"]:.2f}',
                fontsize=12, fontweight='bold')
    
    # --- Subplot 1: RMS Values ---
    ax1 = axes[0]
    ax1.plot(window_ids, df['rms'].values, 'b-', linewidth=0.8, alpha=0.8)
    ax1.axhline(y=classifier.thresholds['rms_threshold'], color='r', linestyle='--', 
               linewidth=2, label=f"Threshold ({classifier.thresholds['rms_threshold']:.1f})")
    ax1.set_ylabel('RMS (m/s²)')
    ax1.set_title('RMS Values per Window')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add tool label separators
    for start, end, label in label_boundaries:
        ax1.axvline(x=start, color='gray', linestyle=':', alpha=0.5)
        ax1.text((start + end) / 2, ax1.get_ylim()[1] * 0.95, label, 
                ha='center', va='top', fontsize=9, fontweight='bold')
    
    # --- Subplot 2: Classification Labels (Predicted vs Ground Truth) ---
    ax2 = axes[1]
    
    # Plot predicted labels (top row)
    for i, pred in enumerate(df['predicted_active']):
        color = '#2ecc71' if pred else '#e74c3c'  # Green=ACTIVE, Red=NON-ACTIVE
        ax2.barh(0.7, 1, left=i, height=0.4, color=color, alpha=0.8)
    
    # Plot ground truth labels (bottom row)
    for i, gt in enumerate(df['ground_truth_active']):
        color = '#27ae60' if gt else '#c0392b'  # Darker green/red for ground truth
        ax2.barh(0.2, 1, left=i, height=0.4, color=color, alpha=0.8)
    
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.7])
    ax2.set_yticklabels(['Ground Truth', 'Predicted'])
    ax2.set_title('Classification: ACTIVE (green) / NON-ACTIVE (red)')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='Predicted ACTIVE'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Predicted NON-ACTIVE'),
        Patch(facecolor='#27ae60', alpha=0.8, label='GT ACTIVE'),
        Patch(facecolor='#c0392b', alpha=0.8, label='GT NON-ACTIVE')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8)
    
    # Add tool separators
    for start, end, label in label_boundaries:
        ax2.axvline(x=start, color='gray', linestyle=':', alpha=0.5)
    
    # --- Subplot 3: Spectral Centroid ---
    ax3 = axes[2]
    ax3.plot(window_ids, df['spectral_centroid'].values, 'r-', linewidth=0.8, alpha=0.8)
    ax3.axhline(y=classifier.thresholds['centroid_min'], color='b', linestyle='--', 
               linewidth=2, label=f"Threshold ({classifier.thresholds['centroid_min']:.0f} Hz)")
    ax3.set_ylabel('Spectral Centroid (Hz)')
    ax3.set_title('Spectral Centroid per Window')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Add tool separators
    for start, end, label in label_boundaries:
        ax3.axvline(x=start, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    if save_plots:
        results_dir = os.path.join(script_dir, '04_validation', 'results', 'visualizations')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'tools_dataset_classification.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Visualization: {output_path}")
    
    plt.show()
    print(f"[OK] Visualization displayed!")


def generate_final_report(metrics, start_time):
    """Step 4: Generate final summary report"""
    print_banner("PIPELINE EXECUTION SUMMARY")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n[TIME]  Execution Time: {duration:.2f} seconds")
    print(f"[DATE] Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n[STATS] Final Results:")
    print(f"  • Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  • Precision:   {metrics['precision']*100:.2f}%")
    print(f"  • Recall:      {metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:    {metrics['f1_score']*100:.2f}%")
    print(f"  • Specificity: {metrics['specificity']*100:.2f}%")
    
    print("\n[FILES] Generated Artifacts:")
    results_dir = os.path.join(script_dir, '04_validation', 'results')
    print(f"  • Location: {results_dir}")
    print(f"  • Confusion Matrix (PNG)")
    print(f"  • Classification Report (TXT)")
    print(f"  • Per-Dataset Accuracy (PNG)")
    print(f"  • Magnitude Analysis (PNG)")
    print(f"  • Dataset Results (CSV)")
    print(f"  • Window Results (CSV)")
    print(f"  • Validation Summary (JSON)")
    
    print("\n" + "="*80)
    print("[OK] PIPELINE COMPLETE!".center(80))
    print("="*80 + "\n")

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='Complete Vibration Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_pipline.py                     # Run full pipeline
  python main_pipline.py --visualize         # Run pipeline + show visualizations
  python main_pipline.py --visualize-only    # Only show classification visualizations
  python main_pipline.py --skip-training     # Use existing model
  python main_pipline.py --validation-only   # Only run validation
  python main_pipline.py --calibrate-tool grinder  # Calibrate new tool
  python main_pipline.py --classify-csv path/to/features.csv  # Classify CSV with pre-extracted features
        """
    )
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip data preprocessing verification')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip classifier training (use existing model)')
    parser.add_argument('--validation-only', action='store_true', 
                       help='Run validation only')
    parser.add_argument('--visualize', action='store_true',
                       help='Show classification visualization after training')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only run visualization (skip training and validation)')
    parser.add_argument('--calibrate-tool', type=str, metavar='TOOL_NAME',
                       help='Calibrate thresholds for a new tool (e.g., grinder, saw)')
    parser.add_argument('--classify-csv', type=str, metavar='CSV_PATH',
                       help='Classify pre-extracted feature CSV dataset')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    # Handle calibration mode separately
    if args.calibrate_tool:
        print_banner(f"TOOL CALIBRATION MODE: {args.calibrate_tool.upper()}")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            success = run_calibration(args.calibrate_tool)
            if success:
                print_banner("CALIBRATION COMPLETE")
                return 0
            else:
                print_banner("CALIBRATION FAILED")
                return 1
        except Exception as e:
            print(f"\n[ERROR] Calibration failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Handle CSV feature classification mode
    if args.classify_csv:
        print_banner("CSV FEATURE DATASET CLASSIFICATION")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"CSV File: {args.classify_csv}")
        
        try:
            success = classify_feature_csv(args.classify_csv, save_results=True, visualize=True)
            if success:
                print_banner("CSV CLASSIFICATION COMPLETE")
                return 0
            else:
                print_banner("CSV CLASSIFICATION FAILED")
                return 1
        except Exception as e:
            print(f"\n[ERROR] CSV classification failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Handle visualization-only mode
    if args.visualize_only:
        print_banner("VISUALIZATION MODE")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Scan for data files
            data_files = scan_data_folders()
            visualize_classification(data_files, save_plots=True)
            print_banner("VISUALIZATION COMPLETE")
            return 0
        except Exception as e:
            print(f"\n[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Standard pipeline execution
    print_banner("VIBRATION ANALYSIS - COMPLETE PIPELINE")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis pipeline will:")
    print("  1. Verify data preprocessing")
    print("  2. Train ON/OFF classifier (multi-feature)")
    print("  3. Run comprehensive validation")
    print("  4. Generate reports and visualizations")
    print("\nFeatures used for classification:")
    print("  • RMS (Root Mean Square)")
    print("  • Spectral Centroid")
    print("  • Spectral Bandwidth")
    print("  • Spectral Flatness")
    print("  • Crest Factor")
    print("  • High-Frequency Energy Ratio")
    print("  • Peak Prominence")
    
    try:
        data_files = None
        
        # Step 1: Preprocessing
        if not args.skip_preprocessing and not args.validation_only:
            result = run_preprocessing()
            if isinstance(result, tuple):
                success, data_files = result
            else:
                success = result
            if not success:
                print("\n[ERROR] Preprocessing verification failed!")
                return 1
        
        # Step 2: Training
        if not args.skip_training and not args.validation_only:
            if not run_training(data_files):
                print("\n[ERROR] Training failed!")
                return 1
        elif args.skip_training:
            print_section("STEP 2: TRAINING (SKIPPED)")
            print("Using existing trained model...")
        
        # Step 2.5: Visualization (if requested)
        if args.visualize:
            visualize_classification(data_files, save_plots=True)
        
        # Step 3: Validation
        metrics = run_validation()
        
        # Step 4: Final Report
        generate_final_report(metrics, start_time)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n[WARN]  Pipeline interrupted by user!")
        return 1
    except Exception as e:
        print(f"\n\n[ERROR] Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
