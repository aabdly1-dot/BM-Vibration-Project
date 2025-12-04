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
    print(f"    • HF ratio min: {classifier.thresholds['hf_ratio_min']}")
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
            hf_ratio_values = [f['hf_energy_ratio'] for f in all_features]
            
            # Create figure with 4 subplots
            fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
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
            
            # --- Subplot 4: HF Energy Ratio ---
            ax4 = axes[3]
            ax4.plot(window_times, hf_ratio_values, 'g-^', markersize=3, linewidth=1)
            ax4.axhline(y=classifier.thresholds['hf_ratio_min'], color='g', linestyle='--', 
                       alpha=0.5, label=f"HF Ratio thresh ({classifier.thresholds['hf_ratio_min']:.2f})")
            ax4.set_ylabel('HF Energy Ratio')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_title('High-Frequency Energy Ratio (>50Hz)')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
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
            'hf_ratio_min': 0.25,
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
