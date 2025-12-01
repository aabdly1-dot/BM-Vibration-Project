"""
MAIN PIPELINE - Complete Vibration Analysis Workflow
Orchestrates: Data Loading -> Preprocessing -> Training -> Validation -> Reporting
"""

import os
import sys
import argparse
from datetime import datetime

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

def run_preprocessing():
    """Step 1: Preprocess raw data"""
    print_section("STEP 1: DATA PREPROCESSING")
    print("Running preprocessing pipeline...")
    
    from preprocess_pipeline import load_raw_data_from_csv
    from highpass_filter import filter_triaxial_data
    
    # Verify data files exist
    raw_path = os.path.join(script_dir, '01_data_collection', 'raw')
    
    csv_files = [
        'processed_No move.csv',
        'processed_Walking test 1.csv',
        'processed_Pms + drill test 2.csv',
        'processed_Psm + drill(app controll).csv'
    ]
    
    print(f"Data directory: {raw_path}")
    print(f"\nVerifying {len(csv_files)} CSV files:")
    
    all_exist = True
    for csv_file in csv_files:
        full_path = os.path.join(raw_path, csv_file)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {csv_file}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ All data files found and ready!")
        return True
    else:
        print("\n❌ Some data files are missing!")
        return False

def run_training():
    """Step 2: Train ON/OFF classifier"""
    print_section("STEP 2: TRAINING ON/OFF CLASSIFIER")
    
    import numpy as np
    from preprocess_pipeline import load_raw_data_from_csv
    from highpass_filter import filter_triaxial_data
    from segmentation import create_overlapping_windows
    from onoff_model import OnOffClassifier, calibrate_threshold
    
    raw_path = os.path.join(script_dir, '01_data_collection', 'raw')
    
    # Load ON data (drill tests)
    print("Loading ON (Tool Active) data...")
    on_files = [
        'processed_Pms + drill test 2.csv',
        'processed_Psm + drill(app controll).csv'
    ]
    
    on_windows = []
    for csv_file in on_files:
        csv_path = os.path.join(raw_path, csv_file)
        print(f"  • {csv_file}")
        df = load_raw_data_from_csv(csv_path)
        df_filtered = filter_triaxial_data(df)
        windows = create_overlapping_windows(df_filtered)
        on_windows.append(windows)
    
    on_windows = np.vstack(on_windows)
    print(f"  Total ON windows: {len(on_windows)}")
    
    # Load OFF data (no movement, walking)
    print("\nLoading OFF (No Tool) data...")
    off_files = [
        'processed_No move.csv',
        'processed_Walking test 1.csv'
    ]
    
    off_windows = []
    for csv_file in off_files:
        csv_path = os.path.join(raw_path, csv_file)
        print(f"  • {csv_file}")
        df = load_raw_data_from_csv(csv_path)
        df_filtered = filter_triaxial_data(df)
        windows = create_overlapping_windows(df_filtered)
        off_windows.append(windows)
    
    off_windows = np.vstack(off_windows)
    print(f"  Total OFF windows: {len(off_windows)}")
    
    # Train classifier
    print("\nCalibrating threshold...")
    optimal_threshold = calibrate_threshold(on_windows, off_windows)
    
    classifier = OnOffClassifier(magnitude_threshold=optimal_threshold)
    classifier.save_thresholds()
    
    print(f"✅ Classifier trained with threshold: {optimal_threshold:.3f}")
    print(f"✅ Model saved to: 03_classifiers/on_off/thresholds.json")
    
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
    
    print("\n✅ Validation complete!")
    return metrics

def generate_final_report(metrics, start_time):
    """Step 4: Generate final summary report"""
    print_banner("PIPELINE EXECUTION SUMMARY")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n⏱️  Execution Time: {duration:.2f} seconds")
    print(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 Final Results:")
    print(f"  • Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  • Precision:   {metrics['precision']*100:.2f}%")
    print(f"  • Recall:      {metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:    {metrics['f1_score']*100:.2f}%")
    print(f"  • Specificity: {metrics['specificity']*100:.2f}%")
    
    print("\n📁 Generated Artifacts:")
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
    print("✅ PIPELINE COMPLETE!".center(80))
    print("="*80 + "\n")

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Complete Vibration Analysis Pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip data preprocessing verification')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip classifier training (use existing model)')
    parser.add_argument('--validation-only', action='store_true', 
                       help='Run validation only')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print_banner("VIBRATION ANALYSIS - COMPLETE PIPELINE")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis pipeline will:")
    print("  1. Verify data preprocessing")
    print("  2. Train ON/OFF classifier")
    print("  3. Run comprehensive validation")
    print("  4. Generate reports and visualizations")
    
    try:
        # Step 1: Preprocessing
        if not args.skip_preprocessing and not args.validation_only:
            if not run_preprocessing():
                print("\n❌ Preprocessing verification failed!")
                return 1
        
        # Step 2: Training
        if not args.skip_training and not args.validation_only:
            if not run_training():
                print("\n❌ Training failed!")
                return 1
        elif args.skip_training:
            print_section("STEP 2: TRAINING (SKIPPED)")
            print("Using existing trained model...")
        
        # Step 3: Validation
        metrics = run_validation()
        
        # Step 4: Final Report
        generate_final_report(metrics, start_time)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user!")
        return 1
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
