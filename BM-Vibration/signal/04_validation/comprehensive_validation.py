"""
Comprehensive Validation Suite for Vibration Classification
Generates detailed performance metrics, confusion matrices, and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
import sys
import json
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_preprocessing', 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_classifiers', 'on_off'))

from preprocess_pipeline import load_raw_data_from_csv
from highpass_filter import filter_triaxial_data
from segmentation import create_overlapping_windows
from onoff_model import OnOffClassifier

# Dataset definitions
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '01_data_collection', 'raw')

DATASETS = {
    'no_move': {
        'csv': os.path.join(RAW_DATA_PATH, 'processed_No move.csv'),
        'label': 0,  # OFF = 0
        'label_name': 'OFF',
        'description': 'Sensor stationary - no movement'
    },
    'walking': {
        'csv': os.path.join(RAW_DATA_PATH, 'processed_Walking test 1.csv'),
        'label': 0,  # OFF = 0
        'label_name': 'OFF',
        'description': 'Walking - noise/movement without tool'
    },
    'pms_drill': {
        'csv': os.path.join(RAW_DATA_PATH, 'processed_Pms + drill test 2.csv'),
        'label': 1,  # ON = 1
        'label_name': 'ON',
        'description': 'PMS with drill test 2'
    },
    'psm_drill': {
        'csv': os.path.join(RAW_DATA_PATH, 'processed_Psm + drill(app controll).csv'),
        'label': 1,  # ON = 1
        'label_name': 'ON',
        'description': 'PSM with drill (app controlled)'
    }
}

class ComprehensiveValidator:
    def __init__(self):
        self.classifier = OnOffClassifier()
        self.results = []
        self.window_results = []
        self.output_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def print_header(self, text):
        print("\n" + "="*80)
        print(text.center(80))
        print("="*80)
    
    def validate_all_datasets(self):
        """Validate all datasets and collect window-level predictions"""
        self.print_header("VALIDATION: COLLECTING PREDICTIONS")
        
        all_true_labels = []
        all_predictions = []
        all_prediction_probs = []
        
        for name, info in DATASETS.items():
            csv_path = info['csv']
            
            if not os.path.exists(csv_path):
                print(f"⚠️  Skipping {name}: File not found")
                continue
            
            print(f"\n📁 Processing: {name}")
            print(f"   {info['description']}")
            
            # Load and process
            df_raw = load_raw_data_from_csv(csv_path)
            if df_raw is None or df_raw.empty:
                print(f"   ❌ Failed to load data")
                continue
            
            df_filtered = filter_triaxial_data(df_raw)
            windows = create_overlapping_windows(df_filtered)
            
            # Get predictions for each window
            predictions = self.classifier.predict_batch(windows)
            ground_truth = [info['label']] * len(windows)
            
            # Calculate magnitudes for probability scores
            magnitudes = []
            for window in windows:
                mag = np.sqrt(np.sum(window**2, axis=1))
                magnitudes.append(np.mean(mag))
            
            # Store results
            all_true_labels.extend(ground_truth)
            all_predictions.extend(predictions)
            all_prediction_probs.extend(magnitudes)
            
            # Per-dataset summary
            n_on = sum(predictions)
            accuracy = sum(p == t for p, t in zip(predictions, ground_truth)) / len(predictions)
            
            print(f"   ✅ Windows: {len(windows)}, Predicted ON: {n_on}, Accuracy: {accuracy*100:.1f}%")
            
            self.results.append({
                'dataset': name,
                'description': info['description'],
                'ground_truth': info['label_name'],
                'n_windows': len(windows),
                'n_correct': sum(p == t for p, t in zip(predictions, ground_truth)),
                'accuracy': accuracy
            })
            
            # Store window-level results
            for i, (pred, truth, mag) in enumerate(zip(predictions, ground_truth, magnitudes)):
                self.window_results.append({
                    'dataset': name,
                    'window_idx': i,
                    'prediction': pred,
                    'ground_truth': truth,
                    'magnitude': mag,
                    'correct': pred == truth
                })
        
        return np.array(all_true_labels), np.array(all_predictions), np.array(all_prediction_probs)
    
    def generate_confusion_matrix(self, y_true, y_pred):
        """Generate and plot confusion matrix"""
        self.print_header("CONFUSION MATRIX")
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Print text version
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 OFF    ON")
        print(f"Actual  OFF    [{cm[0,0]:4d}  {cm[0,1]:4d}]")
        print(f"        ON     [{cm[1,0]:4d}  {cm[1,1]:4d}]")
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print("\nPerformance Metrics:")
        print(f"  Accuracy:    {accuracy*100:.2f}%")
        print(f"  Precision:   {precision*100:.2f}%")
        print(f"  Recall:      {recall*100:.2f}%")
        print(f"  F1-Score:    {f1*100:.2f}%")
        print(f"  Specificity: {specificity*100:.2f}%")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['OFF', 'ON'], 
                    yticklabels=['OFF', 'ON'])
        plt.title('Confusion Matrix - ON/OFF Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved: {output_path}")
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity
        }
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        self.print_header("CLASSIFICATION REPORT")
        
        report = classification_report(y_true, y_pred, 
                                       target_names=['OFF', 'ON'], 
                                       digits=3)
        print(report)
        
        # Save to file
        output_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(output_path, 'w') as f:
            f.write("Classification Report - Vibration ON/OFF Detection\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        print(f"\n💾 Saved: {output_path}")
    
    def plot_per_dataset_performance(self):
        """Plot performance metrics per dataset"""
        self.print_header("PER-DATASET PERFORMANCE")
        
        df = pd.DataFrame(self.results)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(df))
        bars = ax.bar(x_pos, df['accuracy'] * 100, color=['green' if acc > 0.9 else 'orange' if acc > 0.7 else 'red' for acc in df['accuracy']])
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Classification Accuracy by Dataset')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['dataset'], rotation=45, ha='right')
        ax.set_ylim([0, 105])
        ax.axhline(y=100, color='g', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, df['accuracy'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc*100:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'per_dataset_accuracy.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved: {output_path}")
        plt.close()
        
        # Print summary table
        print("\nDataset Summary:")
        print("-" * 80)
        print(f"{'Dataset':<20} {'Ground Truth':<15} {'Windows':<10} {'Correct':<10} {'Accuracy':<10}")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"{row['dataset']:<20} {row['ground_truth']:<15} {row['n_windows']:<10} "
                  f"{row['n_correct']:<10} {row['accuracy']*100:>6.1f}%")
        print("-" * 80)
    
    def plot_magnitude_distribution(self):
        """Plot magnitude distribution for correct vs incorrect predictions"""
        self.print_header("MAGNITUDE ANALYSIS")
        
        df_windows = pd.DataFrame(self.window_results)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Magnitude by prediction correctness
        correct = df_windows[df_windows['correct'] == True]
        incorrect = df_windows[df_windows['correct'] == False]
        
        axes[0].hist(correct['magnitude'], bins=50, alpha=0.6, label='Correct', color='green')
        axes[0].hist(incorrect['magnitude'], bins=50, alpha=0.6, label='Incorrect', color='red')
        axes[0].axvline(self.classifier.magnitude_threshold, color='black', 
                       linestyle='--', linewidth=2, label=f'Threshold ({self.classifier.magnitude_threshold:.2f})')
        axes[0].set_xlabel('Magnitude (m/s²)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Magnitude Distribution: Correct vs Incorrect')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Magnitude by ground truth
        off_windows = df_windows[df_windows['ground_truth'] == 0]
        on_windows = df_windows[df_windows['ground_truth'] == 1]
        
        axes[1].hist(off_windows['magnitude'], bins=50, alpha=0.6, label='OFF (Ground Truth)', color='blue')
        axes[1].hist(on_windows['magnitude'], bins=50, alpha=0.6, label='ON (Ground Truth)', color='orange')
        axes[1].axvline(self.classifier.magnitude_threshold, color='black', 
                       linestyle='--', linewidth=2, label=f'Threshold ({self.classifier.magnitude_threshold:.2f})')
        axes[1].set_xlabel('Magnitude (m/s²)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Magnitude Distribution: OFF vs ON')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'magnitude_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved: {output_path}")
        plt.close()
    
    def analyze_errors(self):
        """Detailed analysis of misclassifications"""
        self.print_header("ERROR ANALYSIS")
        
        df_windows = pd.DataFrame(self.window_results)
        errors = df_windows[df_windows['correct'] == False]
        
        if len(errors) == 0:
            print("\n✅ No errors found! Perfect classification!")
            return
        
        print(f"\nTotal Errors: {len(errors)} out of {len(df_windows)} windows ({len(errors)/len(df_windows)*100:.2f}%)")
        
        # Group errors by dataset
        error_by_dataset = errors.groupby('dataset').size()
        print("\nErrors by Dataset:")
        for dataset, count in error_by_dataset.items():
            total_windows = len(df_windows[df_windows['dataset'] == dataset])
            print(f"  {dataset}: {count} errors / {total_windows} windows ({count/total_windows*100:.1f}%)")
        
        # Analyze error types
        false_positives = errors[errors['ground_truth'] == 0]  # Predicted ON, actually OFF
        false_negatives = errors[errors['ground_truth'] == 1]  # Predicted OFF, actually ON
        
        print(f"\nError Types:")
        print(f"  False Positives (predicted ON, actually OFF): {len(false_positives)}")
        print(f"  False Negatives (predicted OFF, actually ON): {len(false_negatives)}")
        
        # Magnitude statistics for errors
        print(f"\nMagnitude Statistics for Errors:")
        print(f"  Mean magnitude: {errors['magnitude'].mean():.3f}")
        print(f"  Std magnitude: {errors['magnitude'].std():.3f}")
        print(f"  Min magnitude: {errors['magnitude'].min():.3f}")
        print(f"  Max magnitude: {errors['magnitude'].max():.3f}")
        print(f"  Threshold: {self.classifier.magnitude_threshold:.3f}")
    
    def save_detailed_results(self):
        """Save all results to CSV files"""
        self.print_header("SAVING RESULTS")
        
        # Save dataset-level results
        df_datasets = pd.DataFrame(self.results)
        output_path = os.path.join(self.output_dir, 'dataset_results.csv')
        df_datasets.to_csv(output_path, index=False)
        print(f"💾 Dataset results: {output_path}")
        
        # Save window-level results
        df_windows = pd.DataFrame(self.window_results)
        output_path = os.path.join(self.output_dir, 'window_results.csv')
        df_windows.to_csv(output_path, index=False)
        print(f"💾 Window results: {output_path}")
    
    def generate_summary_report(self, metrics):
        """Generate final validation summary"""
        self.print_header("VALIDATION SUMMARY")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            'timestamp': timestamp,
            'classifier': 'OnOffClassifier',
            'threshold': float(self.classifier.magnitude_threshold),
            'consecutive_samples': self.classifier.consecutive_samples,
            'datasets': len(self.results),
            'total_windows': len(self.window_results),
            'metrics': metrics,
            'dataset_results': self.results
        }
        
        # Save JSON report
        output_path = os.path.join(self.output_dir, 'validation_summary.json')
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Summary JSON: {output_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("VALIDATION COMPLETE".center(80))
        print(f"{'='*80}")
        print(f"\nTimestamp: {timestamp}")
        print(f"Datasets Validated: {len(self.results)}")
        print(f"Total Windows: {len(self.window_results)}")
        print(f"\nOverall Performance:")
        print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
        print(f"  Precision:   {metrics['precision']*100:.2f}%")
        print(f"  Recall:      {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:    {metrics['f1_score']*100:.2f}%")
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"{'='*80}")


def main():
    """Main validation execution"""
    print("="*80)
    print("COMPREHENSIVE VALIDATION SUITE".center(80))
    print("Vibration ON/OFF Classification".center(80))
    print("="*80)
    
    validator = ComprehensiveValidator()
    
    # Step 1: Validate all datasets
    y_true, y_pred, y_probs = validator.validate_all_datasets()
    
    # Step 2: Generate confusion matrix and metrics
    metrics = validator.generate_confusion_matrix(y_true, y_pred)
    
    # Step 3: Classification report
    validator.generate_classification_report(y_true, y_pred)
    
    # Step 4: Per-dataset performance
    validator.plot_per_dataset_performance()
    
    # Step 5: Magnitude analysis
    validator.plot_magnitude_distribution()
    
    # Step 6: Error analysis
    validator.analyze_errors()
    
    # Step 7: Save results
    validator.save_detailed_results()
    
    # Step 8: Generate summary
    validator.generate_summary_report(metrics)


if __name__ == '__main__':
    main()
