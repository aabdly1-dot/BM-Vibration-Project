import pandas as pd
import numpy as np
import os
import sys

# --- Dynamic Import Setup ---
# Add the parent directory to the path so we can import modules from 'scripts'
sys.path.append(os.path.dirname(__file__))

# Import the core signal processing functions
from highpass_filter import filter_triaxial_data, FS  # FS is the Sampling Frequency constant
from segmentation import create_overlapping_windows, WINDOW_SIZE, SLIDE_STEP 

# --- Project Path Constants ---
# Assuming your raw data is here (replace with actual loading logic)
RAW_DATA_PATH = '../01_data_collection/raw/' 
# Assuming your output cleaned data will go here (ready for 03_classifiers)
CLEAN_DATA_OUTPUT_PATH = '../02_preprocessing/data_output/'


def load_raw_data(tool_type='tool_drill'):
    """
    Loads raw triaxial acceleration data for a specified tool or noise type.
    
    NOTE: In a real project, this function handles reading the multiple raw files (x, y, z) 
    and merging them into a single DataFrame.
    """
    print(f"Loading raw data for: {tool_type}...")
    
    # --- SIMULATION: Replace this block with your actual file loading logic ---
    # For a real project, you would read 'total_acc_x.txt', 'total_acc_y.txt', etc.
    
    # Simulate loading 5000 samples of noisy, raw data for testing
    N = 5000
    time = np.linspace(0, N/FS, N, endpoint=False)
    # Simulate raw signal: Gravity (9.81) + Walking (0.5 Hz) + Tool Vibration (70 Hz)
    raw_x = 9.81 + 0.5 * np.sin(2 * np.pi * 0.5 * time) + 10 * np.sin(2 * np.pi * 70 * time)
    raw_y = 9.81 + 0.2 * np.sin(2 * np.pi * 0.4 * time) + 8 * np.sin(2 * np.pi * 70 * time)
    raw_z = 9.81 + 0.1 * np.sin(2 * np.pi * 0.3 * time) + 12 * np.sin(2 * np.pi * 70 * time)

    df_raw = pd.DataFrame({
        'accel_x': raw_x,
        'accel_y': raw_y,
        'accel_z': raw_z
    })
    # --- END SIMULATION BLOCK ---
    
    print(f"Successfully loaded {len(df_raw)} raw samples.")
    return df_raw


def run_pipeline(tool_type='tool_drill'):
    """
    Executes the full preprocessing pipeline: Load -> Filter -> Segment.
    """
    print(f"\n--- Running Preprocessing Pipeline for {tool_type.upper()} ---")

    # 1. Load Raw Data
    df_raw = load_raw_data(tool_type)
    
    # 2. Apply High-Pass Filter (removes gravity/walking noise)
    print("Applying high-pass filter...")
    df_filtered = filter_triaxial_data(df_raw)
    
    # VITAL CHECK: Verify that the filter worked
    print(f"Raw Z-axis Mean (Expected ~9.81): {df_raw['accel_z'].mean():.2f}")
    print(f"Filtered Z-axis Mean (Expected ~0.0): {df_filtered['accel_z_filtered'].mean():.2f}")

    # 3. Run Segmentation (breaks continuous stream into windows)
    print(f"Segmenting data (Window Size: {WINDOW_SIZE}, Slide Step: {SLIDE_STEP})...")
    segmented_array = create_overlapping_windows(df_filtered)
    
    print(f"Pipeline complete. Created {segmented_array.shape[0]} windows.")
    return segmented_array


def save_cleaned_data(segmented_array, tool_type='tool_drill'):
    """
    Saves the final 3D NumPy array to a file, ready for the 03_classifiers folder.
    """
    # Ensure the output directory exists
    os.makedirs(CLEAN_DATA_OUTPUT_PATH, exist_ok=True)
    
    output_filename = os.path.join(CLEAN_DATA_OUTPUT_PATH, f'{tool_type}_cleaned_segments.npy')
    
    # Saving the data as a compressed NumPy file (best practice)
    np.save(output_filename, segmented_array)
    
    print(f"Cleaned data saved to: {output_filename}")


if __name__ == '__main__':
    # --- Main Execution ---
    
    # Step 1: Run the pipeline for your 'drill' data
    segments_drill = run_pipeline(tool_type='tool_drill')
    
    # Step 2: Save the result (ready for the Classifier Notebooks)
    save_cleaned_data(segments_drill, tool_type='tool_drill')
    
    # Step 3: Repeat for noise data (e.g., 'noise_walking')
    segments_walking = run_pipeline(tool_type='noise_walking')
    save_cleaned_data(segments_walking, tool_type='noise_walking')
