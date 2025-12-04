import pandas as pd
import numpy as np
import os
import sys
import json

# --- Dynamic Import Setup ---
# Add the parent directory to the path so we can import modules from 'scripts'
sys.path.append(os.path.dirname(__file__))

# Import the core signal processing functions
from highpass_filter import filter_triaxial_data, FS, BANDPASS_LOW, BANDPASS_HIGH
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


def load_raw_data_from_csv(csv_path):
    """
    Loads raw triaxial acceleration data from a CSV file.
    Also calculates effective sampling frequency from timestamps if available.
    
    Parameters:
    - csv_path (str): Path to the CSV file.
    
    Returns:
    - df_raw (pd.DataFrame): DataFrame with accel_x, accel_y, accel_z columns.
    - effective_fs (float): Calculated sampling frequency, or FS default if timestamps unavailable.
    """
    df = pd.read_csv(csv_path)
    
    # Calculate effective sampling frequency if timestamp column exists
    effective_fs = FS  # Default
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].values
        if len(timestamps) > 1:
            # Timestamps are typically in milliseconds
            dt_ms = np.diff(timestamps)
            mean_dt_ms = np.mean(dt_ms)
            if mean_dt_ms > 0:
                effective_fs = 1000.0 / mean_dt_ms
                print(f"Effective Fs calculated: {effective_fs:.1f} Hz")
    
    # Standardize column names if needed
    df_raw = pd.DataFrame({
        'accel_x': df['accel_x'].values if 'accel_x' in df.columns else df.iloc[:, 1].values,
        'accel_y': df['accel_y'].values if 'accel_y' in df.columns else df.iloc[:, 2].values,
        'accel_z': df['accel_z'].values if 'accel_z' in df.columns else df.iloc[:, 3].values
    })
    
    print(f"Loaded {len(df_raw)} samples from {os.path.basename(csv_path)}")
    return df_raw, effective_fs


def load_raw_data_from_json(json_path):
    """
    Loads raw triaxial acceleration data from a Movesense JSON file.
    Parses the ArrayAcc structure and calculates effective sampling frequency.
    
    Expected JSON structure:
    {
        "data": [
            {
                "acc": {
                    "Timestamp": <int>,
                    "ArrayAcc": [{"x": <float>, "y": <float>, "z": <float>}, ...]
                }
            }, ...
        ]
    }
    
    Parameters:
    - json_path (str): Path to the JSON file.
    
    Returns:
    - df_raw (pd.DataFrame): DataFrame with accel_x, accel_y, accel_z columns.
    - effective_fs (float): Calculated sampling frequency.
    """
    print(f"Loading JSON: {os.path.basename(json_path)}")
    
    with open(json_path, 'r') as f:
        try:
            data_json = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None, FS

    if 'data' not in data_json:
        print("Error: JSON does not contain 'data' key.")
        return None, FS

    # Extract packet data
    packet_timestamps = []
    packet_sample_counts = []
    raw_samples_per_packet = []

    for entry in data_json['data']:
        if 'acc' in entry:
            acc_data = entry['acc']
            timestamp = acc_data.get('Timestamp')
            array_acc = acc_data.get('ArrayAcc', [])
            
            if timestamp is not None and array_acc:
                packet_timestamps.append(timestamp)
                packet_sample_counts.append(len(array_acc))
                raw_samples_per_packet.append(array_acc)

    if not packet_timestamps:
        print("No valid 'acc' data found in JSON.")
        return None, FS

    # Calculate effective sampling rate
    t_packets = np.array(packet_timestamps)
    
    if len(t_packets) > 1:
        dt_packets = np.diff(t_packets)
        avg_dt_packet = np.mean(dt_packets)
        avg_samples_per_packet = np.mean(packet_sample_counts)
        
        # Effective sample interval (ms)
        sample_interval_ms = avg_dt_packet / avg_samples_per_packet
        effective_fs = 1000.0 / sample_interval_ms if sample_interval_ms > 0 else FS
    else:
        sample_interval_ms = 1000.0 / FS
        effective_fs = FS
    
    print(f"Estimated sample interval: {sample_interval_ms:.4f} ms (Fs={effective_fs:.1f} Hz)")
    
    # Generate timestamps for each sample and flatten acceleration data
    final_timestamps = []
    final_acc_x = []
    final_acc_y = []
    final_acc_z = []
    
    for i, packet_samples in enumerate(raw_samples_per_packet):
        packet_ts = packet_timestamps[i]
        num_samples = len(packet_samples)
        
        # Generate sample times backwards from packet timestamp
        packet_times = [packet_ts - (num_samples - 1 - k) * sample_interval_ms for k in range(num_samples)]
        final_timestamps.extend(packet_times)
        
        for sample in packet_samples:
            final_acc_x.append(sample.get('x', 0.0))
            final_acc_y.append(sample.get('y', 0.0))
            final_acc_z.append(sample.get('z', 0.0))
    
    df_raw = pd.DataFrame({
        'timestamp': final_timestamps,
        'accel_x': final_acc_x,
        'accel_y': final_acc_y,
        'accel_z': final_acc_z
    })
    
    print(f"Loaded {len(df_raw)} samples from JSON")
    return df_raw, effective_fs


def run_pipeline_from_json(json_path, apply_bandpass=True):
    """
    Executes the full preprocessing pipeline from a Movesense JSON file: Load -> Filter -> Segment.
    
    Parameters:
    - json_path (str): Path to the JSON file.
    - apply_bandpass (bool): If True, apply bandpass filter (40-400Hz) after highpass.
    
    Returns:
    - segmented_array (np.array): 3D array of shape (N_windows, window_size, 3).
    - effective_fs (float): The calculated sampling frequency.
    """
    print(f"\n--- Running Preprocessing Pipeline for {os.path.basename(json_path)} ---")

    # 1. Load Raw Data from JSON
    df_raw, effective_fs = load_raw_data_from_json(json_path)
    
    if df_raw is None:
        print("Failed to load JSON data!")
        return None, effective_fs
    
    # 2. Apply High-Pass Filter + optional Bandpass Filter
    if apply_bandpass:
        print(f"Applying highpass + bandpass filter ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)...")
    else:
        print("Applying high-pass filter only...")
    
    df_filtered = filter_triaxial_data(df_raw, fs=effective_fs, apply_bandpass=apply_bandpass)
    
    # VITAL CHECK: Verify that the filter worked
    print(f"Raw Z-axis Mean: {df_raw['accel_z'].mean():.2f}")
    print(f"Filtered Z-axis Mean (Expected ~0.0): {df_filtered['accel_z_filtered'].mean():.2f}")

    # 3. Run Segmentation (breaks continuous stream into windows)
    print(f"Segmenting data (Window Size: {WINDOW_SIZE}, Slide Step: {SLIDE_STEP})...")
    segmented_array = create_overlapping_windows(df_filtered)
    
    print(f"Pipeline complete. Created {segmented_array.shape[0]} windows.")
    return segmented_array, effective_fs


def run_pipeline(tool_type='tool_drill', apply_bandpass=True):
    """
    Executes the full preprocessing pipeline: Load -> Filter -> Segment.
    
    Parameters:
    - tool_type (str): Type of tool data to process.
    - apply_bandpass (bool): If True, apply bandpass filter (40-400Hz) after highpass.
    """
    print(f"\n--- Running Preprocessing Pipeline for {tool_type.upper()} ---")

    # 1. Load Raw Data
    df_raw = load_raw_data(tool_type)
    
    # 2. Apply High-Pass Filter + optional Bandpass Filter
    if apply_bandpass:
        print(f"Applying highpass + bandpass filter ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)...")
    else:
        print("Applying high-pass filter only...")
    
    df_filtered = filter_triaxial_data(df_raw, fs=FS, apply_bandpass=apply_bandpass)
    
    # VITAL CHECK: Verify that the filter worked
    print(f"Raw Z-axis Mean (Expected ~9.81): {df_raw['accel_z'].mean():.2f}")
    print(f"Filtered Z-axis Mean (Expected ~0.0): {df_filtered['accel_z_filtered'].mean():.2f}")

    # 3. Run Segmentation (breaks continuous stream into windows)
    print(f"Segmenting data (Window Size: {WINDOW_SIZE}, Slide Step: {SLIDE_STEP})...")
    segmented_array = create_overlapping_windows(df_filtered)
    
    print(f"Pipeline complete. Created {segmented_array.shape[0]} windows.")
    return segmented_array, FS


def run_pipeline_from_csv(csv_path, apply_bandpass=True):
    """
    Executes the full preprocessing pipeline from a CSV file: Load -> Filter -> Segment.
    
    Parameters:
    - csv_path (str): Path to the CSV file.
    - apply_bandpass (bool): If True, apply bandpass filter (40-400Hz) after highpass.
    
    Returns:
    - segmented_array (np.array): 3D array of shape (N_windows, window_size, 3).
    - effective_fs (float): The calculated sampling frequency.
    """
    print(f"\n--- Running Preprocessing Pipeline for {os.path.basename(csv_path)} ---")

    # 1. Load Raw Data from CSV
    df_raw, effective_fs = load_raw_data_from_csv(csv_path)
    
    # 2. Apply High-Pass Filter + optional Bandpass Filter
    if apply_bandpass:
        print(f"Applying highpass + bandpass filter ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)...")
    else:
        print("Applying high-pass filter only...")
    
    df_filtered = filter_triaxial_data(df_raw, fs=effective_fs, apply_bandpass=apply_bandpass)
    
    # VITAL CHECK: Verify that the filter worked
    print(f"Raw Z-axis Mean: {df_raw['accel_z'].mean():.2f}")
    print(f"Filtered Z-axis Mean (Expected ~0.0): {df_filtered['accel_z_filtered'].mean():.2f}")

    # 3. Run Segmentation (breaks continuous stream into windows)
    print(f"Segmenting data (Window Size: {WINDOW_SIZE}, Slide Step: {SLIDE_STEP})...")
    segmented_array = create_overlapping_windows(df_filtered)
    
    print(f"Pipeline complete. Created {segmented_array.shape[0]} windows.")
    return segmented_array, effective_fs


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
    
    # Step 1: Run the pipeline for your 'drill' data (with bandpass filtering)
    segments_drill, fs_drill = run_pipeline(tool_type='tool_drill', apply_bandpass=True)
    
    # Step 2: Save the result (ready for the Classifier Notebooks)
    save_cleaned_data(segments_drill, tool_type='tool_drill')
    
    # Step 3: Repeat for noise data (e.g., 'noise_walking')
    segments_walking, fs_walking = run_pipeline(tool_type='noise_walking', apply_bandpass=True)
    save_cleaned_data(segments_walking, tool_type='noise_walking')
    
    print(f"\nSampling frequencies: Drill={fs_drill}Hz, Walking={fs_walking}Hz")
