import numpy as np
import pandas as pd

# --- Project Constants (Base these on standard HAR practice) ---
# NOTE: These values need to be finalized through testing, but 1.28s is common.

# The window size defines the length of each segment analyzed by the classifier.
WINDOW_SIZE = 64  # samples. (e.g., if FS=50Hz, 64 samples is 1.28 seconds)

# The overlap defines how much the next window moves forward.
# A 50% overlap (0.5) is common to prevent missing key events at window boundaries.
OVERLAP_RATIO = 0.5  
SLIDE_STEP = int(WINDOW_SIZE * OVERLAP_RATIO)  # e.g., 32 samples


def create_overlapping_windows(df_filtered, window_size=WINDOW_SIZE, slide_step=SLIDE_STEP):
    """
    Breaks a continuous, filtered sensor stream (DataFrame) into a 3D NumPy array
    using fixed-size, overlapping windows.

    This function prepares data for feature extraction (fft_feature_extract.py)
    and subsequent classification (train_onoff.ipynb).

    Parameters:
    - df_filtered (pd.DataFrame): The input DataFrame containing filtered triaxial
      acceleration data (columns: accel_x_filtered, accel_y_filtered, accel_z_filtered).
    - window_size (int): The number of samples in each window.
    - slide_step (int): The number of samples to slide before creating the next window.

    Returns:
    - np.array: A 3D array of shape (N_windows, window_size, 3)
                (where 3 represents the X, Y, Z axes).
    """
    
    # 1. Prepare Data
    # Convert DataFrame to a NumPy array for fast iteration
    data_array = df_filtered[['accel_x_filtered', 'accel_y_filtered', 'accel_z_filtered']].values
    
    # Calculate the total length of the data
    n_samples = len(data_array)
    
    # List to hold all the individual windows
    windows = []
    
    # 2. Loop and Segment
    # Iterate through the data, stopping when a full window cannot be created
    for start in range(0, n_samples - window_size + 1, slide_step):
        end = start + window_size
        window = data_array[start:end, :]
        windows.append(window)
    
    # 3. Final Output
    # Convert the list of windows into a single 3D NumPy array
    return np.array(windows)


# Optional: Function to retrieve corresponding labels if you were doing full HAR
def create_window_labels(df_raw_labels, window_size=WINDOW_SIZE, slide_step=SLIDE_STEP):
    """
    Retrieves the majority label (e.g., 'Vibration ON' or 'OFF') for each window.
    """
    n_samples = len(df_raw_labels)
    window_labels = []
    
    for start in range(0, n_samples - window_size + 1, slide_step):
        window = df_raw_labels[start:start + window_size]
        # Assign the label that occurs most frequently in the window (majority vote)
        # NOTE: For your 'On/Off' classifier, you might use a simpler rule: 
        # if ANY sample is 'Vibration ON', the whole window is 'Vibration ON'.
        majority_label = window.mode()[0]
        window_labels.append(majority_label)
        
    return np.array(window_labels)


if __name__ == '__main__':
    # --- Example Usage (Testing the function) ---
    
    # 1. Create dummy filtered data (100 samples total, 3 axes)
    # This simulates data that has been cleaned by highpass_filter.py
    total_samples = 100
    df_test_filtered = pd.DataFrame({
        'accel_x_filtered': np.random.rand(total_samples),
        'accel_y_filtered': np.random.rand(total_samples),
        'accel_z_filtered': np.random.rand(total_samples)
    })
    
    # Create windows using constants (Window Size 64, Slide Step 32)
    segmented_data = create_overlapping_windows(df_test_filtered)
    
    # Expected dimensions:
    # N_windows = floor((Total_samples - Window_size) / Slide_step) + 1
    # N_windows = floor((100 - 64) / 32) + 1 = floor(36 / 32) + 1 = 1 + 1 = 2
    
    print(f"--- Segmentation Test Complete ---")
    print(f"Total Samples In: {total_samples}")
    print(f"Window Size: {WINDOW_SIZE} | Slide Step: {SLIDE_STEP}")
    print(f"Output Array Shape (N_windows, Window_size, N_axes): {segmented_data.shape}")
    print(f"Expected Number of Windows: 2")
    
    # Check the first window's start time (index 0) and the second window's start time (index 32)
    print(f"\nFirst Window starts at Index: 0")
    print(f"Second Window starts at Index: {SLIDE_STEP}")
