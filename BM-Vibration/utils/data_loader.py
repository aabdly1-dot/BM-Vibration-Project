import pandas as pd
import numpy as np
import os

# ==============================================================================
# DATA LOADER UTILITY: data_loader.py
# Purpose: Load, merge, and organize raw triaxial sensor data and activity labels
#          from the UCI HAR Dataset (or similar structure).
# ==============================================================================

# --- 1. Project Constants (Based on UCI HAR Dataset Structure) ---

# Base path structure where the UCI HAR Dataset folder is located 
# NOTE: Adjust this BASE_PATH to point to your main project directory.
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'UCI HAR Dataset')

# Standard column names for the output DataFrame
COLUMN_NAMES = ['accel_x', 'accel_y', 'accel_z', 'label_id', 'activity_name']


def load_activity_labels():
    """
    Loads the activity name to ID mapping from activity_labels.txt.
    
    Returns:
    - pd.Series: Maps activity ID (int) to name (string).
    """
    # File path for the labels key
    labels_path = os.path.join(BASE_PATH, 'activity_labels.txt')
    
    # Read the file: format is 'ID ACTIVITY_NAME' (e.g., '1 WALKING')
    df_labels = pd.read_csv(labels_path, header=None, sep=' ', names=['id', 'name'])
    
    # Map from ID (index 0) to Name (index 1)
    return df_labels.set_index('id')['name']


def load_raw_data_set(set_name='train'):
    """
    Loads raw triaxial acceleration data (total_acc) and links it to activity labels.
    
    Parameters:
    - set_name (str): Either 'train' or 'test'.
    
    Returns:
    - pd.DataFrame: DataFrame containing merged sensor data and activity names.
    """
    if set_name not in ['train', 'test']:
        raise ValueError("set_name must be 'train' or 'test'.")
        
    # --- 2. Define File Paths ---
    set_dir = os.path.join(BASE_PATH, set_name)
    inertial_dir = os.path.join(set_dir, 'Inertial Signals')

    # File names for raw total acceleration signals
    signal_files = {
        'accel_x': os.path.join(inertial_dir, f'total_acc_x_{set_name}.txt'),
        'accel_y': os.path.join(inertial_dir, f'total_acc_y_{set_name}.txt'),
        'accel_z': os.path.join(inertial_dir, f'total_acc_z_{set_name}.txt'),
    }
    
    # File names for labels
    label_path = os.path.join(set_dir, f'y_{set_name}.txt')
    
    # --- 3. Load Signals (X, Y, Z) and Labels ---
    
    # Read raw signal files. Each file contains columns of time series data.
    # The data is already formatted such that rows correspond to time-steps/instances.
    df_x = pd.read_csv(signal_files['accel_x'], header=None, delim_whitespace=True)
    df_y = pd.read_csv(signal_files['accel_y'], header=None, delim_whitespace=True)
    df_z = pd.read_csv(signal_files['accel_z'], header=None, delim_whitespace=True)
    
    # Read labels (Activity IDs)
    df_y = pd.read_csv(label_path, header=None, names=['label_id'])
    
    # --- 4. Merge Data (Flatten 7352 rows x 128 features to 941056 rows x 3 features) ---
    
    # Flatten the time series data for each axis into a single column
    # For UCI HAR, 7352 rows x 128 features is flattened to 941,056 rows x 1 feature
    df_x_flat = df_x.stack().reset_index(level=[0, 1], drop=True).rename('accel_x')
    df_y_flat = df_y.stack().reset_index(level=[0, 1], drop=True).rename('accel_y')
    df_z_flat = df_z.stack().reset_index(level=[0, 1], drop=True).rename('accel_z')
    
    # The label must also be expanded to match the new, flattened length
    # Repeat each label 128 times (the window size of the UCI HAR features)
    WINDOW_SIZE_HAR = df_x.shape[1] # 128 features/window in this dataset
    df_y_flat = df_y.reindex(df_y.index.repeat(WINDOW_SIZE_HAR)).reset_index(drop=True)
    
    # Combine all flattened data
    df_combined = pd.concat([df_x_flat, df_y_flat, df_z_flat, df_y_flat], axis=1)
    df_combined.columns = ['accel_x', 'accel_y', 'accel_z', 'label_id']
    
    # --- 5. Link Labels and Activity Names ---
    activity_map = load_activity_labels()
    df_combined['activity_name'] = df_combined['label_id'].map(activity_map)
    
    return df_combined


def filter_activities(df_combined, activity_names):
    """
    Filters the combined DataFrame to include only the specified list of activities.
    
    Parameters:
    - df_combined (pd.DataFrame): Output from load_raw_data_set.
    - activity_names (list): List of activity names (strings) to keep.
    
    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    return df_combined[df_combined['activity_name'].isin(activity_names)].copy()


if __name__ == '__main__':
    # --- Demonstration: Load and Filter ONLY Walking Data (Your Noise) ---
    
    print("--- Running Data Loader Demonstration ---")
    
    # 1. Load the full raw training set
    df_raw_full = load_raw_data_set(set_name='train')
    
    print(f"Total rows loaded: {len(df_raw_full):,}")
    print(f"Unique activities found: {df_raw_full['activity_name'].unique()}")
    print("-" * 40)
    
    # 2. Filter the data to include ONLY the Noise Data required for filtering experiments
    NOISE_ACTIVITIES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING']
    
    df_noise_data = filter_activities(df_raw_full, NOISE_ACTIVITIES)
    
    print(f"Total rows filtered for Noise Data: {len(df_noise_data):,}")
    print(f"Filtered activities: {df_noise_data['activity_name'].unique()}")
    
    # This 'df_noise_data' is now ready to be used by the 'filtering_experiments.ipynb'
    # and the 'highpass_filter.py' for validation.
