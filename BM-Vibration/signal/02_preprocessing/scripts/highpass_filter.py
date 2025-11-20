
import numpy as np
from scipy import signal
import pandas as pd # You'll likely use pandas to load and handle your raw data

# --- Project Constants (Adjust these values based on testing) ---
# The Movesense sensor samples at 52Hz (approx), but let's assume a common rate for now.
# NOTE: Replace '50.0' with the actual sampling rate (fs) of your Movesense sensor
FS = 50.0  # Sampling Frequency (Hz). CHECK YOUR SENSOR'S RATE!

# Cutoff frequency (fc) determines what gets filtered out.
# Gravity and walking are typically below 1 Hz. A cutoff of 0.5-1.5 Hz is common
# to remove these low-frequency components.
# Lower frequency = more noise passes. Higher frequency = tool vibration might be lost.
CUTOFF_FREQ = 0.5  # Hz. Needs empirical testing on your 'noise_walking' data!

# The order (N) of the filter. Higher order = sharper cutoff, but more computation/instability.
# Order 4-10 is a good starting point for IIR filters.
ORDER = 4


def butter_highpass_filter(data, cutoff, fs, order):
    """
    Designs and applies a Butterworth High-Pass Filter to a signal.
    
    This function uses signal.filtfilt for zero phase shift, which is critical
    for preserving the timing accuracy needed for your 'On/Off Classifier'.
    
    Parameters:
    - data (np.array): The input signal array (e.g., one axis of acceleration data).
    - cutoff (float): The cutoff frequency of the filter (Hz).
    - fs (float): The sampling frequency of the data (Hz).
    - order (int): The order of the filter.

    Returns:
    - np.array: The filtered signal array.
    """
    
    # Calculate the Nyquist frequency (half the sampling frequency)
    nyquist = 0.5 * fs
    # Normalize the cutoff frequency to the Nyquist frequency (0 to 1 range)
    normalized_cutoff = cutoff / nyquist
    
    # 1. Design the filter (Get the coefficients B and A)
    # output='ba' returns numerator (b) and denominator (a) polynomials
    # NOTE: You could also use output='sos' (second-order sections) with signal.sosfiltfilt
    # to avoid numerical instability, which is generally safer for high orders.
    b, a = signal.butter(order, normalized_cutoff, btype='highpass', analog=False)
    
    # 2. Apply the filter forward and backward to eliminate phase shift
    # This is crucial for keeping your vibration events accurately timed.
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    
    return filtered_data


def filter_triaxial_data(df_raw, cutoff=CUTOFF_FREQ, fs=FS, order=ORDER):
    """
    Applies the high-pass filter to all three acceleration axes (X, Y, Z).
    
    Parameters:
    - df_raw (pd.DataFrame or np.array): Raw triaxial acceleration data (columns: X, Y, Z).
    """
    
    # Ensure data is a NumPy array for Scipy compatibility
    if isinstance(df_raw, pd.DataFrame):
        data = df_raw[['accel_x', 'accel_y', 'accel_z']].values
    else:
        # Assuming the input is a NumPy array with columns X, Y, Z
        data = df_raw
    
    # Apply the filter to each axis independently
    data_x_filtered = butter_highpass_filter(data[:, 0], cutoff, fs, order)
    data_y_filtered = butter_highpass_filter(data[:, 1], cutoff, fs, order)
    data_z_filtered = butter_highpass_filter(data[:, 2], cutoff, fs, order)
    
    # Combine the filtered axes into a single array/DataFrame
    df_filtered = pd.DataFrame({
        'accel_x_filtered': data_x_filtered,
        'accel_y_filtered': data_y_filtered,
        'accel_z_filtered': data_z_filtered
    })
    
    return df_filtered


if __name__ == '__main__':
    # --- Example Usage (In a real scenario, this loads files from 01_data_collection) ---
    
    # Dummy data: Tool vibration (high freq) + Walking (low freq) + Gravity (DC offset)
    # Movesense sensor data will be much cleaner, but this simulates the challenge.
    
    time = np.linspace(0, 10, int(FS * 10), endpoint=False)
    # 1. Gravity/Walking (low frequency, slow variation) - The NOISE
    low_freq_noise = 9.81 + 0.5 * np.sin(2 * np.pi * 0.2 * time) 
    # 2. Tool Vibration (high frequency) - The SIGNAL
    high_freq_signal = 10 * np.sin(2 * np.pi * 70 * time) 
    
    # Create the RAW signal (simulating one axis, e.g., total_acc_z)
    raw_signal_z = low_freq_noise + high_freq_signal
    
    # Create a DataFrame (simulating your input data structure)
    df_raw_example = pd.DataFrame({
        'accel_x': raw_signal_z, # Just use Z for simplicity in this example
        'accel_y': raw_signal_z,
        'accel_z': raw_signal_z
    })
    
    # Apply the high-pass filter
    df_filtered_example = filter_triaxial_data(df_raw_example)
    
    # The output 'df_filtered_example' should now be centered near zero,
    # containing only the high-frequency vibration component (the tool signal),
    # demonstrating that the walking/gravity noise has been successfully removed.
    
    print("--- Filter Test Complete ---")
    print(f"Filter Cutoff: {CUTOFF_FREQ} Hz | Order: {ORDER}")
    print(f"Mean of RAW signal (should be near 9.81): {np.mean(raw_signal_z):.2f}")
    print(f"Mean of FILTERED signal (should be near 0.0): {np.mean(df_filtered_example['accel_z_filtered']):.2f}")


The mean of the FILTERED signal being close to zero is the key success metric, showing that the constant gravity component has been removed. You can find resources on high-pass filter implementation with Python using the link below. [Simple Lowpass and Highpass Filters with Python Implementation](https://www.youtube.com/watch?v=Aht4letBAmA)


http://googleusercontent.com/youtube_content/0
