
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import pandas as pd

# --- Project Constants (Adjust these values based on testing) ---
FS = 833.0  # Default Sampling Frequency (Hz) - Movesense typical rate

# Highpass filter settings (removes gravity/walking < 0.5 Hz)
CUTOFF_FREQ = 0.5  # Hz

# Bandpass filter settings (isolates tool vibration band)
BANDPASS_LOW = 40.0   # Hz - lower bound for tool vibrations
BANDPASS_HIGH = 400.0 # Hz - upper bound for tool vibrations

# Filter order
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
    normalized_cutoff = cutoff / nyquist # TODO undertand what is the cut-off frequency (the specific number), 
    # adaptive cut-off might be good, but only after working with dataset (Power Spectrum Density, FFT, Welch FFT) and understanding the limits of freq for each tool or walking
    
    # 1. Design the filter (Get the coefficients B and A)
    # output='ba' returns numerator (b) and denominator (a) polynomials
    # NOTE: You could also use output='sos' (second-order sections) with signal.sosfiltfilt
    # to avoid numerical instability, which is generally safer for high orders.
    b, a = signal.butter(order, normalized_cutoff, btype='highpass', analog=False)
    
    # 2. Apply the filter forward and backward to eliminate phase shift
    # This is crucial for keeping your vibration events accurately timed.
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    
    return filtered_data


def butter_bandpass_filter(data, fs, f_low=BANDPASS_LOW, f_high=BANDPASS_HIGH, order=ORDER):
    """
    Applies a Nyquist-aware Butterworth Band-Pass Filter to isolate tool vibration frequencies.
    
    This filter is designed to be robust across different sampling frequencies by
    automatically clamping the cutoff frequencies to valid Nyquist range.
    
    Parameters:
    - data (np.array): The input signal array.
    - fs (float): The sampling frequency of the data (Hz).
    - f_low (float): Lower cutoff frequency (Hz). Default 40 Hz.
    - f_high (float): Upper cutoff frequency (Hz). Default 400 Hz.
    - order (int): The order of the filter.

    Returns:
    - np.array: The filtered signal array, or original if filter cannot be applied.
    """
    nyquist = 0.5 * fs
    
    # Safety: clamp frequencies to valid Nyquist range (0, 1) exclusive
    low = max(f_low / nyquist, 0.001)
    high = min(f_high / nyquist, 0.999)
    
    # Check if bandpass is possible with this sampling frequency
    if low >= high:
        print(f"WARNING: Bandpass filter skipped (Fs={fs}Hz too low for {f_low}-{f_high}Hz band)")
        return data
    
    if low >= 1.0:
        print(f"WARNING: Bandpass filter skipped (f_low={f_low}Hz >= Nyquist={nyquist}Hz)")
        return data
    
    try:
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        print(f"WARNING: Bandpass filter failed ({e}). Returning original signal.")
        return data


def filter_triaxial_data(df_raw, cutoff=CUTOFF_FREQ, fs=FS, order=ORDER, apply_bandpass=False,
                         bandpass_low=BANDPASS_LOW, bandpass_high=BANDPASS_HIGH):
    """
    Applies high-pass filter (and optionally bandpass) to all three acceleration axes (X, Y, Z).
    
    Parameters:
    - df_raw (pd.DataFrame or np.array): Raw triaxial acceleration data (columns: X, Y, Z).
    - cutoff (float): Highpass cutoff frequency (Hz).
    - fs (float): Sampling frequency (Hz).
    - order (int): Filter order.
    - apply_bandpass (bool): If True, apply additional bandpass filter after highpass.
    - bandpass_low (float): Lower bandpass cutoff frequency (Hz).
    - bandpass_high (float): Upper bandpass cutoff frequency (Hz).
    
    Returns:
    - pd.DataFrame: Filtered triaxial data with columns accel_x_filtered, accel_y_filtered, accel_z_filtered.
    """
    
    # Ensure data is a NumPy array for Scipy compatibility
    if isinstance(df_raw, pd.DataFrame):
        data = df_raw[['accel_x', 'accel_y', 'accel_z']].values
    else:
        # Assuming the input is a NumPy array with columns X, Y, Z
        data = df_raw
    
    # Step 1: Apply highpass filter to each axis (removes gravity/walking)
    data_x_filtered = butter_highpass_filter(data[:, 0], cutoff, fs, order)
    data_y_filtered = butter_highpass_filter(data[:, 1], cutoff, fs, order)
    data_z_filtered = butter_highpass_filter(data[:, 2], cutoff, fs, order)
    
    # Step 2: Optionally apply bandpass filter (isolates tool vibration band)
    if apply_bandpass:
        data_x_filtered = butter_bandpass_filter(data_x_filtered, fs, bandpass_low, bandpass_high, order)
        data_y_filtered = butter_bandpass_filter(data_y_filtered, fs, bandpass_low, bandpass_high, order)
        data_z_filtered = butter_bandpass_filter(data_z_filtered, fs, bandpass_low, bandpass_high, order)
    
    # Combine the filtered axes into a single DataFrame
    df_filtered = pd.DataFrame({
        'accel_x_filtered': data_x_filtered,
        'accel_y_filtered': data_y_filtered,
        'accel_z_filtered': data_z_filtered
    })
    
    return df_filtered


if __name__ == '__main__':
    """
    Example Usage: Test both highpass and bandpass filters.
    """
    
    # Create dummy data: Tool vibration (high freq) + Walking (low freq) + Gravity (DC offset)
    time = np.linspace(0, 10, int(FS * 10), endpoint=False)
    
    # 1. Gravity/Walking (low frequency, slow variation) - The NOISE
    low_freq_noise = 9.81 + 0.5 * np.sin(2 * np.pi * 0.2 * time) 
    # 2. Tool Vibration (high frequency) - The SIGNAL
    high_freq_signal = 10 * np.sin(2 * np.pi * 70 * time) 
    
    # Create the RAW signal
    raw_signal_z = low_freq_noise + high_freq_signal
    
    # Create a DataFrame
    df_raw_example = pd.DataFrame({
        'accel_x': raw_signal_z,
        'accel_y': raw_signal_z,
        'accel_z': raw_signal_z
    })
    
    # Test 1: Highpass filter only
    print("--- Test 1: Highpass Filter Only ---")
    df_hp_only = filter_triaxial_data(df_raw_example, fs=FS, apply_bandpass=False)
    print(f"Highpass Cutoff: {CUTOFF_FREQ} Hz | Order: {ORDER}")
    print(f"Mean of RAW signal (should be near 9.81): {np.mean(raw_signal_z):.2f}")
    print(f"Mean of HP FILTERED (should be near 0.0): {np.mean(df_hp_only['accel_z_filtered']):.2f}")
    
    # Test 2: Highpass + Bandpass filter
    print("\n--- Test 2: Highpass + Bandpass Filter ---")
    df_hp_bp = filter_triaxial_data(df_raw_example, fs=FS, apply_bandpass=True)
    print(f"Bandpass: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print(f"Mean of HP+BP FILTERED (should be near 0.0): {np.mean(df_hp_bp['accel_z_filtered']):.2f}")
    
    # Test 3: Low Fs scenario (should warn and skip bandpass)
    print("\n--- Test 3: Low Fs Warning Test ---")
    df_low_fs = filter_triaxial_data(df_raw_example, fs=55.0, apply_bandpass=True)
    print("Test complete (check for warning above).")
