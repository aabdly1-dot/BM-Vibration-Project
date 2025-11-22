import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import sys

def load_movesense_json(file_path):
    """
    Parses a Movesense JSON file and converts it into a DataFrame.
    
    Expected structure:
    {
        "data": [
            {
                "acc": {
                    "Timestamp": <int>,
                    "ArrayAcc": [
                        {"x": <float>, "y": <float>, "z": <float>},
                        ...
                    ]
                }
            },
            ...
        ]
    }
    """
    print(f"Loading file: {file_path}")
    with open(file_path, 'r') as f:
        try:
            data_json = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None

    if 'data' not in data_json:
        print("Error: JSON does not contain 'data' key.")
        return None

    # Extract data
    all_samples = []
    
    # We need to handle timestamps carefully. 
    # Movesense timestamps are usually for the last sample in the packet or the packet transmission time.
    # We will collect all packet timestamps and samples to interpolate later.
    
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
        print("No valid 'acc' data found.")
        return None

    # Flatten samples
    # Strategy for timestamps:
    # The timestamp in the packet usually refers to the end of the packet capture.
    # We can estimate Fs from the average time difference between packets divided by samples per packet.
    
    # Convert timestamps to numpy array
    t_packets = np.array(packet_timestamps)
    
    # Handle potential rollover or non-monotonic timestamps if necessary (assuming clean for now)
    # Calculate effective sampling rate from total duration
    if len(t_packets) > 1:
        total_duration_ms = t_packets[-1] - t_packets[0]
        total_samples_between = sum(packet_sample_counts[:-1]) # Samples up to the last packet start
        # Actually, it's better to use all gaps
        
        # Calculate average time per packet
        dt_packets = np.diff(t_packets)
        avg_dt_packet = np.mean(dt_packets)
        avg_samples_per_packet = np.mean(packet_sample_counts)
        
        # Effective sample interval (ms)
        sample_interval_ms = avg_dt_packet / avg_samples_per_packet
    else:
        sample_interval_ms = 1000.0 / 833.0 # Default assumption if only 1 packet
    
    print(f"Estimated sample interval: {sample_interval_ms:.4f} ms")
    
    # Generate timestamps for each sample
    # We will reconstruct absolute timestamps by working backwards from packet timestamp
    # or interpolating. Linear interpolation between packet timestamps is robust.
    
    final_timestamps = []
    final_acc_x = []
    final_acc_y = []
    final_acc_z = []
    
    # Iterate through packets
    current_time = t_packets[0] - (packet_sample_counts[0] * sample_interval_ms) # Estimate start
    
    # A better approach: Interpolate time between packet T_i and T_{i+1}
    # But for the first packet, we only have T_0.
    # Let's simply use the estimated sample_interval_ms to space points within a packet,
    # anchored at the packet timestamp.
    
    for i, packet_samples in enumerate(raw_samples_per_packet):
        packet_ts = packet_timestamps[i]
        num_samples = len(packet_samples)
        
        # Create time points for this packet ending at packet_ts
        # t_sample = packet_ts - (N - 1 - k) * dt
        # This assumes packet_ts is the time of the LAST sample.
        packet_times = [packet_ts - (num_samples - 1 - k) * sample_interval_ms for k in range(num_samples)]
        
        final_timestamps.extend(packet_times)
        for sample in packet_samples:
            final_acc_x.append(sample.get('x', 0.0))
            final_acc_y.append(sample.get('y', 0.0))
            final_acc_z.append(sample.get('z', 0.0))

    df = pd.DataFrame({
        'timestamp': final_timestamps,
        'accel_x': final_acc_x,
        'accel_y': final_acc_y,
        'accel_z': final_acc_z
    })
    
    return df

def check_signal_quality(df, nominal_fs=None):
    """
    Checks signal quality: Effective Fs vs Nominal Fs, DC offset.
    """
    print("\n--- Signal Quality Check ---")
    
    # 1. Calculate Effective Fs
    timestamps = df['timestamp'].values
    if len(timestamps) > 1:
        # Calculate average delta t in seconds (assuming timestamps are in ms)
        duration_sec = (timestamps[-1] - timestamps[0]) / 1000.0
        n_samples = len(timestamps)
        effective_fs = n_samples / duration_sec if duration_sec > 0 else 0
        print(f"Effective Sampling Frequency: {effective_fs:.2f} Hz")
    else:
        effective_fs = 0
        print("Not enough data to calculate Fs.")

    # 2. Compare with Nominal Fs
    if nominal_fs:
        print(f"Nominal Sampling Frequency: {nominal_fs} Hz")
        if effective_fs > 0:
            deviation = abs(effective_fs - nominal_fs) / nominal_fs * 100
            print(f"Deviation: {deviation:.2f}%")
            if deviation > 5:
                print("WARNING: Significant deviation (>5%) from nominal Fs!")
        else:
            print("Cannot compare: Effective Fs is 0.")
    
    # 3. Check DC Offset (Mean)
    print("DC Offsets (Mean values):")
    print(f"  X: {df['accel_x'].mean():.4f} m/s^2")
    print(f"  Y: {df['accel_y'].mean():.4f} m/s^2")
    print(f"  Z: {df['accel_z'].mean():.4f} m/s^2")
    
    # Calculate Magnitude
    df['magnitude'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    print(f"  Magnitude Mean: {df['magnitude'].mean():.4f} m/s^2 (Expected ~9.81 for static)")

    return effective_fs

def plot_time_series(df):
    """
    Plots time series for X, Y, Z and Magnitude.
    """
    timestamps_sec = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000.0
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Time Series Acceleration Data', fontsize=16)
    
    axs[0].plot(timestamps_sec, df['accel_x'], color='r', label='X')
    axs[0].set_ylabel('Accel X (m/s^2)')
    axs[0].legend(loc='upper right')
    
    axs[1].plot(timestamps_sec, df['accel_y'], color='g', label='Y')
    axs[1].set_ylabel('Accel Y (m/s^2)')
    axs[1].legend(loc='upper right')
    
    axs[2].plot(timestamps_sec, df['accel_z'], color='b', label='Z')
    axs[2].set_ylabel('Accel Z (m/s^2)')
    axs[2].legend(loc='upper right')
    
    axs[3].plot(timestamps_sec, df['magnitude'], color='k', label='Magnitude')
    axs[3].set_ylabel('Magnitude (m/s^2)')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def compute_fft(signal_data, fs):
    """
    Computes the FFT of a signal.
    """
    n = len(signal_data)
    fft_vals = np.fft.fft(signal_data)
    fft_freq = np.fft.fftfreq(n, d=1/fs)
    
    # Keep only positive frequencies
    pos_mask = fft_freq > 0
    fft_freq = fft_freq[pos_mask]
    fft_mag = np.abs(fft_vals)[pos_mask] / n  # Normalize
    
    return fft_freq, fft_mag

def compute_welch_psd(signal_data, fs, window_sec=1.0, overlap_ratio=0.5):
    """
    Computes Power Spectral Density using Welch's method.
    """
    nperseg = int(window_sec * fs)
    noverlap = int(nperseg * overlap_ratio)
    
    freqs, psd = signal.welch(signal_data, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    return freqs, psd

def plot_spectral_analysis(df, fs, stable_start_idx=None, stable_end_idx=None):
    """
    Plots FFT (Full), FFT (Stable), and Welch PSD.
    """
    # If no stable segment defined, use the middle 50% of the file
    if stable_start_idx is None or stable_end_idx is None:
        n_samples = len(df)
        stable_start_idx = int(n_samples * 0.25)
        stable_end_idx = int(n_samples * 0.75)
        print(f"No stable segment selected. Using middle 50%: Index {stable_start_idx} to {stable_end_idx}")

    stable_segment = df.iloc[stable_start_idx:stable_end_idx]
    
    axes = ['accel_x', 'accel_y', 'accel_z', 'magnitude']
    colors = ['r', 'g', 'b', 'k']
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f'Spectral Analysis (Fs={fs:.1f} Hz)', fontsize=16)
    
    # 1. FFT of Full File
    axs[0].set_title("FFT - Full File")
    for ax_name, color in zip(axes, colors):
        freq, mag = compute_fft(df[ax_name].values, fs)
        axs[0].plot(freq, mag, label=ax_name, color=color, alpha=0.7)
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Magnitude")
    axs[0].legend()
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 2. FFT of Stable Segment
    axs[1].set_title(f"FFT - Stable Segment (Idx {stable_start_idx}-{stable_end_idx})")
    for ax_name, color in zip(axes, colors):
        freq, mag = compute_fft(stable_segment[ax_name].values, fs)
        axs[1].plot(freq, mag, label=ax_name, color=color, alpha=0.7)
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].legend()
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 3. Welch PSD (Stable Segment)
    axs[2].set_title("Welch PSD - Stable Segment")
    for ax_name, color in zip(axes, colors):
        freq, psd = compute_welch_psd(stable_segment[ax_name].values, fs)
        axs[2].semilogy(freq, psd, label=ax_name, color=color, alpha=0.7) # Log scale for PSD
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("PSD (V^2/Hz or g^2/Hz)")
    axs[2].legend()
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def save_to_csv(df, original_filename, output_dir):
    """
    Saves the processed DataFrame to a CSV file.
    """
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    output_path = os.path.join(output_dir, f"processed_{base_name}.csv")
    
    print(f"Saving processed data to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Save successful.")

if __name__ == "__main__":
    # --- Interactive Setup ---
    
    # Define paths
    # Assumes script is in BM-Vibration/utils
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust path to data directory (../data relative to utils)
    data_dir = os.path.join(script_dir, '..', 'data')
    
    # Normalize path
    data_dir = os.path.normpath(data_dir)
    
    print(f"Looking for data in: {data_dir}")
    
    if not os.path.exists(data_dir):
        print("Data directory not found!")
        sys.exit(1)
        
    # List JSON files
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not files:
        print("No .json files found in the data directory.")
        sys.exit(1)
        
    print("\n--- Available Files ---")
    for i, f in enumerate(files):
        print(f"{i + 1}. {f}")
        
    # User Selection
    while True:
        try:
            selection = input("\nSelect a file number to process (or 'q' to quit): ")
            if selection.lower() == 'q':
                sys.exit(0)
            
            file_idx = int(selection) - 1
            if 0 <= file_idx < len(files):
                selected_file = files[file_idx]
                break
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    full_file_path = os.path.join(data_dir, selected_file)
    
    # User Nominal Fs Input (Optional)
    nominal_fs_input = input("Enter nominal sampling frequency (Hz) [Default: 833]: ").strip()
    nominal_fs = float(nominal_fs_input) if nominal_fs_input else 833.0
    
    # --- Pipeline Execution ---
    
    # 1. Load
    df_data = load_movesense_json(full_file_path)
    
    if df_data is not None and not df_data.empty:
        # 2. Quality Checks
        effective_fs = check_signal_quality(df_data, nominal_fs)
        
        # 3. Visualization (Time Series)
        print("Displaying Time Series... (Close plot to continue)")
        plot_time_series(df_data)
        
        # 4. Spectral Analysis
        # Ask user for stable segment? Or just use default middle?
        print("Displaying Spectral Analysis... (Close plot to continue)")
        
        # Use effective Fs if valid, else nominal
        fs_to_use = effective_fs if effective_fs > 1 else nominal_fs
        
        plot_spectral_analysis(df_data, fs_to_use)
        
        # 5. Save to CSV
        save_to_csv(df_data, selected_file, data_dir)
        
    else:
        print("Failed to load data.")

