import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from sklearn.cluster import KMeans
from PyEMD import CEEMDAN
import os
import sys

# Adjust path to use the Movesense loader if needed, though we prefer CSVs here
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
try:
    from loader_vizualizer_FFT_Welch import load_movesense_json
except ImportError:
    pass # We will rely on CSV loading mostly

def load_data(path):
    """
    Loads data from CSV or JSON.
    Returns: t (time in s), amag (magnitude array)
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        # Handle potential timestamp formats
        # Assuming processed CSV has 'timestamp', 'accel_x', 'accel_y', 'accel_z'
        if 'timestamp' in df.columns:
            t = (df['timestamp'] - df['timestamp'].iloc[0]).values / 1000.0
        else:
            # Fallback if no timestamp column (just indices)
            # Need FS to create time vector. Assuming standard processed file has it.
            # Let's try to infer FS or just use index if strictly necessary
            t = np.arange(len(df)) # Placeholder, will adjust if FS known
            
        ax = df['accel_x'].values
        ay = df['accel_y'].values
        az = df['accel_z'].values
        amag = np.sqrt(ax**2 + ay**2 + az**2)
        return t, amag, df
        
    elif path.endswith('.json'):
        # Use the existing utility to parse JSON
        try:
            from loader_vizualizer_FFT_Welch import load_movesense_json
            df = load_movesense_json(path)
            if df is not None:
                t = (df['timestamp'] - df['timestamp'].iloc[0]).values / 1000.0
                ax = df['accel_x'].values
                ay = df['accel_y'].values
                az = df['accel_z'].values
                amag = np.sqrt(ax**2 + ay**2 + az**2)
                return t, amag, df
        except ImportError:
            print("Error: Could not import JSON loader. Please use processed CSV files.")
            return None, None, None
            
    return None, None, None

def bandpass(signal_data, fs, f_low=30.0, f_high=500.0, order=4):
    """
    Bandpass filter to isolate tool vibration range.
    """
    nyq = 0.5 * fs
    low = f_low / nyq
    high = f_high / nyq
    
    # Safety bounds
    if low <= 0: low = 0.001
    if high >= 1: high = 0.999
    
    # CRITICAL FIX: Check if low < high.
    # If Fs is very low (e.g. 55Hz -> Nyquist 27.5Hz), then f_low (30Hz) > Nyquist!
    # In that case, bandpass is impossible. We should skip filtering or just do highpass/lowpass.
    
    if low >= high:
        print(f"Warning: Filter range [{f_low}-{f_high} Hz] is invalid for Fs={fs:.1f} Hz (Nyquist={nyq:.1f} Hz).")
        # If signal is aliased/low-fs, we can't filter for tool (30Hz+) properly.
        # Fallback: If Nyquist < f_low, return original signal (or zeros?)?
        # Better: Return original, as we can't extract the band.
        return signal_data 

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal_data)

def short_time_rms(x, fs, win_sec=0.5, overlap=0.5):
    """
    Computes Short-Time RMS energy.
    """
    nperseg = int(win_sec * fs)
    step = int(nperseg * (1 - overlap))
    
    # Ensure step is at least 1
    step = max(1, step)
    
    starts = np.arange(0, len(x) - nperseg + 1, step)
    rms = np.zeros(len(starts))
    
    for i, s in enumerate(starts):
        seg = x[s:s + nperseg]
        rms[i] = np.sqrt(np.mean(seg**2))
        
    return rms, starts, nperseg

def kmeans_labels(rms):
    """
    Clusters RMS values into 2 states: Active (High RMS) vs Rest (Low RMS).
    """
    if len(rms) < 2:
        return np.zeros(len(rms), dtype=int)
        
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = kmeans.fit_predict(rms.reshape(-1, 1))
    
    means = np.array([rms[labels == k].mean() for k in [0, 1]])
    active_cluster = np.argmax(means)
    
    state = np.where(labels == active_cluster, 1, 0)  # 1 = ACTIVE, 0 = REST
    return state

def expand_state_to_samples(state, starts, nperseg, n_samples):
    """
    Expands window-level labels back to sample-level mask.
    """
    labels = np.zeros(n_samples, dtype=int)
    
    # Simple expansion: fill window with label
    # Since windows overlap, later windows overwrite earlier ones.
    # Ideally, we'd average or vote, but simple overwrite works for segmentation visualization
    for st, lab in zip(starts, state):
        labels[st:st + nperseg] = lab
        
    return labels

def compute_spectrogram(amag, fs, nperseg=1024, noverlap=512):
    f, t_spec, Sxx = spectrogram(amag, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return f, t_spec, Sxx_db

def run_ceemdan_on_segment(signal_data, fs, label_name, output_dir):
    """
    Runs CEEMDAN on a specific signal segment (concatenated Active or Rest).
    """
    if len(signal_data) < fs * 0.5: # Skip if too short (< 0.5s)
        print(f"Segment {label_name} too short for CEEMDAN.")
        return

    print(f"Running CEEMDAN on {label_name} segment ({len(signal_data)/fs:.2f}s)...")
    
    # Center signal
    S = signal_data - np.mean(signal_data)
    t = np.arange(len(S)) / fs
    
    ceemdan = CEEMDAN(trials=50, epsilon=0.005) # Fewer trials for speed in batch
    IMFs = ceemdan(S)
    num_imfs = IMFs.shape[0]
    
    # Plot
    fig, axes = plt.subplots(num_imfs + 1, 1, figsize=(10, 2 * (num_imfs + 1)), sharex=True)
    fig.suptitle(f'CEEMDAN: {label_name} Segment', fontsize=16)
    
    axes[0].plot(t, S, 'k')
    axes[0].set_title("Original Concatenated Segment")
    axes[0].set_ylabel("Accel")
    
    for i in range(num_imfs):
        axes[i+1].plot(t, IMFs[i], 'b')
        axes[i+1].set_ylabel(f"IMF {i+1}")
        
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, f"CEEMDAN_{label_name}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved CEEMDAN plot to {save_path}")

def process_dataset(folder_path, output_dir_base):
    """
    Main processing logic for one dataset folder.
    """
    print(f"\nProcessing folder: {folder_path}")
    
    # Find CSV
    target_file = None
    for f in os.listdir(folder_path):
        if f.startswith("processed_") and f.endswith(".csv"):
            target_file = os.path.join(folder_path, f)
            break
            
    if not target_file:
        print("No processed CSV found.")
        return

    t, amag, df = load_data(target_file)
    if t is None: return

    # Calculate FS
    if len(t) > 1:
        fs = 1.0 / np.mean(np.diff(t))
    else:
        fs = 833.0 # Default
    print(f"Estimated Fs: {fs:.1f} Hz")

    # 1. Feature Extraction & Labeling
    # Filter for vibration band (Tools usually > 30Hz, Walking < 10Hz)
    # Using 30-500Hz bandpass to detect "Active Tool" energy
    amag_bp = bandpass(amag, fs, f_low=30.0, f_high=400.0) # 400Hz is safe Nyquist limit for 833Hz
    
    rms, starts, nperseg = short_time_rms(amag_bp, fs)
    state_win = kmeans_labels(rms)
    state_samples = expand_state_to_samples(state_win, starts, nperseg, len(amag))
    
    # 2. Spectrogram
    f, t_spec, Sxx_db = compute_spectrogram(amag, fs)
    
    # 3. Plotting (Bishop Style)
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 0.5]})
    
    # Subplot 1: Spectrogram
    im = axs[0].pcolormesh(t_spec, f, Sxx_db, shading="gouraud", cmap='inferno')
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_title("Full Spectrogram")
    fig.colorbar(im, ax=axs[0], label="Power (dB)")
    
    # Subplot 2: Waveform
    axs[1].plot(t, amag, linewidth=0.5, color='k', alpha=0.7)
    axs[1].set_ylabel("Magnitude (m/s^2)")
    axs[1].set_title("Waveform Magnitude")
    
    # Subplot 3: Label Track
    # Create a color mesh for the bar
    cmap = plt.get_cmap('bwr') # Blue=Rest (0), Red=Active (1)
    axs[2].imshow(state_samples[np.newaxis, :], 
                  extent=[t[0], t[-1], 0, 1], 
                  aspect="auto", 
                  cmap=cmap, vmin=0, vmax=1)
    axs[2].set_yticks([])
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Automated Segmentation (Blue=Rest, Red=Active Tool)")
    
    plt.tight_layout()
    
    # Save Main Figure
    save_name = os.path.basename(folder_path) + "_BishopPlot.png"
    save_path = os.path.join(folder_path, save_name)
    plt.savefig(save_path)
    print(f"Saved Bishop Plot to: {save_path}")
    plt.close(fig)
    
    # 4. Additional Outputs: Concatenated Segments Analysis
    # Extract Active and Rest segments from original Z-axis signal (for CEEMDAN)
    # Using Z-axis ('accel_z') often captures vibration best
    raw_signal = df['accel_z'].values
    
    active_signal = raw_signal[state_samples == 1]
    rest_signal = raw_signal[state_samples == 0]
    
    # Create subfolder for detailed results
    res_dir = os.path.join(folder_path, "Detailed_Analysis")
    os.makedirs(res_dir, exist_ok=True)
    
    # Run CEEMDAN on segments
    if len(active_signal) > 0:
        run_ceemdan_on_segment(active_signal, fs, "ACTIVE_Tool", res_dir)
    
    if len(rest_signal) > 0:
        run_ceemdan_on_segment(rest_signal, fs, "REST_Idle", res_dir)
        
    # Separate Spectrograms (Optional - simple plot)
    if len(active_signal) > fs*0.1:
        plt.figure(figsize=(10, 4))
        f_a, t_a, Sxx_a = spectrogram(active_signal, fs=fs)
        plt.pcolormesh(t_a, f_a, 10*np.log10(Sxx_a + 1e-12), shading='gouraud', cmap='inferno')
        plt.title("Spectrogram: ACTIVE Segments Only")
        plt.ylabel("Freq (Hz)")
        plt.xlabel("Time (concatenated)")
        plt.colorbar(label="dB")
        plt.savefig(os.path.join(res_dir, "Spectrogram_ACTIVE.png"))
        plt.close()

if __name__ == "__main__":
    # Define Base Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming this script is in BM-Vibration/data analysis/
    base_analysis_dir = script_dir # Current dir is root for analysis
    
    print(f"Scanning directory: {base_analysis_dir}")
    
    # List directories
    # Filter for directories only, and ensure we handle spaces correctly in names
    all_folders = sorted(
        [
            d for d in os.listdir(base_analysis_dir)
            if os.path.isdir(os.path.join(base_analysis_dir, d)) and not d.startswith('.') and not d == "__pycache__"
        ],
        key=lambda x: (int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf'), x)
    )

    
    
    print("\n--- Available Folders ---")
    for i, f in enumerate(all_folders):
        print(f) # Quote names with spaces for clarity in display

        
    # Selection
    while True:
        inp = input("\nEnter folder number to process (or 'all', 'q'): ")
        if inp.lower() == 'q':
            break
        
        folders_to_process = []
        if inp.lower() == 'all':
            folders_to_process = all_folders
        else:
            try:
                idx = int(inp) - 1
                if 0 <= idx < len(all_folders):
                    folders_to_process = [all_folders[idx]]
                else:
                    print("Invalid index.")
                    continue
            except ValueError:
                print("Invalid input.")
                continue
                
        for folder in folders_to_process:
            full_path = os.path.join(base_analysis_dir, folder)
            process_dataset(full_path, base_analysis_dir)
            
        if inp.lower() != 'all':
            print("\nFinished.")

