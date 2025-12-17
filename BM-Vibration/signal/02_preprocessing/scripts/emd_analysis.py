import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from PyEMD import CEEMDAN
import os
import sys
import json

# Adjust path to use the Movesense loader
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils'))
try:
    from loader_vizualizer_FFT_Welch import load_movesense_json
except ImportError:
    # Fallback if running from scripts dir directly
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils'))
    from loader_vizualizer_FFT_Welch import load_movesense_json

def compute_fft(signal_data, fs):
    """
    Computes FFT for spectral visualization.
    """
    n = len(signal_data)
    fft_vals = np.fft.fft(signal_data)
    fft_freq = np.fft.fftfreq(n, d=1/fs)
    
    pos_mask = fft_freq > 0
    return fft_freq[pos_mask], np.abs(fft_vals)[pos_mask] / n

def run_ceemdan_analysis(df, fs, output_dir, filename_prefix):
    """
    Runs CEEMDAN on the Z-axis acceleration data.
    Visualizes IMFs and their spectra.
    """
    print(f"Running CEEMDAN on {filename_prefix}...")
    
    # Extract Z-axis signal (most relevant for vibration usually)
    # Center the signal (remove DC) for better EMD performance
    S = df['accel_z'].values
    S = S - np.mean(S)
    t = np.arange(len(S)) / fs

    # Configure CEEMDAN
    # noise_strength: standard deviation of the added noise (relative to signal std)
    ceemdan = CEEMDAN(trials=100, epsilon=0.005) 
    # Note: 'trials' is the ensemble size. Higher = better but slower. 
    # 'epsilon' is noise amplitude.
    
    # Execute Decomposition
    print("  Decomposing signal... (this may take a moment)")
    IMFs = ceemdan(S)
    
    num_imfs = IMFs.shape[0]
    print(f"  Extracted {num_imfs} IMFs.")
    
    # --- Visualization ---
    # Plot IMFs in Time Domain and Frequency Domain
    
    fig, axes = plt.subplots(num_imfs + 1, 2, figsize=(16, 3 * (num_imfs + 1)))
    fig.suptitle(f'CEEMDAN Decomposition: {filename_prefix}', fontsize=16)
    
    # Plot Original Signal
    axes[0, 0].plot(t, S, 'k')
    axes[0, 0].set_title("Original Signal (Z-axis)")
    axes[0, 0].set_ylabel("Accel (m/s^2)")
    
    f_orig, mag_orig = compute_fft(S, fs)
    axes[0, 1].plot(f_orig, mag_orig, 'k')
    axes[0, 1].set_title("Spectrum - Original")
    axes[0, 1].set_xlim(0, fs/2)
    axes[0, 1].grid(True)

    # Plot IMFs
    for i in range(num_imfs):
        imf = IMFs[i]
        row = i + 1
        
        # Time Domain
        axes[row, 0].plot(t, imf, 'b')
        axes[row, 0].set_ylabel(f"IMF {i+1}")
        axes[row, 0].grid(True)
        
        # Frequency Domain
        f_imf, mag_imf = compute_fft(imf, fs)
        axes[row, 1].plot(f_imf, mag_imf, 'r')
        axes[row, 1].set_ylabel("Magnitude")
        axes[row, 1].set_xlim(0, fs/2)
        axes[row, 1].grid(True)
        
        # Annotate Peak Frequency
        if len(mag_imf) > 0:
            peak_idx = np.argmax(mag_imf)
            peak_freq = f_imf[peak_idx]
            axes[row, 1].text(0.7, 0.8, f"Peak: {peak_freq:.1f} Hz", 
                              transform=axes[row, 1].transAxes, 
                              bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save Plot
    save_path = os.path.join(output_dir, f"{filename_prefix}_CEEMDAN_Analysis.png")
    plt.savefig(save_path)
    print(f"  Saved plot to: {save_path}")
    plt.close(fig)
    
    # Save IMFs to CSV
    imf_dict = {f'IMF_{i+1}': IMFs[i] for i in range(num_imfs)}
    imf_dict['timestamp'] = df['timestamp']
    df_imfs = pd.DataFrame(imf_dict)
    
    csv_save_path = os.path.join(output_dir, f"{filename_prefix}_IMFs.csv")
    df_imfs.to_csv(csv_save_path, index=False)
    print(f"  Saved IMFs to: {csv_save_path}")

def process_dataset(data_dir, folder_name, filename_keyword=None):
    """
    Finds the target file in the folder and processes it.
    """
    folder_path = os.path.join(data_dir, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # Find the processed CSV (faster) or JSON
    # Priority: processed CSV -> JSON
    
    target_file = None
    
    # 1. Look for processed CSV
    for f in os.listdir(folder_path):
        if f.startswith("processed_") and f.endswith(".csv"):
            target_file = os.path.join(folder_path, f)
            print(f"Found processed CSV: {f}")
            break
            
    # 2. If no CSV, look for JSON in the parent 'data' folder corresponding to this analysis folder?
    # The 'data analysis' structure seems to be folders derived from the 'data' JSONs.
    # Let's assume the user wants to re-process the original JSON if CSV isn't perfect, 
    # but reading the CSV is much faster for EMD.
    
    if target_file:
        df = pd.read_csv(target_file)
    else:
        print(f"No processed data found in {folder_path}. Checking for source JSON...")
        # Try to map folder name to JSON in ../data
        # Folder format: "19. 20251120T140830Z..."
        # JSON format: "20251120T140830Z...json"
        
        # Extract timestamp part
        parts = folder_name.split(' ', 1)
        if len(parts) > 1:
            json_name_part = parts[1]
            json_path = os.path.join(data_dir, '..', f"{json_name_part}.json")
            
            if os.path.exists(json_path):
                print(f"Found source JSON: {json_path}")
                df = load_movesense_json(json_path)
            else:
                print("Could not locate source JSON.")
                return
        else:
            return

    if df is None or df.empty:
        print("Data load failed.")
        return

    # Determine Sampling Frequency
    # Calculate effective Fs
    timestamps = df['timestamp'].values
    duration_sec = (timestamps[-1] - timestamps[0]) / 1000.0
    fs_est = len(timestamps) / duration_sec
    print(f"Effective Fs: {fs_est:.2f} Hz")
    
    # Create output sub-directory
    output_dir = os.path.join(folder_path, "EMD_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Analysis
    run_ceemdan_analysis(df, fs_est, output_dir, folder_name)


if __name__ == "__main__":
    # Define Base Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # data analysis folder is sibling to 'signal' -> 'BM-Vibration/data analysis'
    base_analysis_dir = os.path.join(script_dir, '..', '..', '..', 'data_analysis')
    base_analysis_dir = os.path.normpath(base_analysis_dir)
    
    print(f"Scanning directory: {base_analysis_dir}")
    
    # Target specific folders first (19, 20, 22) as requested
    target_folders = [
        "28. PLSH-test(20s)", # PLSH tool
        "29. PMF-Test1(45s)", # PMF tool
    ]
    
    # Interactive mode or Batch? Let's do the requested batch.
    
    print("\n--- Processing Priority Folders ---")
    for folder in target_folders:
        process_dataset(base_analysis_dir, folder)
        
    # Ask to process others
    response = input("\nProcess ALL other folders in 'data_analysis'? (y/n): ")
    if response.lower() == 'y':
        all_folders = sorted([d for d in os.listdir(base_analysis_dir) if os.path.isdir(os.path.join(base_analysis_dir, d))])
        for folder in all_folders:
            if folder not in target_folders:
                # Skip numeric prefixes if they don't look like data folders? 
                # Your folders seem to start with "1. ", "2. ", etc.
                process_dataset(base_analysis_dir, folder)
                
    print("\nDone.")

