"""
Robust ON/OFF Classifier with Multi-Feature Extraction

This module provides a robust vibration detection classifier that uses multiple
spectral features to distinguish between tool active (ON) and inactive (OFF) states.

Features used:
- RMS (Root Mean Square)
- Spectral Centroid
- Spectral Bandwidth
- Spectral Flatness
- Crest Factor
- High-Frequency Energy Ratio
- Peak Prominence
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import gmean
from scipy.signal import find_peaks
import json
import os

# Path to threshold configuration
THRESHOLDS_FILE = os.path.join(os.path.dirname(__file__), 'thresholds.json')

# Default thresholds based on physics of tool vibrations
DEFAULT_THRESHOLDS = {
    "rms_threshold": 2.0,
    "centroid_min": 40.0,
    "flatness_max": 0.5,
    "crest_factor_min": 2.0,
    "peak_prominence_min": 3.0,
    "consecutive_samples": 5,
    "tool_profiles": {}
}


class RobustFeatureExtractor:
    """
    Extracts multiple spectral and time-domain features from acceleration windows.
    
    These features are designed to robustly distinguish tool vibrations from
    human motion (walking) and idle states across different tools and sampling rates.
    """
    
    def __init__(self, fs=833.0):
        """
        Initialize the feature extractor.
        
        Parameters:
        - fs (float): Sampling frequency in Hz.
        """
        self.fs = fs
    
    def compute_magnitude(self, window):
        """Compute the magnitude (vector norm) of triaxial acceleration."""
        return np.sqrt(np.sum(window**2, axis=1))
    
    def compute_rms(self, signal):
        """Compute Root Mean Square of the signal."""
        return np.sqrt(np.mean(signal**2))
    
    def compute_crest_factor(self, signal):
        """Compute Crest Factor = Peak / RMS."""
        rms = self.compute_rms(signal)
        if rms > 0:
            return np.max(np.abs(signal)) / rms
        return 0.0
    
    def compute_fft_spectrum(self, signal):
        """Compute the FFT magnitude spectrum (positive frequencies only)."""
        N = len(signal)
        fft_vals = fft(signal)
        fft_freq = fftfreq(N, d=1/self.fs)
        
        # Take positive frequencies only
        pos_mask = fft_freq > 0
        freqs = fft_freq[pos_mask]
        magnitudes = np.abs(fft_vals[pos_mask]) / N
        
        return freqs, magnitudes
    
    def compute_spectral_centroid(self, freqs, magnitudes):
        """
        Compute Spectral Centroid = sum(f * |X(f)|) / sum(|X(f)|)
        
        Represents the "center of mass" of the spectrum - higher values
        indicate more high-frequency content (tool vibration).
        """
        total_magnitude = np.sum(magnitudes)
        if total_magnitude > 0:
            return np.sum(freqs * magnitudes) / total_magnitude
        return 0.0
    
    def compute_spectral_bandwidth(self, freqs, magnitudes, centroid):
        """
        Compute Spectral Bandwidth = sqrt(sum((f - centroid)^2 * |X(f)|) / sum(|X(f)|))
        
        Measures the spread of the spectrum around the centroid.
        """
        total_magnitude = np.sum(magnitudes)
        if total_magnitude > 0:
            variance = np.sum((freqs - centroid)**2 * magnitudes) / total_magnitude
            return np.sqrt(variance)
        return 0.0
    
    def compute_spectral_flatness(self, magnitudes):
        """
        Compute Spectral Flatness = geometric_mean(|X(f)|) / arithmetic_mean(|X(f)|)
        
        Values close to 1 indicate noise-like (flat) spectrum.
        Values close to 0 indicate tonal (peaky) spectrum - typical for tools.
        """
        # Avoid log(0) by adding small epsilon
        magnitudes_safe = magnitudes + 1e-12
        geometric_mean = gmean(magnitudes_safe)
        arithmetic_mean = np.mean(magnitudes_safe)
        
        if arithmetic_mean > 0:
            return geometric_mean / arithmetic_mean
        return 0.0
    
    # HF energy ratio retained for analysis but not used in classification rules
    def compute_hf_energy_ratio(self, freqs, magnitudes, cutoff=60.0):
        """
        Compute High-Frequency Energy Ratio = energy(f > cutoff) / total_energy.
        Not used for classification; kept for optional diagnostics.
        """
        total_energy = np.sum(magnitudes**2)
        if total_energy > 0:
            hf_mask = freqs > cutoff
            hf_energy = np.sum(magnitudes[hf_mask]**2)
            return hf_energy / total_energy
        return 0.0
    
    def compute_peak_prominence(self, magnitudes):
        """
        Compute Peak Prominence = max_peak_height / mean_spectrum
        
        Higher values indicate clear spectral peaks - typical for rotating tools.
        """
        mean_mag = np.mean(magnitudes)
        if mean_mag > 0:
            # Find peaks
            peaks, properties = find_peaks(magnitudes, height=mean_mag)
            if len(peaks) > 0:
                max_peak = np.max(properties['peak_heights'])
                return max_peak / mean_mag
        return 1.0  # No peaks = flat spectrum
    
    def extract_features(self, window):
        """
        Extract all features from a window of triaxial acceleration data.
        
        Parameters:
        - window (np.array): Shape (N_samples, 3) with X, Y, Z acceleration.
        
        Returns:
        - dict: Dictionary containing all computed features.
        """
        # Compute magnitude (vector norm)
        magnitude = self.compute_magnitude(window)
        
        # Time-domain features
        rms = self.compute_rms(magnitude)
        crest_factor = self.compute_crest_factor(magnitude)
        
        # Frequency-domain features
        freqs, fft_mag = self.compute_fft_spectrum(magnitude)
        
        if len(freqs) > 0:
            centroid = self.compute_spectral_centroid(freqs, fft_mag)
            bandwidth = self.compute_spectral_bandwidth(freqs, fft_mag, centroid)
            flatness = self.compute_spectral_flatness(fft_mag)
            hf_ratio = self.compute_hf_energy_ratio(freqs, fft_mag)
            peak_prominence = self.compute_peak_prominence(fft_mag)
        else:
            centroid = 0.0
            bandwidth = 0.0
            flatness = 1.0
            hf_ratio = 0.0
            peak_prominence = 1.0
        
        return {
            'rms': rms,
            'spectral_centroid': centroid,
            'spectral_bandwidth': bandwidth,
            'spectral_flatness': flatness,
            'crest_factor': crest_factor,
            'hf_energy_ratio': hf_ratio,
            'peak_prominence': peak_prominence
        }


class OnOffClassifier:
    """
    Robust Vibration ON/OFF Detection Classifier
    
    Uses multiple spectral features with threshold-based rules to detect
    tool vibration states. Designed to be robust across different tools
    and sampling frequencies.
    
    Classification Logic:
    - ACTIVE (ON) if: centroid > min AND rms > threshold
    - REST (OFF) otherwise
    - Warnings for anomalous signals (e.g., low-frequency high-energy tools)
    """
    
    def __init__(self, fs=833.0, thresholds=None):
        """
        Initialize the classifier.
        
        Parameters:
        - fs (float): Sampling frequency in Hz.
        - thresholds (dict): Custom thresholds, or None to load from file.
        """
        self.fs = fs
        self.extractor = RobustFeatureExtractor(fs=fs)
        
        if thresholds is None:
            self.load_thresholds()
        else:
            self.thresholds = thresholds
    
    def load_thresholds(self):
        """Load thresholds from JSON file or use defaults."""
        if os.path.exists(THRESHOLDS_FILE):
            try:
                with open(THRESHOLDS_FILE, 'r') as f:
                    self.thresholds = json.load(f)
                    # Ensure all keys exist
                    for key, value in DEFAULT_THRESHOLDS.items():
                        if key not in self.thresholds:
                            self.thresholds[key] = value
            except Exception as e:
                print(f"Warning: Could not load thresholds ({e}). Using defaults.")
                self.thresholds = DEFAULT_THRESHOLDS.copy()
        else:
            self.thresholds = DEFAULT_THRESHOLDS.copy()
    
    def save_thresholds(self):
        """Save current thresholds to JSON file."""
        with open(THRESHOLDS_FILE, 'w') as f:
            json.dump(self.thresholds, f, indent=2)
    
    def predict_window(self, window_data, verbose=False):
        """
        Predict if a window contains tool vibration (ON) or not (OFF).
        
        Parameters:
        - window_data (np.array): Shape (N_samples, 3) triaxial acceleration.
        - verbose (bool): If True, print feature values and reasoning.
        
        Returns:
        - bool: True = VIBRATION_ON, False = VIBRATION_OFF
        - dict: Dictionary of extracted features (if verbose=True)
        """
        # Extract features
        features = self.extractor.extract_features(window_data)
        
        # Apply threshold-based rules (HF ratio removed)
        rms_ok = features['rms'] > self.thresholds['rms_threshold']
        centroid_ok = features['spectral_centroid'] > self.thresholds['centroid_min']
        flatness_ok = features['spectral_flatness'] < self.thresholds['flatness_max']
        
        # Main classification
        is_active = rms_ok and centroid_ok
        
        # Additional confidence checks
        crest_ok = features['crest_factor'] > self.thresholds['crest_factor_min']
        peak_ok = features['peak_prominence'] > self.thresholds['peak_prominence_min']
        
        # Warnings for anomalous signals
        warning = None
        if rms_ok and not centroid_ok:
            # High energy but low frequency - could be low-rpm tool or impact
            warning = "LOW_FREQ_TOOL"
            if verbose:
                print(f"WARNING: High RMS ({features['rms']:.2f}) but low centroid ({features['spectral_centroid']:.1f}Hz). Possible low-frequency tool.")
        
        if verbose:
            print(f"Features: RMS={features['rms']:.2f}, Centroid={features['spectral_centroid']:.1f}Hz, "
                  f"Flatness={features['spectral_flatness']:.3f}")
            print(f"Decision: {'ACTIVE (ON)' if is_active else 'REST (OFF)'}")
            return is_active, features, warning
        
        return is_active
    
    def predict_batch(self, segmented_data, verbose=False):
        """
        Predict labels for a batch of windows.
        
        Parameters:
        - segmented_data (np.array): Shape (N_windows, window_size, 3).
        - verbose (bool): If True, return features for all windows.
        
        Returns:
        - list of bool: True = ON, False = OFF for each window.
        - list of dict (if verbose): Features for each window.
        """
        predictions = []
        all_features = [] if verbose else None
        
        for i, window in enumerate(segmented_data):
            if verbose:
                # Call with verbose=True to get features
                is_active, features, _ = self.predict_window(window, verbose=True)
                predictions.append(is_active)
                all_features.append(features)
            else:
                predictions.append(self.predict_window(window, verbose=False))
        
        if verbose:
            return predictions, all_features
        return predictions
    
    def predict_from_features(self, features_dict, verbose=False):
        """
        Predict if a window is ACTIVE based on pre-extracted features.
        
        Use this method when features are already computed (e.g., from CSV dataset).
        No feature extraction needed - applies threshold rules directly.
        
        Parameters:
        - features_dict (dict): Dictionary containing pre-extracted features:
            - 'rms': Root Mean Square
            - 'spectral_centroid': Spectral Centroid (Hz)
            - 'hf_energy_ratio': High-Frequency Energy Ratio
            - Optional: 'spectral_flatness', 'crest_factor', 'peak_prominence'
        - verbose (bool): If True, print decision reasoning.
        
        Returns:
        - bool: True = ACTIVE (ON), False = REST (OFF)
        - str or None: Warning message for anomalous signals (if verbose)
        """
        # Extract required features with defaults for missing optional ones
        rms = features_dict.get('rms', 0.0)
        centroid = features_dict.get('spectral_centroid', 0.0)
        flatness = features_dict.get('spectral_flatness', 0.5)
        crest_factor = features_dict.get('crest_factor', 1.0)
        peak_prominence = features_dict.get('peak_prominence', 1.0)
        
        # Apply threshold-based rules (HF ratio removed)
        rms_ok = rms > self.thresholds['rms_threshold']
        centroid_ok = centroid > self.thresholds['centroid_min']
        
        # Main classification
        is_active = rms_ok and centroid_ok
        
        # Warnings for anomalous signals
        warning = None
        if rms_ok and not centroid_ok:
            warning = "LOW_FREQ_TOOL"
        elif not rms_ok and centroid_ok:
            warning = "IDLE_TOOL"
        
        if verbose:
            print(f"Features: RMS={rms:.2f}, Centroid={centroid:.1f}Hz, "
                  f"Flatness={flatness:.3f}")
            print(f"Thresholds: RMS>{self.thresholds['rms_threshold']:.1f}, "
                  f"Centroid>{self.thresholds['centroid_min']:.0f}Hz")
            print(f"Decision: {'ACTIVE (ON)' if is_active else 'REST (OFF)'}")
            if warning:
                print(f"Warning: {warning}")
            return is_active, warning
        
        return is_active
    
    def predict_batch_from_features(self, features_list, verbose=False):
        """
        Predict labels for a batch of pre-extracted feature dictionaries.
        
        Parameters:
        - features_list (list of dict): List of feature dictionaries.
        - verbose (bool): If True, print progress.
        
        Returns:
        - list of bool: True = ACTIVE, False = REST for each window.
        """
        predictions = []
        warnings = []
        
        for i, features in enumerate(features_list):
            if verbose:
                is_active, warning = self.predict_from_features(features, verbose=False)
                warnings.append(warning)
            else:
                is_active = self.predict_from_features(features, verbose=False)
            predictions.append(is_active)
        
        if verbose:
            return predictions, warnings
        return predictions
    
    def get_confidence(self, features):
        """
        Calculate confidence score for the prediction (0.0 to 1.0).
        
        Higher scores indicate clearer tool vibration signatures.
        """
        scores = []
        
        # RMS score (normalized)
        rms_score = min(features['rms'] / (self.thresholds['rms_threshold'] * 2), 1.0)
        scores.append(rms_score)
        
        # Centroid score (normalized)
        centroid_score = min(features['spectral_centroid'] / (self.thresholds['centroid_min'] * 2), 1.0)
        scores.append(centroid_score)
        
        # Flatness score (inverted - lower is better)
        flatness_score = 1.0 - min(features['spectral_flatness'], 1.0)
        scores.append(flatness_score)
        
        return np.mean(scores)


def calibrate_threshold(training_data_on, training_data_off, fs=833.0):
    """
    Calibrate optimal thresholds from training data using multi-feature analysis.
    
    Parameters:
    - training_data_on (list of np.array): Windows with tool active.
    - training_data_off (list of np.array): Windows with tool inactive.
    - fs (float): Sampling frequency.
    
    Returns:
    - dict: Optimized thresholds.
    """
    extractor = RobustFeatureExtractor(fs=fs)
    
    # Extract features from all windows
    on_features = [extractor.extract_features(w) for w in training_data_on]
    off_features = [extractor.extract_features(w) for w in training_data_off]
    
    # Calculate statistics
    def get_stats(features_list, key):
        values = [f[key] for f in features_list]
        return np.mean(values), np.std(values), np.min(values), np.max(values)
    
    print("\n=== Feature Statistics ===")
    print("\nON (Tool Active) Data:")
    for key in ['rms', 'spectral_centroid', 'spectral_flatness']:
        mean, std, min_v, max_v = get_stats(on_features, key)
        print(f"  {key}: mean={mean:.3f}, std={std:.3f}, range=[{min_v:.3f}, {max_v:.3f}]")
    
    print("\nOFF (Tool Inactive) Data:")
    for key in ['rms', 'spectral_centroid', 'spectral_flatness']:
        mean, std, min_v, max_v = get_stats(off_features, key)
        print(f"  {key}: mean={mean:.3f}, std={std:.3f}, range=[{min_v:.3f}, {max_v:.3f}]")
    
    # Calculate optimal thresholds (midpoint between ON/OFF means with safety margin)
    on_rms_mean = np.mean([f['rms'] for f in on_features])
    off_rms_mean = np.mean([f['rms'] for f in off_features])
    optimal_rms = (on_rms_mean + off_rms_mean) / 2.0
    
    on_centroid_mean = np.mean([f['spectral_centroid'] for f in on_features])
    off_centroid_mean = np.mean([f['spectral_centroid'] for f in off_features])
    # Use lower bound of ON centroid with margin
    optimal_centroid = max(off_centroid_mean * 1.5, 40.0)
    
    on_flatness_mean = np.mean([f['spectral_flatness'] for f in on_features])
    off_flatness_mean = np.mean([f['spectral_flatness'] for f in off_features])
    optimal_flatness = (on_flatness_mean + off_flatness_mean) / 2.0
    
    optimal_thresholds = {
        'rms_threshold': optimal_rms,
        'centroid_min': optimal_centroid,
        'flatness_max': optimal_flatness,
        'crest_factor_min': 2.0,
        'peak_prominence_min': 3.0,
        'consecutive_samples': 5,
        'tool_profiles': {}
    }
    
    print("\n=== Optimal Thresholds ===")
    print(f"  RMS threshold: {optimal_rms:.3f}")
    print(f"  Centroid min: {optimal_centroid:.1f} Hz")
    print(f"  Flatness max: {optimal_flatness:.3f}")
    
    return optimal_thresholds


def calibrate_tool_profile(tool_data, tool_name, fs=833.0):
    """
    Calibrate a profile for a specific tool type.
    
    Parameters:
    - tool_data (list of np.array): Windows recorded with the specific tool.
    - tool_name (str): Name of the tool (e.g., 'drill', 'sander', 'grinder').
    - fs (float): Sampling frequency.
    
    Returns:
    - dict: Tool-specific profile with feature statistics.
    """
    extractor = RobustFeatureExtractor(fs=fs)
    features_list = [extractor.extract_features(w) for w in tool_data]
    
    profile = {
        'name': tool_name,
        'n_samples': len(tool_data),
        'fs': fs,
        'features': {}
    }
    
    for key in ['rms', 'spectral_centroid', 'spectral_bandwidth', 'hf_energy_ratio', 
                'spectral_flatness', 'crest_factor', 'peak_prominence']:
        values = [f[key] for f in features_list]
        profile['features'][key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    print(f"\n=== Tool Profile: {tool_name} ===")
    print(f"Samples: {len(tool_data)}")
    print(f"Sampling Frequency: {fs} Hz")
    for key, stats in profile['features'].items():
        print(f"  {key}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    return profile


if __name__ == '__main__':
    # Test the classifier
    print("Testing Robust ON/OFF Classifier...")
    
    classifier = OnOffClassifier(fs=833.0)
    
    # Create test windows
    # Simulated ACTIVE window (high-frequency vibration)
    t = np.linspace(0, 0.1, 83)  # ~100ms at 833Hz
    active_window = np.column_stack([
        10 * np.sin(2 * np.pi * 100 * t),  # 100Hz vibration X
        8 * np.sin(2 * np.pi * 100 * t + 0.5),  # Y
        12 * np.sin(2 * np.pi * 100 * t + 1.0)  # Z
    ])
    
    # Simulated REST window (low-frequency drift)
    rest_window = np.column_stack([
        0.5 * np.sin(2 * np.pi * 0.5 * t),  # Slow drift X
        0.3 * np.sin(2 * np.pi * 0.3 * t),  # Y
        9.81 + 0.2 * np.sin(2 * np.pi * 0.2 * t)  # Z with gravity
    ])
    
    print("\n--- Testing ACTIVE window ---")
    is_active, features, warning = classifier.predict_window(active_window, verbose=True)
    
    print("\n--- Testing REST window ---")
    is_rest, features, warning = classifier.predict_window(rest_window, verbose=True)
    
    # Save default thresholds
    classifier.save_thresholds()
    print(f"\n✅ Thresholds saved to {THRESHOLDS_FILE}")
