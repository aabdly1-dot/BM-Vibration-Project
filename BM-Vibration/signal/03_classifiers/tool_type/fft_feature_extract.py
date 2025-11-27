import numpy as np
from scipy.fft import fft, fftfreq
import pickle
import os

class FFTFeatureExtractor:
    def __init__(self, fs=833):
        self.fs = fs
    
    def extract_fft_features(self, window_data):
        features = {}
        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            axis_data = window_data[:, axis_idx]
            N = len(axis_data)
            fft_vals = fft(axis_data)
            fft_freq = fftfreq(N, d=1/self.fs)
            pos_mask = fft_freq > 0
            fft_freq_pos = fft_freq[pos_mask]
            fft_mag = np.abs(fft_vals[pos_mask]) / N
            features[f'{axis_name}_dominant_freq'] = fft_freq_pos[np.argmax(fft_mag)]
            features[f'{axis_name}_peak_magnitude'] = np.max(fft_mag)
            features[f'{axis_name}_mean_magnitude'] = np.mean(fft_mag)
        return features

class ToolTypeClassifier:
    def __init__(self, model_path=None):
        self.feature_extractor = FFTFeatureExtractor()
        self.model = None
        self.tool_labels = ['drill', 'grinder', 'saw']
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def predict(self, window_data):
        features = self.feature_extractor.extract_fft_features(window_data)
        if self.model is None:
            return 'unknown'
        return 'drill'  # Placeholder until trained
    
    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump({'model': self.model}, f)
    
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']

if __name__ == '__main__':
    print("FFT Feature Extractor ready")

