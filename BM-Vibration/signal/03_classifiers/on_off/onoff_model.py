
import numpy as np
import json
import os

# Path to threshold configuration
THRESHOLDS_FILE = os.path.join(os.path.dirname(__file__), 'thresholds.json')

class OnOffClassifier:
    """
    Classifier v1: Vibration ON/OFF Detection
    
    This classifier determines if tool vibration is present based on
    the magnitude of filtered acceleration data.
    """
    
    def __init__(self, magnitude_threshold=None, consecutive_samples=5):
        self.consecutive_samples = consecutive_samples
        if magnitude_threshold is None:
            self.load_thresholds()
        else:
            self.magnitude_threshold = magnitude_threshold
    
    def load_thresholds(self):
        if os.path.exists(THRESHOLDS_FILE):
            try:
                with open(THRESHOLDS_FILE, 'r') as f:
                    config = json.load(f)
                    self.magnitude_threshold = config.get('magnitude_threshold', 2.0)
                    self.consecutive_samples = config.get('consecutive_samples', 5)
            except:
                self.magnitude_threshold = 2.0
        else:
            self.magnitude_threshold = 2.0
    
    def save_thresholds(self):
        config = {
            'magnitude_threshold': self.magnitude_threshold,
            'consecutive_samples': self.consecutive_samples
        }
        with open(THRESHOLDS_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    def predict_window(self, window_data):
        magnitudes = np.sqrt(np.sum(window_data**2, axis=1))
        above_threshold = magnitudes > self.magnitude_threshold
        max_consecutive = 0
        current_consecutive = 0
        for is_above in above_threshold:
            if is_above:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return max_consecutive >= self.consecutive_samples
    
    def predict_batch(self, segmented_data):
        return [self.predict_window(window) for window in segmented_data]


def calibrate_threshold(training_data_on, training_data_off):
    """Calibrate optimal threshold from training data"""
    on_magnitudes = []
    off_magnitudes = []
    
    for window in training_data_on:
        mag = np.sqrt(np.sum(window**2, axis=1))
        on_magnitudes.extend(mag)
    
    for window in training_data_off:
        mag = np.sqrt(np.sum(window**2, axis=1))
        off_magnitudes.extend(mag)
    
    mean_on = np.mean(on_magnitudes)
    mean_off = np.mean(off_magnitudes)
    optimal_threshold = (mean_on + mean_off) / 2.0
    
    print(f"Mean ON magnitude: {mean_on:.3f}")
    print(f"Mean OFF magnitude: {mean_off:.3f}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    
    return optimal_threshold


if __name__ == '__main__':
    classifier = OnOffClassifier()
    classifier.save_thresholds()
    print("ON/OFF Classifier ready")
