import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_preprocessing', 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_classifiers', 'on_off'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '03_classifiers', 'tool_type'))

from highpass_filter import filter_triaxial_data
from segmentation import create_overlapping_windows
from preprocess_pipeline import load_raw_data_from_json
from onoff_model import OnOffClassifier
from fft_feature_extract import ToolTypeClassifier

class ValidationPipeline:
    def __init__(self):
        self.onoff_classifier = OnOffClassifier()
        self.tool_classifier = ToolTypeClassifier()
        self.results = []
    
    def validate_file(self, file_path, ground_truth_label=None, tool_type=None):
        print(f"\nValidating: {os.path.basename(file_path)}")
        df_raw = load_raw_data_from_json(file_path)
        if df_raw is None or df_raw.empty:
            return None
        df_filtered = filter_triaxial_data(df_raw)
        segmented_data = create_overlapping_windows(df_filtered)
        onoff_predictions = self.onoff_classifier.predict_batch(segmented_data)
        n_on = sum(onoff_predictions)
        n_off = len(onoff_predictions) - n_on
        on_percentage = (n_on / len(onoff_predictions)) * 100
        print(f"  Windows ON: {n_on} ({on_percentage:.1f}%)")
        print(f"  Windows OFF: {n_off}")
        result = {
            'file': os.path.basename(file_path),
            'n_windows': len(segmented_data),
            'n_on': n_on,
            'on_percentage': on_percentage
        }
        self.results.append(result)
        return result

if __name__ == '__main__':
    print("VALIDATION SCRIPT")
    validator = ValidationPipeline()
    data_path = os.path.join(os.path.dirname(__file__), '..', '01_data_collection', 'raw')
    drill_path = os.path.join(data_path, 'tool_drill')
    if os.path.exists(drill_path):
        for f in os.listdir(drill_path):
            if f.endswith('.json'):
                validator.validate_file(os.path.join(drill_path, f), 'ON', 'drill')
    print(f"\nValidated {len(validator.results)} files")

