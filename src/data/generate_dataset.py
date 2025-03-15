import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.data_generator import EVChargingDataGenerator, DataPreprocessor

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize data generator
    generator = EVChargingDataGenerator(
        num_sessions=1000,  # Generate 1000 sessions
        time_interval_minutes=15,  # 15-minute intervals
        base_price=0.5  # Base electricity price
    )
    
    # Generate synthetic data
    print("Generating synthetic charging session data...")
    raw_data = generator.generate_charging_sessions()
    
    # Save raw data
    raw_data_path = data_dir / "raw_charging_sessions.csv"
    raw_data.to_csv(raw_data_path, index=False)
    print(f"Raw data saved to {raw_data_path}")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_data, scalers = preprocessor.preprocess(raw_data)
    
    # Save processed data
    processed_data_path = data_dir / "processed_charging_sessions.csv"
    processed_data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
    
    # Save preprocessing scalers
    scalers_path = data_dir / "preprocessing_scalers.json"
    with open(scalers_path, 'w') as f:
        json.dump(scalers, f, indent=2, cls=NumpyEncoder)
    print(f"Preprocessing scalers saved to {scalers_path}")
    
    # Print data statistics
    print("\nDataset Statistics:")
    print(f"Number of sessions: {len(processed_data)}")
    print("\nFeatures:")
    for column in processed_data.columns:
        print(f"- {column}")
    
    print("\nSample of processed data:")
    print(processed_data.head())

if __name__ == "__main__":
    main() 