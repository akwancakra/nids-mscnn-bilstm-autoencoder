import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class GlobalScaler:
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.is_fitted = False

    def fit(self, data):
        """Fit scaler to data (should be training data only)."""
        self.scaler.fit(data)
        self.is_fitted = True

    def transform(self, data):
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform!")
        return self.scaler.transform(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save(self, filepath):
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            self.scaler = joblib.load(filepath)
            self.is_fitted = True
            print(f"Scaler loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Scaler file not found: {filepath}")

def create_sequences(data, time_steps=20, stride=10):
    """
    Create sliding window sequences from 2D data.
    
    Args:
        data: 2D array (samples, features)
        time_steps: Length of each window sequence
        stride: Step size for sliding window
        
    Returns:
        np.array: 3D array (num_sequences, time_steps, features)
    """
    xs = []
    # If data is DataFrame, convert to numpy
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    # Generate sequences
    for i in range(0, len(data) - time_steps + 1, stride):
        xs.append(data[i:(i + time_steps)])
        
    return np.array(xs)

def load_and_preprocess_file(filepath, scaler=None, fit_scaler=False):
    """
    Load a CSV file, handle basic cleaning, and apply scaling.
    
    Args:
        filepath: Path to CSV file
        scaler: GlobalScaler instance
        fit_scaler: Whether to fit the scaler on this data (True for Train, False for Test)
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    print(f"Loading {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
        
    # Basic cleaning
    # Replace Inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fill NaN with mean or 0 (using 0 is safer for sparse network data)
    df = df.fillna(0)
    
    # Drop non-numeric columns if any (like Label if it's string, though usually encoded)
    # Assuming 'Label' is the target and we want features only for AE input
    # If Label exists, we separate it
    label = None
    if 'Label' in df.columns:
        label = df['Label']
        df = df.drop(columns=['Label'])
    elif 'label' in df.columns: # Case insensitive check
        label = df['label']
        df = df.drop(columns=['label'])
        
    # Drop Timestamp/IPs if present (usually already removed in processed datasets but good to check)
    cols_to_drop = ['Timestamp', 'Flow ID', 'Source IP', 'Destination IP', 'SimillarHTTP'] 
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        
    # Scaling
    if scaler:
        if fit_scaler:
            print("Fitting scaler on training data...")
            data_scaled = scaler.fit_transform(df.values)
        else:
            print("Transforming data with existing scaler...")
            data_scaled = scaler.transform(df.values)
        
        df_scaled = pd.DataFrame(data_scaled, columns=df.columns)
        return df_scaled, label
    
    return df, label
