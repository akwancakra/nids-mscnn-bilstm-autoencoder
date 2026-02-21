import os
import glob
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import sys

# Add parent directory to path to import utils if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_dataframe(df):
    """
    Clean the dataframe: drop non-numeric, handle Inf/NaN.
    """
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Columns to drop (identifiers, timestamps)
    # Note: 'Label' is preserved for filtering, dropped later
    drop_cols = [
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port',
        'Protocol', 'Timestamp', 'SimillarHTTP', 'Inbound', 'Unnamed: 0'
    ]
    
    # Drop existing columns
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Standardize Label column if it exists
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip().str.upper()
        # Remove header rows that might have leaked into the data
        df = df[df['Label'] != 'LABEL']
    
    # Replace Inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

def get_benign_data(df):
    """
    Filter for Benign traffic only.
    Assumes 'Label' column exists.
    """
    if 'Label' in df.columns:
        # Label is already uppercase from clean_dataframe
        return df[df['Label'] == 'BENIGN'].drop(columns=['Label'], errors='ignore')
    return df

def fit_scaler_incrementally(files, scaler):
    """
    Compute global Min/Max by iterating through files.
    """
    print("Computing global statistics for scaling...")
    for f in tqdm(files, desc="Fitting Scaler"):
        try:
            df = pd.read_csv(f)
            df = clean_dataframe(df)
            df = get_benign_data(df) # Only fit on Benign data
            
            # Ensure only numeric columns remain
            df = df.select_dtypes(include=[np.number])
            
            if not df.empty:
                scaler.partial_fit(df.values)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return scaler

def create_sequences(data, seq_len, stride):
    """
    Create sliding window sequences.
    """
    sequences = []
    for i in range(0, len(data) - seq_len + 1, stride):
        sequences.append(data[i : i + seq_len])
    return np.array(sequences)

def create_label_sequences(labels, seq_len, stride):
    """
    Create label sequences (taking the max/attack label in the window).
    """
    sequences = []
    for i in range(0, len(labels) - seq_len + 1, stride):
        window = labels[i : i + seq_len]
        # If any attack in window -> Attack (1)
        # This is standard for NIDS to catch attack packets
        if np.any(window == 1):
            sequences.append(1)
        else:
            sequences.append(0)
    return np.array(sequences)

def process_files(files, output_dir, scaler, seq_len=10, stride=5, mode='train'):
    """
    Process files, apply scaling, sequence, and save shards.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    shard_id = 0
    for f in tqdm(files, desc=f"Processing {mode} data"):
        try:
            df = pd.read_csv(f)
            
            # 1. Clean
            df = clean_dataframe(df)
            
            # 2. Handle Label
            labels = None
            if 'Label' in df.columns:
                if mode == 'train':
                    # For training, we only want Benign
                    df = df[df['Label'] == 'BENIGN']
                    # Drop Label
                    df = df.drop(columns=['Label'])
                else:
                    # For testing
                    # 0 = Benign, 1 = Attack
                    # Handle different label names if needed
                    labels = (df['Label'] != 'BENIGN').astype(int).values
                    df = df.drop(columns=['Label'])
            
            # Ensure only numeric columns
            df = df.select_dtypes(include=[np.number])
            
            if df.empty:
                continue
                
            # 3. Scale
            # Note: scaler expects exact same number of columns as fit
            try:
                data_scaled = scaler.transform(df.values)
                # CLIPPING: Force range [0, 1] to handle outliers in Test data
                # This prevents massive values like 119400326.0 from destroying the model
                data_scaled = np.clip(data_scaled, 0.0, 1.0)
            except ValueError as e:
                print(f"Skipping {f} due to shape mismatch: {e}")
                continue
            
            # 4. Sequence
            X_seq = create_sequences(data_scaled, seq_len, stride)
            
            y_seq = None
            if labels is not None:
                y_seq = create_label_sequences(labels, seq_len, stride)
            else:
                y_seq = np.zeros(len(X_seq)) # Default to 0 if no label found (assume benign)
            
            # 5. Save Shard
            if len(X_seq) > 0:
                shard_name = f"{mode}_shard_{shard_id}.npz"
                save_path = os.path.join(output_dir, shard_name)
                # Ensure X and y have same length
                min_len = min(len(X_seq), len(y_seq))
                X_seq = X_seq[:min_len]
                y_seq = y_seq[:min_len]
                
                np.savez_compressed(save_path, X=X_seq, y=y_seq)
                shard_id += 1
                
        except Exception as e:
            print(f"Error processing {f}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Preprocessed Data for MSCNN-BiLSTM")
    parser.add_argument("--raw_train", type=str, required=True, help="Path to raw training CSVs (CIC-IDS2017)")
    parser.add_argument("--raw_test", type=str, required=True, help="Path to raw test CSVs (CSE-CIC-IDS2018)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride (1 for max data)")
    
    args = parser.parse_args()
    
    # 1. Setup paths
    # Handle wildcard if provided in quotes, or directory
    if '*' in args.raw_train:
        train_files = glob.glob(args.raw_train)
    else:
        train_files = glob.glob(os.path.join(args.raw_train, "*.csv"))
        
    if '*' in args.raw_test:
        test_files = glob.glob(args.raw_test)
    else:
        test_files = glob.glob(os.path.join(args.raw_test, "*.csv"))
    
    print(f"Found {len(train_files)} training files.")
    print(f"Found {len(test_files)} test files.")
    
    if not train_files:
        print("No training files found! Check path.")
        return

    # 2. Fit Scaler (MinMax 0-1)
    # Use GlobalScaler logic: Fit on Benign Training Data ONLY
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = fit_scaler_incrementally(train_files, scaler)
    
    # Save Scaler
    os.makedirs(args.output_dir, exist_ok=True)
    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # 3. Process Train Data (Benign Only)
    train_out = os.path.join(args.output_dir, "train")
    process_files(train_files, train_out, scaler, args.seq_len, args.stride, mode='train')
    
    # 4. Process Test Data (Mixed)
    test_out = os.path.join(args.output_dir, "test")
    process_files(test_files, test_out, scaler, args.seq_len, args.stride, mode='test')
    
    # 5. Process CIC-IDS2017 Test Data (Mixed)
    test_cic_out = os.path.join(args.output_dir, "test_cic")
    process_files(train_files, test_cic_out, scaler, args.seq_len, args.stride, mode='test')
    
    print("Preprocessing Complete!")

if __name__ == "__main__":
    main()
