import numpy as np
import glob
import os
from tqdm.auto import tqdm

def load_npz_data(data_dir, limit_files=None):
    """
    Load data from NPZ shards with progress bar.
    Assumes keys 'x' and optionally 'y'.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        print(f"Warning: No .npz files found in {data_dir}")
        return None, None

    if limit_files:
        files = files[:limit_files]
        
    print(f"Loading {len(files)} NPZ shards from {data_dir}...")
    
    xs = []
    ys = []
    
    # Use tqdm for progress bar
    for f in tqdm(files, desc="Loading Shards", unit="file"):
        try:
            with np.load(f, allow_pickle=True) as data:
                if 'x' in data:
                    xs.append(data['x'])
                if 'y' in data:
                    ys.append(data['y'])
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not xs:
        return None, None
        
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0) if ys else None
    
    return X, y
