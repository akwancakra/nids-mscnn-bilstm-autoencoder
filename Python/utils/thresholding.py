import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

def calculate_reconstruction_error(model, X, batch_size=256):
    """
    Calculate MSE for each sample in X based on model reconstruction.
    Uses batch processing with TQDM progress bar to avoid OOM and provide feedback.
    
    Args:
        model: Trained Keras model (Autoencoder)
        X: Input data (3D array: samples, timesteps, features)
        batch_size: Size of batches for prediction
        
    Returns:
        np.array: MSE for each sample (1D array)
    """
    print("Predicting reconstruction...")
    n_samples = len(X)
    mse_list = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, n_samples, batch_size), desc="Calculating Error", unit="batch"):
        batch_X = X[i : i + batch_size]
        batch_pred = model.predict(batch_X, verbose=0) # verbose=0 to silence Keras
        
        # Calculate MSE for this batch
        # Shape: (batch_size, timesteps, features)
        # MSE = mean((X - X_pred)^2, axis=(1, 2))
        batch_mse = np.mean(np.square(batch_X - batch_pred), axis=(1, 2))
        mse_list.append(batch_mse)
        
    mse = np.concatenate(mse_list, axis=0)
    return mse

def get_threshold_percentile(errors, percentile=95):
    """
    Determine threshold based on a given percentile of errors.
    """
    threshold = np.percentile(errors, percentile)
    print(f"Threshold determined at {percentile}th percentile: {threshold:.6f}")
    return threshold

def plot_error_distribution(errors, threshold=None, save_path=None):
    """
    Plot histogram of reconstruction errors.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='blue', label='Reconstruction Error')
    if threshold:
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error Distribution (Benign)')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()
