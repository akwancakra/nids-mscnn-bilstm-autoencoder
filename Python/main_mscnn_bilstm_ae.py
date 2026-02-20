import yaml
import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow_addons as tfa

from models.mscnn_bilstm_ae import build_multiscale_cnn_bilstm_ae
from preprocessing.data_loader_npz import load_npz_data
from utils.thresholding import calculate_reconstruction_error, get_threshold_percentile, plot_error_distribution

def load_config(config_path="config.yaml"):
    # Load config from root directory
    if not os.path.exists(config_path):
        config_path = os.path.join("..", config_path)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    
    # --- 1. Load Training Data (CIC-IDS2017 Benign Shards) ---
    print("\n[Phase 1] Loading Training Data (NPZ Shards)...")
    X_train, _ = load_npz_data(
        config['paths']['train_data'], 
        limit_files=10 # Use more shards for actual training
    )
    
    if X_train is None:
        print("Error: No training data found! Check config path.")
        return
        
    print(f"Training Data Shape: {X_train.shape}")
    
    # Validation Split
    X_train_split, X_val_split = train_test_split(X_train, test_size=config['training']['validation_split'], random_state=42)
    
    # --- 2. Build & Train Model ---
    print("\n[Phase 2] Building & Training Model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input Shape: {input_shape}")
    
    model = build_multiscale_cnn_bilstm_ae(input_shape, encoding_dim=config['model']['encoding_dim'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=config['training']['patience'], 
        restore_best_weights=True
    )
    
    # Try adding TQDM callback
    callbacks = [early_stopping]
    try:
        callbacks.append(tfa.callbacks.TQDMProgressBar())
        verbose_mode = 0
    except:
        verbose_mode = 1

    history = model.fit(
        X_train_split, X_train_split, # Autoencoder target is input itself
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        validation_data=(X_val_split, X_val_split),
        callbacks=callbacks,
        verbose=verbose_mode
    )
    
    model_path = config['paths']['model_save_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # --- 3. Determine Threshold ---
    print("\n[Phase 3] Determining Threshold...")
    val_errors = calculate_reconstruction_error(model, X_val_split)
    threshold = get_threshold_percentile(val_errors, percentile=config['thresholding']['percentile'])
    
    # Plot
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    plot_error_distribution(val_errors, threshold, save_path=os.path.join(output_dir, 'error_dist_val.png'))
    
    # --- 4. Load Test Data (CSE-CIC-IDS2018 Mixed Shards) ---
    print("\n[Phase 4] Loading Test Data (NPZ Shards)...")
    X_test, y_test = load_npz_data(
        config['paths']['test_data'], 
        limit_files=5 # TESTING LIMIT
    )
    
    if X_test is None:
        print("Error: No test data found!")
        return
        
    print(f"Test Data Shape: {X_test.shape}")
    if y_test is not None:
        print(f"Test Labels Shape: {y_test.shape}")
        
    # --- 5. Evaluate ---
    print("\n[Phase 5] Evaluation...")
    test_errors = calculate_reconstruction_error(model, X_test)
    
    # Predict: Error > Threshold -> Anomaly (1)
    y_pred = (test_errors > threshold).astype(int)
    
    # Ensure y_test is binary (0/1)
    if y_test is not None:
        unique_labels = np.unique(y_test)
        print(f"Unique Test Labels: {unique_labels}")
        
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, test_errors)
        except:
            auc = 0.0
            
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n" + "="*40)
        print("MSCNN-BiLSTM-AE EVALUATION REPORT")
        print("="*40)
        print(f"Threshold (Percentile {config['thresholding']['percentile']}): {threshold:.6f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print("-" * 20)
        print("Confusion Matrix:")
        print(cm)
        print("="*40)
        
        results = {
            'threshold': float(threshold),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist()
        }
        with open(os.path.join(output_dir, 'evaluation_results.yaml'), 'w') as f:
            yaml.dump(results, f)

if __name__ == "__main__":
    main()
