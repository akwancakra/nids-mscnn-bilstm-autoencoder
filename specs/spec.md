# Spesifikasi Implementasi: SOTA Multi-Scale CNN-BiLSTM Autoencoder NIDS

Dokumen ini merinci rencana perbaikan total pada `IntrusionDetectionSystem` untuk mencapai performa State-of-the-Art (SOTA) dalam deteksi intrusi jaringan secara *unsupervised* dan *cross-dataset*.

## 1. Latar Belakang Masalah
Implementasi sebelumnya gagal mencapai performa yang memuaskan (F1-Score < 0.4 pada target domain) karena:
1.  **Arsitektur Salah:** Menggunakan Dense Autoencoder atau CNN-LSTM *single-stream* dengan `sequence_length=1`.
2.  **Scaling Inconsistent:** Melakukan scaling per file, merusak distribusi data antar dataset.
3.  **Thresholding Naif:** Menggunakan statistik Gaussian sederhana (Mean+Std) yang tidak cocok untuk distribusi error anomali.

## 2. Solusi Arsitektur SOTA
Kami akan mengimplementasikan **Multi-Scale Convolutional Bi-Directional LSTM Autoencoder (MS-CNN-BiLSTM-AE)** dengan mekanisme **Attention**.

### 2.1 Model Architecture (`Python/models/sota_model.py`)
*   **Input Shape:** `(Batch, Sequence_Length=20, Features)`
*   **Encoder (Multi-Scale):**
    *   **Cabang 1 (Short-term):** Conv1D (Filters=32, Kernel=3, Padding='same') -> ReLU
    *   **Cabang 2 (Medium-term):** Conv1D (Filters=32, Kernel=5, Padding='same') -> ReLU
    *   **Cabang 3 (Long-term):** Conv1D (Filters=32, Kernel=7, Padding='same') -> ReLU
    *   **Fusion:** Concatenate output cabang 1, 2, 3.
    *   **Temporal:** Bi-Directional LSTM (Units=64, Return Sequences=True) -> Dropout(0.2).
    *   **Attention:** Self-Attention Mechanism untuk membobot timestep penting.
*   **Bottleneck (Latent):** Dense Layer (Units=32 atau 16) -> Code Representation.
*   **Decoder (Reconstruction):**
    *   RepeatVector (Sequence_Length).
    *   Bi-Directional LSTM (Units=64, Return Sequences=True).
    *   Upsampling / Transposed Conv (untuk mengembalikan dimensi fitur).
    *   **Output:** TimeDistributed(Dense(Features)) -> Linear Activation.

### 2.2 Data Pipeline (`Python/preprocessing/pipeline.py`)
*   **Sequence Generation:** Menggunakan teknik *Sliding Window* dengan `window_size=20` dan `stride=10` (50% overlap).
*   **Global Scaling:**
    *   **Fit:** HANYA pada Dataset Training (CIC-IDS2017 Benign).
    *   **Save:** Simpan parameter scaler (Min, Max, Mean, Std) ke file `.pkl`.
    *   **Transform:** Load scaler dan transform Dataset Testing (CSE-CIC-IDS2018) tanpa fit ulang.
*   **Handling NaNs/Inf:** Imputasi dengan 0 atau Mean sebelum scaling.

### 2.3 Training & Evaluation Strategy (`Python/train_eval.py`)
*   **Loss Function:** Mean Squared Error (MSE) antara Input dan Output Rekonstruksi.
*   **Optimizer:** Adam (Learning Rate=0.001) dengan Early Stopping.
*   **Thresholding Dinamis:**
    *   Hitung distribusi Reconstruction Error pada data validasi (Benign).
    *   Tentukan threshold pada **Persentil ke-95, 99, atau 99.9** (dapat dikonfigurasi).
    *   (Opsional) Implementasi **Peaks Over Threshold (POT)** berbasis Extreme Value Theory (EVT) untuk deteksi robust.

## 3. Struktur Direktori Baru
```
IntrusionDetectionSystem/
├── Python/
│   ├── models/
│   │   ├── __init__.py
│   │   └── sota_model.py       # Definisi Arsitektur Baru
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Windowing & Global Scaling
│   ├── utils/
│   │   ├── __init__.py
│   │   └── thresholding.py     # Percentile & EVT Logic
│   ├── config.yaml             # Konfigurasi Hyperparameter
│   └── main_sota.py            # Script Utama (Train/Eval)
```

## 4. Metrik Keberhasilan
*   **F1-Score Target (CSE-CIC-IDS2018):** > 0.75 (Signifikan lebih baik dari 0.35).
*   **False Positive Rate:** < 5%.
*   **Pipeline:** Berjalan end-to-end dari raw CSV CIC2017 -> Model SOTA -> Evaluasi CSE2018.
