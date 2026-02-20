# Daftar Tugas Implementasi SOTA NIDS

Tugas-tugas berikut dirancang untuk menggantikan implementasi lama yang cacat dengan arsitektur Multi-Scale CNN-BiLSTM Autoencoder yang robust.

- [ ] **1. Persiapan Struktur Proyek** <!-- id: 0 -->
    - [ ] Buat folder `Python/models`, `Python/preprocessing`, `Python/utils`. <!-- id: 1 -->
    - [ ] Pindahkan file lama (`IDS.py`, `Preprocess.py`, `Main.py`) ke `Python/legacy/`. <!-- id: 2 -->
    - [ ] Buat `Python/__init__.py` kosong di setiap folder baru. <!-- id: 3 -->

- [ ] **2. Implementasi Model SOTA (`Python/models/sota_model.py`)** <!-- id: 4 -->
    - [ ] Definisikan kelas `MultiScaleAutoencoder` (subclass `tf.keras.Model`). <!-- id: 5 -->
    - [ ] Implementasikan layer Encoder Multi-Scale (Conv1D k=3, k=5, k=7). <!-- id: 6 -->
    - [ ] Implementasikan layer Bi-LSTM dan Attention Mechanism. <!-- id: 7 -->
    - [ ] Implementasikan layer Decoder (Mirror Encoder). <!-- id: 8 -->
    - [ ] Tambahkan metode `compile_model` dengan loss MSE dan optimizer Adam. <!-- id: 9 -->

- [ ] **3. Implementasi Pipeline Data (`Python/preprocessing/data_loader.py`)** <!-- id: 10 -->
    - [ ] Buat kelas `GlobalScaler` untuk menangani scaling (fit on train, transform on test). <!-- id: 11 -->
    - [ ] Implementasikan fungsi `create_sequences` untuk sliding window (default `window_size=20`, `stride=10`). <!-- id: 12 -->
    - [ ] Implementasikan fungsi `load_data` yang membaca CSV, membersihkan NaNs, dan menerapkan scaling. <!-- id: 13 -->

- [ ] **4. Implementasi Thresholding Dinamis (`Python/utils/thresholding.py`)** <!-- id: 14 -->
    - [ ] Buat fungsi `calculate_reconstruction_error(model, X)`. <!-- id: 15 -->
    - [ ] Buat fungsi `get_threshold_percentile(errors, percentile=95)`. <!-- id: 16 -->
    - [ ] (Opsional) Implementasikan `get_threshold_pot` (Peaks Over Threshold). <!-- id: 17 -->

- [ ] **5. Implementasi Skrip Utama (`Python/main_sota.py`)** <!-- id: 18 -->
    - [ ] Muat konfigurasi dari `config.yaml` (buat file ini dulu). <!-- id: 19 -->
    - [ ] Panggil `load_data` untuk CIC-IDS2017 (Train) dan CSE-CIC-IDS2018 (Test). <!-- id: 20 -->
    - [ ] Inisialisasi dan latih `MultiScaleAutoencoder`. <!-- id: 21 -->
    - [ ] Evaluasi model pada data Test, hitung F1-Score, Precision, Recall, AUC. <!-- id: 22 -->
    - [ ] Simpan model (`.h5`) dan scaler (`.pkl`). <!-- id: 23 -->
    - [ ] Cetak laporan evaluasi ke terminal dan file log. <!-- id: 24 -->

- [ ] **6. Verifikasi & Debugging** <!-- id: 25 -->
    - [ ] Jalankan training pada subset kecil data (1000 sampel) untuk memastikan tidak ada error dimensi. <!-- id: 26 -->
    - [ ] Periksa apakah loss menurun seiring epoch. <!-- id: 27 -->
    - [ ] Pastikan output shape sesuai dengan input shape. <!-- id: 28 -->
