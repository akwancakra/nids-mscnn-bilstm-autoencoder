# Checklist Implementasi SOTA NIDS

Gunakan checklist ini untuk melacak progres implementasi SOTA Multi-Scale CNN-BiLSTM Autoencoder.

- [ ] **1. Persiapan Struktur Proyek**
    - [ ] Folder `Python/models`, `Python/preprocessing`, `Python/utils` dibuat.
    - [ ] File lama dipindahkan ke `Python/legacy/`.
    - [ ] `Python/__init__.py` ada di setiap folder baru.

- [ ] **2. Implementasi Model SOTA (`Python/models/sota_model.py`)**
    - [ ] `MultiScaleAutoencoder` didefinisikan (subclass `tf.keras.Model`).
    - [ ] Encoder Multi-Scale (k=3, 5, 7) terimplementasi.
    - [ ] Bi-LSTM dan Attention Mechanism terimplementasi.
    - [ ] Decoder (Mirror Encoder) terimplementasi.
    - [ ] `compile_model` (MSE, Adam) ditambahkan.

- [ ] **3. Implementasi Pipeline Data (`Python/preprocessing/data_loader.py`)**
    - [ ] `GlobalScaler` (fit on train, transform on test) dibuat.
    - [ ] `create_sequences` (sliding window, default `window_size=20`) dibuat.
    - [ ] `load_data` (read CSV, clean NaNs, scale) dibuat.

- [ ] **4. Implementasi Thresholding Dinamis (`Python/utils/thresholding.py`)**
    - [ ] `calculate_reconstruction_error` dibuat.
    - [ ] `get_threshold_percentile` dibuat.
    - [ ] `get_threshold_pot` (opsional) dibuat.

- [ ] **5. Implementasi Skrip Utama (`Python/main_sota.py`)**
    - [ ] Konfigurasi dimuat dari `config.yaml`.
    - [ ] `load_data` dipanggil untuk Train (CIC-IDS2017) dan Test (CSE-CIC-IDS2018).
    - [ ] `MultiScaleAutoencoder` diinisialisasi dan dilatih.
    - [ ] Evaluasi model pada data Test dilakukan.
    - [ ] Model (`.h5`) dan scaler (`.pkl`) disimpan.
    - [ ] Laporan evaluasi dicetak.

- [ ] **6. Verifikasi & Debugging**
    - [ ] Training pada subset data berhasil tanpa error dimensi.
    - [ ] Loss menurun seiring epoch.
    - [ ] Output shape sesuai dengan input shape.
