import numpy as np
import os

path = r"d:\Codingan\python\nids-projects\nids-cnn-lstm-autoencoder\data\research\rs10_full_feature_filter_corr090\processed\shards\cic\train\cic_train_0000.npz"

if os.path.exists(path):
    data = np.load(path)
    print(f"Keys: {list(data.keys())}")
    for k in data.keys():
        print(f"Shape of {k}: {data[k].shape}")
else:
    print(f"File not found: {path}")
