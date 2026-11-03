import pandas as pd
import numpy as np
import data_helper

SEED = 42
NUM_NOISE_SAMPLES = 5


test = pd.read_csv("UJIndoorData\\validationData.csv")      # UJIIndoorLoc test/validation set
wap_cols = [c for c in test.columns if c.startswith("WAP")]  # WAP001..WAP520

X_test = test[wap_cols].values
for i in range(NUM_NOISE_SAMPLES):
    # X_test_noisy = data_helper.add_gaussian_noise(X_test, mean=0, std=5, seed=SEED + i)
    X_test_noisy = data_helper.add_simple_noise(X_test, std=5, seed=SEED + i)

    test_noisy = test.copy()
    test_noisy[wap_cols] = X_test_noisy
    test_noisy.to_csv(f"UJIndoorData\\validationData_simple_std5_{i+1}.csv", index=False)