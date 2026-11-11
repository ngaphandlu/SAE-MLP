import pandas as pd
import numpy as np
import data_helper

SEED = 42
NUM_NOISE_SAMPLES = 5

stds = [1, 2, 3, 4, 5] #

test = pd.read_csv(f"UJIIndoorLocData\\validationData.csv")
# 'TUTData\\Test_data_TUT_Diff_Distance.csv'

wap_cols = [c for c in test.columns if c.startswith("WAP") or c.startswith("AP")]  # WAP001..WAP520

X_test = test[wap_cols].values
for std in stds:
    for i in range(NUM_NOISE_SAMPLES):
        #X_test_noisy = data_helper.add_gaussian_noise(X_test, rssi_min=-102, no_signal=-105, std=std, seed=SEED + i)
        X_test_noisy = data_helper.add_gaussian_noise(X_test, std=std, seed=SEED + i)

        test_noisy = test.copy()
        test_noisy[wap_cols] = X_test_noisy
        test_noisy.to_csv(f"UJIndoorLocData\\validationData_noisy_int_std{std}_{i}.csv", index=False)
        # f"TUTData\\Test_data_TUT_noisy_int_std{std}_{i}.csv", index=False)
        