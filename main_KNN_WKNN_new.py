import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os
import data_helper
import datetime
import time

train = pd.read_csv(r"UJIndoorData\trainingData.csv")
val = pd.read_csv(r"UJIndoorData\validationData.csv")
dataset_name = "UJIIndoorLoc"

# train = pd.read_csv(r"Data\Training_data_TUT_Diff_Distance.csv")
# val = pd.read_csv(r"Data\Test_data_TUT_Diff_Distance.csv")
# dataset_name = "TUT"

wap_cols = [c for c in train.columns if c.startswith("WAP") or c.startswith("AP")]

Xtr = train[wap_cols].values.astype(float)
Xte = val[wap_cols].values.astype(float)
longitude_col = "LONGITUDE" in train.columns
ytr = train[["LONGITUDE", "LATITUDE"]].values.astype(float) if longitude_col else train[["x", "y"]].values.astype(float)
yte = val[["LONGITUDE", "LATITUDE"]].values.astype(float) if longitude_col else val[["x", "y"]].values.astype(float)


norm =  True  # False # True  #    # Set to True if you want to normalize the data
no_signal_value = 100  # Value to replace no signal (100) with
if norm:
    Xtr = data_helper.normalizeX(Xtr)

if dataset_name == "TUT":
        # For TUT dataset, replace -105 with 100 before normalization
        Xtr = np.where(Xtr == -105, 100, Xtr)
        Xte = np.where(Xte == -105, 100, Xte)

for i in range(1,6):
    # Load the noisy test data
    noisy_test_file = f"UJIndoorData\\validationData_noisy_std5_{i}.csv"
    val_noisy = pd.read_csv(noisy_test_file)
    Xte = val_noisy[wap_cols].values.astype(float)
    yte = val_noisy[["LONGITUDE", "LATITUDE"]].values.astype(float) if longitude_col else val_noisy[["x", "y"]].values.astype(float)

    # Normalize the data if needed
    if norm:
        Xte = data_helper.normalizeX(Xte)

    # Euclidean distance in RSS space
    D = cdist(Xte, Xtr, metric="euclidean")

    results = []

    for k in [1]:#[5, 7, 9]:
        idx = np.argpartition(D, k, axis=1)[:, :k]
        # sort these k by actual distance
        rows = np.arange(D.shape[0])[:, None]
        dk = D[rows, idx]
        order = np.argsort(dk, axis=1)
        idx_sorted = idx[rows, order]
        dk_sorted = dk[rows,  order]

        # KNN: simple average of coordinates
        coords_knn = ytr[idx_sorted]  # (n_test, k, 2)
        start_time_knn = time.time()
        pred_knn = coords_knn.mean(axis=1)
        end_time_knn = time.time()
        knn_time = end_time_knn - start_time_knn
        knn_pred_time = knn_time / len(Xte)*1000  # Average prediction time per sample in milliseconds
        err_knn = np.sqrt(((pred_knn - yte) ** 2).sum(axis=1))
        mean_err_knn = err_knn.mean()

        # WKNN: weight = 1/distance (avoid div by zero)
        w = 1.0 / (dk_sorted + 1e-6)
        w_norm = w / w.sum(axis=1, keepdims=True)
        start_time_wknn = time.time()
        pred_wknn = (coords_knn * w_norm[:, :, None]).sum(axis=1)
        end_time_wknn = time.time()
        wknn_time = end_time_wknn - start_time_wknn
        wknn_pred_time = wknn_time / len(Xte) * 1000  # Average prediction time per sample in milliseconds
        err_wknn = np.sqrt(((pred_wknn - yte) ** 2).sum(axis=1))
        mean_err_wknn = err_wknn.mean()

        print(f"k={k}: KNN={mean_err_knn:.4f} m, WKNN={mean_err_wknn:.4f} m")

        data_to_save = pd.DataFrame({
        "Dataset": dataset_name,
        "k": k,
        "Noise": "5dB Gaussian Noise no normalization",
        "Noisy": i,
        "Normalized": norm,
        "True_Longitude": yte[:, 0],
        "True_Latitude": yte[:, 1],
        "Pred_KNN_Longitude": pred_knn[:, 0],
        "Pred_KNN_Latitude": pred_knn[:, 1],
        "Pred_WKNN_Longitude": pred_wknn[:, 0],
        "Pred_WKNN_Latitude": pred_wknn[:, 1],
        "Error_KNN": err_knn,
        "Error_WKNN": err_wknn,
    })

        file_name = f'knn_wknn_detail_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'
        df = pd.DataFrame(data_to_save)
        df.to_csv(
            file_name,
            mode='a',                
            index=False,
            header=not os.path.exists(file_name)   # chỉ ghi header nếu file chưa tồn tại
        )

        results.append(
            {"Dataset": dataset_name, "k": k, "Noise": "5dB Gaussian Noise no normalization", "noisy": i, "KNN_mean_error_m": mean_err_knn, "KNN_avg_pred_time_s": knn_pred_time, "WKNN_mean_error_m": mean_err_wknn, "WKNN_avg_pred_time_s": wknn_pred_time, "Normalized": norm, "No_signal_value": no_signal_value}
        )


    file_name = f'knn_wknn_results_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'
    df = pd.DataFrame(results)
    df.to_csv(
        file_name,
        mode='a',                
        index=False,
        header=not os.path.exists(file_name)   # chỉ ghi header nếu file chưa tồn tại
    )

