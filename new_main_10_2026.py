import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
from keras.optimizers import Nadam, Adamax, Adam
import math
import data_helper
import openpyxl
import os
import time
import matplotlib.pyplot as plt
import csv

base_dir= os.getcwd()

TF_ENABLE_ONEDNN_OPTS=0
from model import TrainingModel
b = 2.8
p = 10
epoch_sae = 80
epoch_position = 100

save_picture_dir = os.path.join(base_dir,'Images\\08-2025')
save_dir_10_run = os.path.join(base_dir,'results\\10_run\\')
log_dir='New_new_log_10_2026.txt'

dataset_name = 'UJIIndoorLoc'
training_file = "UJIIndoorLocData\\trainingData.csv"
validation_file = "UJIndoorLocData\\validationData.csv"

# training_file = r"Data\Training_data_TUT_Diff_Distance.csv"
# validation_file = r"Data\Test_data_TUT_Diff_Distance.csv"
# dataset_name = "TUT"

train_x, train_y, valid_x, valid_y, test_x, test_y = data_helper.load(training_file, validation_file, random_seed=1)


model_types = [ 'SAE-MLP', 'SAE-CNN', 'MLP', 'CNN','Transformer', 'SAE-Transformer', 'SAE-CNN-Transformer'] # 'SAE-MLP' #

# save_names = [
# "TampereSAE(256-128-64)-MLP-0to1Norm_run2",
# "TampereSAE(256-128-64)-CNN-0to1Norm_run1",
# "TampereMLP-0to1Norm_run4",
#"Tampere-CNN-0to1Norm_run5",
#"Tampere-Transformer-0to1Norm_run4",
#"Tampere-SAE-Transformer-0to1Norm_run4",
#"Tampere-SAE-CNN-Transformer-0to1Norm_run1"]
save_names = [
"UJIIndoorLoc-SAE(256-128-64)-MLP-0to1Norm_run3",
"UJIIndoorLoc-SAE-CNN-0to1Norm_run5",
"UJIIndoorLoc-MLP-0to1Norm_run1",
"UJIIndooLocOrg-CNN-0to1Norm_run5",
"UJIIndooLocOrg-Transformer-0to1Norm_run1",
"UJIIndooLocOrg-SAE-Transformer-0to1Norm_run4",
"UJIIndooLocOrg-SAE-CNN-Transformer-0to1Norm_run4"]

for run in range(3):
    model_type = model_types[run]
    save_name = save_names[run]
    model = TrainingModel(save_name, train_x, dataset_name)
    model._preprocess(train_x, train_y, valid_x, valid_y)


    stds = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    NUM_NOISE_SAMPLES = 5
  

    for std in stds:
        for seed in range(NUM_NOISE_SAMPLES):
            mean_errors = []
            noisy_test_file = f"UJIIndoorLocData\\validationData_noisy_int_std{std}_{seed}.csv"
            # f"TUTData\\Test_data_TUT_noisy_int_std{std}_{seed}.csv"
            val_noisy = pd.read_csv(noisy_test_file)
            wap_cols = [c for c in val_noisy.columns if c.startswith("WAP") or c.startswith("AP")]
            longitude_col = "LONGITUDE" in val_noisy.columns
            Xte_noisy = val_noisy[wap_cols]
            yte_noisy = val_noisy[["LONGITUDE", "LATITUDE"]] if longitude_col else val_noisy[["x", "y"]]

            longitude_error,latitude_error,mse, rmse, mae, predict_time = model.error(Xte_noisy, yte_noisy)
            mean_errors.extend([mse, rmse, mae, predict_time, "mean=0", dataset_name, model_type, save_name, seed, noise])

            file_name = f"results\\10_run\\new_new_new_results.csv"

            headers = [
                'MSE', 'RMSE', 'MAE', 'Predict_Time', 'Training_Time',
                'Dataset', 'Model', 'Save_Name', 'Seed', 'Noise'
            ]

            file_exists = os.path.exists(file_name)

            with open(file_name, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                if not file_exists:
                    writer.writerow(headers)

                writer.writerow(mean_errors)