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

# dataset_name = 'TUT'
# training_file = "Data\\Training_data_TUT_Diff_Distance.csv"
# validation_file = "Data\\Test_data_TUT_Diff_Distance.csv"


random_seeds = range(1, 6) 

kernel_sizes = [3, 5, 7]
filters = [32, 64, 128]
hidden_layers_1 = [128,64,32, 128,64]
hidden_layers_2 = [64,32,16, 32, 16]
embedding_dims = [32, 64, 128]
ff_dims = [64, 128, 256]

NUM_NOISE_SAMPLES = 5
model_type = 'SAE-CNN'

with open(log_dir,'a') as file:
        file.write('\n=================' + dataset_name  + f'{model_type}-0to1Norm 10 times 1 noisy test=======') #0to1Norm Joint model x_minNorm Pretrained SAE-256-128-64-

for k in [1]: # kernel_sizes
    for seed in range(1,11): # random_seeds
        save_name = dataset_name  + f'{model_type}-0to1Norm t_run{seed}_noise0' #SAE-256-128-64- JointModel
        mean_errors = []

        train_x, train_y, valid_x, valid_y, test_x, test_y = data_helper.load(training_file, validation_file, random_seed=seed)
        print(train_x.shape, train_y.shape, valid_x.shape,valid_y.shape, test_x.shape, test_y.shape)

        model = TrainingModel(save_name, train_x, dataset_name)
        model.patience=int(p)
        model.b=b
        model.epoch_AE=epoch_sae
        model.epoch_position=epoch_position
        model.loss='mean_squared_error'
        model.opt=Adamax(learning_rate=0.005)

        start_time = time.time()

        # h=model.fit_JointModel(train_x, train_y, valid_x=valid_x, valid_y=valid_y, test_x=test_x, k=k)
        #h=model.fit_CNN (train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        #h=model.fit_AECNNTrans(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        #h=model.fit_transformer(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        # h=model.fit_MLP(train_x, train_y, valid_x=valid_x, valid_y=valid_y, test_x=test_x)
        #h=model.fit_CNNTrans (train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        model.fit(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        # h=model.fit_AETrans(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        # model.fit_AE_MLP(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        # h=model.fit_encoder_MLP(train_x, train_y, valid_x=valid_x, valid_y=valid_y)

        end_time =time.time()
        training_time=end_time-start_time


        longitude_error,latitude_error,mse, rmse, mae, predict_time = model.error(test_x, test_y)
        mean_errors.extend([mse, rmse, mae, predict_time, training_time, dataset_name, model_type, save_name, seed, "0", k])

        with open(log_dir,'a') as file:
            file.write(f'\n--Dataset {dataset_name} run {seed}  : {mean_errors}')
    

        file_name = save_dir_10_run + 'result_10_26.csv'
        headers = [
            'MSE', 'RMSE', 'MAE', 'Predict_Time', 'Training_Time',
            'Dataset', 'Model', 'Save_Name', 'Seed', 'Run', 'K'
        ]

        file_exists = os.path.exists(file_name)

        with open(file_name, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(headers)

            writer.writerow(mean_errors)



