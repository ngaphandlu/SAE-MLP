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



base_dir= os.getcwd()

TF_ENABLE_ONEDNN_OPTS=0
from model import TrainingModel
b = 2.8
p = 10
epoch_sae = 80
epoch_position = 100


save_picture_dir = os.path.join(base_dir,'Images\\08-2025')
save_dir_10_run = os.path.join(base_dir,'results\\10_run\\')
log_dir='New_new_log_07_2025.txt'


# dataset_name = 'UJIIndooLoc'
# training_file = "UJIIndoorLocData\\trainingData.csv"
# validation_file = "UJIIndoorLocData\\validationData.csv"

dataset_name = 'TUT'
training_file = "TUTData\\Training_data_TUT_Diff_Distance.csv"
validation_file = "TUTData\\Test_data_TUT_Diff_Distance.csv"

with open(log_dir,'a') as file:
        file.write('\n=================' + dataset_name  + '-SAE(256-128-64)-MLP-0to1Norm=======') #0to1Norm Joint model x_minNorm Pretrained SAE-256-128-64-

for k in [1]:
    for i in range (1, 6):
        save_name = dataset_name  + '-SAE(256-128-64)-MLP-0to1Norm_run' + str(i) #SAE-128-64- JointModel
        mean_errors = []
   
        
        
        train_x, train_y, valid_x, valid_y, test_x, test_y = data_helper.load(training_file, validation_file)
        print(train_x.shape, train_y.shape, valid_x.shape,valid_y.shape, test_x.shape, test_y.shape)

        model = TrainingModel(save_name, train_x, dataset_name)
        model.patience=int(p)
        model.b=b
        model.epoch_AE=epoch_sae
        model.epoch_position=epoch_position
        model.loss='mean_squared_error'
        model.opt=Adamax(learning_rate=0.005)

        start_time = time.time()

        #h=model.fit_JointModel(train_x, train_y, valid_x=valid_x, valid_y=valid_y, test_x=test_x, k=k)
        #h=model.fit_CNN (train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        #h=model.fit_AECNNTrans(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        #h=model.fit_transformer(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        #h=model.fit_MLP(train_x, train_y, valid_x=valid_x, valid_y=valid_y, test_x = test_x)
        #h=model.fit_CNNTrans (train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        # h=model.fit(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        # h=model.fit_AETrans(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        h=model.fit_AE_MLP(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
        #h=model.fit_encoder_MLP(train_x, train_y, valid_x=valid_x, valid_y=valid_y)

        end_time =time.time()
        training_time=end_time-start_time


        longitude_error,latitude_error,mse, rmse, mae, predict_time = model.error(test_x, test_y)
        mean_errors.extend([mse, rmse, mae, predict_time, training_time, dataset_name, 'SAE(256-128-64)-MLP', save_name])

        with open(log_dir,'a') as file:
            file.write('\n--Dataset ' + dataset_name + ' run ' + str(i) + ' : ' + str(mean_errors))
      

        file_name = save_dir_10_run + 'UJIIndoorLoc_10_run.xlsx'
        workbook = openpyxl.load_workbook(file_name)
        sheet = workbook["SAE-MLP"] #"SAECNN", "AETrans", "SAECNNTrans", "SAE-MLP", "MLP", "PCA-MLP", "CNN"
        max_row = sheet.max_row
                
        
        print(mean_errors)

        sheet.append(mean_errors)

        workbook.save(file_name) 



