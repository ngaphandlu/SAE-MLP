#from encoder_model import EncoderDNN #UJIndoor
from model import TrainingModel
import numpy as np
import data_helper
import time
import keras
import openpyxl
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras.optimizers import Adam,Adagrad,Adadelta,Nadam,Adamax,SGD,RMSprop
from keras.losses import mean_squared_error #MSE,MAE,MAPE,MSLE,KLD,squared_hinge,hinge,categorical_hinge,categorical_crossentropy,sparse_categorical_crossentropy,kullback_leibler_divergence,poisson

#os.environ["CUDA_VISIBLE_DEVICES"]='0'
#from keras.backend.tensorflow_backend import set_session
#config=tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction=0.9
#set_session(tf.Session(config=config))

# OPT=[Adam,Adagrad,Nadam,Adamax,RMSprop]
OPT=[Adamax]

LOSS=[mean_squared_error]
LR=[0.005]
rng=np.random.RandomState(888)

base_dir= os.getcwd()
# train_csv_path = os.path.join(base_dir,'trainingData.csv')
test_csv_path=os.path.join(base_dir,'TestData.csv')
valid_csv_path=os.path.join(base_dir,'ValuationData.csv')
train_csv_path=os.path.join(base_dir,'TrainingData.csv')

log_dir='New_new_log_07_2025.txt'

save_picture_dir = os.path.join(base_dir,'Images\\08-2025')
save_dir_10_run = os.path.join(base_dir,'results\\10_run\\')
labels = ['105']

dataset_name = 'UJIIndoorLoc-B'


if __name__ == '__main__':
    b = 2.8
    p = 10
    epoch_sae = 60
    epoch_position = 100
    dp = 0.7
    for i in range(1, 6):
        for m in labels:
            # UJIIndoorLoc-B
            training_file = "_training_data_m" + m + ".csv"
            validation_file = "_validation_data_m" + m +".csv"

            
            for k in [1]:
                mean_errors = []
                rmses = []
                maes = []
                predict_times = []
                training_times = []
                for b in range(3):
                    #Bo du lieu chia theo building - UJIndoorLoc-B
                    train_file_name = "UJIIndoorLocData\\B"+ str(b) + training_file
                    test_file_name = "UJIndoorLocData\\B"+ str(b) + validation_file

                
                    train_x, train_y, valid_x, valid_y, test_x, test_y = data_helper.load(train_file_name, test_file_name, random_seed=i)
                    print(train_x.shape, train_y.shape, valid_x.shape,valid_y.shape, test_x.shape, test_y.shape)
                    for opt in OPT:
                        for lr in LR:
                            # Training
                            #encode_dnn_model = EncoderDNN()
                            dataset_name_b = dataset_name + '_B' + str(b)
                            save_name = dataset_name_b  + '_m' + str(m) + '-SAE-MLP-0to1Norm' + str(i) #AECNNTransSwiGLU SAE-256-128-64-
                            with open(log_dir,'a') as file:
                                file.write('\n=================' + save_name + '=====================')

                            model = TrainingModel(save_name, train_x, dataset_name_b)
                            model.patience=int(p)
                            model.b=b
                            model.epoch_AE=epoch_sae
                            model.epoch_position=epoch_position
                            model.loss='mean_squared_error'
                            model.opt=opt(learning_rate=lr)
                            strat = time.time()



                            #h= model.fit_JointModel(train_x, train_y, valid_x=valid_x, valid_y=valid_y, test_x=test_x, k=k)
                            #h = model.fit_CNN(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
                            #h = model.fit(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
                            #h = model.fit_AECNNTrans(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
                            #h = model.fit_transformer(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
                            #h = model.fit_CNNTrans(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
                            #h = model.fit_AETrans(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
                            #h = model.fit_MLP(train_x, train_y, valid_x=valid_x, valid_y=valid_y, test_x=test_x)
                            h = model.fit_AE_MLP(train_x, train_y, valid_x=valid_x, valid_y=valid_y)
                            #h=model.fit_encoder_MLP(train_x, train_y, valid_x=valid_x, valid_y=valid_y)

                            end=time.time()
                            trining_time=end-strat

                            longitude_error,latitude_error,mse, rmse, mae, predict_time = model.error(test_x, test_y) # UJIndoor
                            mean_errors.append(mse)
                            rmses.append(rmse)
                            maes.append(mae)
                            predict_times.append(predict_time)
                            training_times.append(trining_time)
                           
                file_name = save_dir_10_run + dataset_name + '_10_run.xlsx'
                workbook = openpyxl.load_workbook(file_name)
                sheet = workbook["SAE-MLP"] #"Trans", "SAECNNTrans", "CNNTrans", "AETrans", "AEKAN", "MLP", "AEMLP", "SAECNN"
                max_row = sheet.max_row
                            
                mean_errors.append(np.average(mean_errors))
                
                rmses.append(np.average(rmses))
                maes.append(np.average(maes))            
                predict_times.append(np.average(predict_times))
                training_times.append(np.average(training_times))

                mean_errors.append('Position error')
                mean_errors.append(save_name)
                rmses.append('RMSE')
                maes.append('MAE')
                predict_times.append('Prediction time')
                training_times.append('Training time')
             
                print(mean_errors)
                with open(log_dir,'a') as file:
                    file.write('\n--Dataset ' + dataset_name + ' Position error run ' + str(i) + ' : ' + str(mean_errors))
                    file.write('\n--Dataset ' + dataset_name + ' RMSE run ' + str(i) + ' : ' + str(rmses))
                    file.write('\n--Dataset ' + dataset_name + ' MAE run ' + str(i) + ' : ' + str(maes))
                
                sheet.append(mean_errors)
                sheet.append(rmses)
                sheet.append(maes)
                sheet.append(predict_times)
                sheet.append(training_times)

            
                workbook.save(file_name)       