from sklearn.preprocessing import MinMaxScaler,RobustScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import  os
from sklearn.model_selection import train_test_split

# Function to read CSV or XLSX files
def read_file(file_path, sheet_name=""):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        if sheet_name =="":
            return pd.read_excel(file_path)
        else:
            return pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

def add_gaussian_noise(X, mean=0.0, std=5.0, seed=42,
                       no_signal=100, rssi_min=-104, rssi_max=0):
    """
    Add Gaussian noise to RSSI values of a UJIIndoorLoc-style array.
    - Only APs that were actually detected (value != 100) get noise.
    - Undetected APs (100) stay unchanged.
    - Noisy values are clipped to the valid RSSI range [-104, 0].
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    noisy = X.copy()

    detected = X != no_signal
    noise = rng.normal(mean, std, size=X.shape)

    noisy[detected] += noise[detected]
    noisy[detected] = np.clip(noisy[detected], rssi_min, rssi_max)
    return noisy


def add_simple_noise(X, std=5, seed=42, no_signal=100, rssi_min=-104, rssi_max=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    noisy = X.copy()

    detected = X != no_signal
    magnitude = rng.uniform(1, std * std, size=X.shape)   # số ngẫu nhiên trong (1, std^2)
    sign = rng.choice([-1, 1], size=X.shape)              # cộng hoặc trừ ngẫu nhiên

    noisy[detected] += (sign * magnitude)[detected]
    noisy[detected] = np.clip(noisy[detected], rssi_min, rssi_max)
    return noisy

def load(file_path_train, file_path_valid, random_seed=42, sheet_name_train="", sheet_name_test=""):
    # Validate file paths
    if file_path_train is None or file_path_valid is None:
        raise ValueError("File path is None, please provide valid file paths.")

    # Read the data
    train_data_frame = read_file(file_path_train, sheet_name_train)
    test_data_frame = read_file(file_path_valid, sheet_name_test)
    train_data_frame = train_data_frame.fillna(100)
    test_data_frame = test_data_frame.fillna(100)

    # Splitting validation data from training data
    rest_data_frame = train_data_frame.copy()
    valid_data_frame = pd.DataFrame(columns=train_data_frame.columns)
    valid_num = int(len(train_data_frame) / 10)
    
    sample_row = rest_data_frame.sample(valid_num, random_state=random_seed)
    rest_data_frame = rest_data_frame.drop(sample_row.index)

    if not sample_row.empty and not sample_row.isna().all(axis=None):
        valid_data_frame = pd.concat([valid_data_frame, sample_row])

    train_df = rest_data_frame
    wap_cols = [c for c in train_df.columns if c.startswith('WAP') or c.startswith("AP") or c.startswith("MAC")]
    print("number of columns: ", len(wap_cols))
    # NOW split into x and y — after deduplication
    longitude_col = "LONGITUDE" in train_df.columns
    train_x = train_df[wap_cols]    
    test_x = test_data_frame[wap_cols]    
    valid_x = valid_data_frame[wap_cols]
    if "EEIL" in file_path_train:
        train_y = train_df[['X', 'Y']]
        test_y = test_data_frame[['X', 'Y']]
        valid_y = valid_data_frame[['X', 'Y']]
    else:
        test_y = test_data_frame[['LONGITUDE', 'LATITUDE']] if longitude_col else test_data_frame[["x", "y"]]
        train_y = train_df[['LONGITUDE', 'LATITUDE']] if longitude_col else train_df[["x", "y"]]
        valid_y = valid_data_frame[['LONGITUDE', 'LATITUDE']] if longitude_col else valid_data_frame[["x", "y"]]

    return train_x,train_y,valid_x,valid_y,test_x,test_y

def load_train_file(file_path_train, file_path_valid, sheet_name_train="", sheet_name_test=""):
    
    # Read the data
    train_data_frame = read_file(file_path_train, sheet_name_train)
    train_data_frame = train_data_frame.fillna(100)
    test_data_frame = read_file(file_path_valid, sheet_name_test)
    test_data_frame = test_data_frame.fillna(100)

    train_df = train_data_frame.copy()
    #train_data = train_data_frame.sample(frac=1, random_state=42).reset_index(drop=True)

    wap_cols = [c for c in train_df.columns if c.startswith('WAP') or c.startswith("AP") or c.startswith("MAC")]

    # NOW split into x and y — after deduplication
    train_x = train_df[wap_cols]
    test_x = test_data_frame[wap_cols]
    if "EEIL" in file_path_train:
        train_y = train_df[['X', 'Y']]
        test_y = test_data_frame[['X', 'Y']]
    else: 
        train_y = train_df[['LONGITUDE', 'LATITUDE']] if "LONGITUDE" in train_df.columns else train_df[["x", "y"]]
        test_y = test_data_frame[['LONGITUDE', 'LATITUDE']] if "LONGITUDE" in train_df.columns else test_data_frame[["x", "y"]]
    return train_x, train_y, test_x, test_y


def normalizeX_zero_to_one(arr):
    res = np.copy(arr).astype(np.float64)
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if (res[i][j] == 100)|(res[i][j]==None):
                res[i][j] = 0
            elif res[i][j]<-100:
                res[i][j]=0
            else :
                res[i][j] = 0.01 * (100 + res[i][j])
    return res

def normalizeX_zero_one(arr):
    res = np.copy(arr).astype(np.float64)
    x_max = np.max(res)
    x_min = np.min(res)
    x_scaled = x_max - x_min
    
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            res[i][j] = (res[i][j]- x_min)/x_scaled
    return res


def normalizeX_zero_one_edge_based(arr):
    res = np.copy(arr).astype(np.float64)
    #x_max = np.max(res)
    x_min = np.min(res)
    
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if res[i][j] > 0:
                res[i][j] = 0
            else:
                res[i][j] = (res[i][j]- x_min)/(-x_min)
    return res

def normalizeX_powed(arr,b):
    res = np.copy(arr).astype(np.float64)
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            if (res[i][j] >50)|(res[i][j]==None)|(res[i][j]<-95):
                res[i][j] = 0
            elif (res[i][j]>=0):
                res[i][j]=1

            else :
                res[i][j] = ((105 + res[i][j])/105.0) ** b
            # res[i][j] = (0.01 * (110 + res[i][j])) ** 2.71828
    return res


def normalizeX(arr,b=2.71828): 

    # return normalizeX_powed(arr,b).reshape([-1, arr.shape[1]])

    return normalizeX_zero_to_one(arr).reshape([-1, arr.shape[1]])

    #return normalizeX_zero_one_edge_based(arr).reshape([-1, arr.shape[1]]) 


class NormY(object):
    long_max=None
    long_min=None
    lati_max=None
    lati_min=None
    long_scale=None
    lati_scale=None

    def __init__(self):
        pass

    def fit(self,long,lati):
        self.long_max=max(long)
        self.long_min=min(long)
        self.lati_max=max(lati)
        self.lati_min=min(lati)
        self.long_scale=self.long_max-self.long_min
        self.lati_scale=self.lati_max-self.lati_min

    def normalizeY(self,longitude_arr, latitude_arr):

        longitude_arr = np.reshape(longitude_arr, [-1, 1])
        latitude_arr = np.reshape(latitude_arr, [-1, 1])
        long=(longitude_arr-self.long_min)/self.long_scale
        lati=(latitude_arr-self.lati_min)/self.lati_scale
        # long = longitude_arr-self.long_min
        # lati=latitude_arr-self.lati_min
        return long,lati

    def reverse_normalizeY(self,longitude_arr, latitude_arr):

        longitude_arr = np.reshape(longitude_arr, [-1, 1])
        latitude_arr = np.reshape(latitude_arr, [-1, 1])
        long=(longitude_arr*self.long_scale)+self.long_min
        lati=(latitude_arr*self.lati_scale)+self.lati_min
        #long = longitude_arr+self.long_min
        #lati=latitude_arr+self.lati_min
        return long,lati

