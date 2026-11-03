import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
from keras.layers import LSTM, GRU, Add, Dense, Input, Dropout, Conv1D, Flatten, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization, GlobalAveragePooling1D, ReLU, Conv1DTranspose, AveragePooling1D, LeakyReLU
from keras.constraints import max_norm
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
#from keras.optimizers.legacy import Adamax, Adam # dung cho env py11
from keras.optimizers import Adam, Adamax, Nadam #,Adagrad,Adadelta,Nadam,Adamax,SGD,RMSprop
import data_helper
import numpy as np
import pandas as pd
import keras
import time
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
# from transformer import  TransformerEncoder, WiFiRSSEmbedding, SwiGLU, AttentionPooling
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

base_dir=os.getcwd()
AE_model_dir=os.path.join(base_dir,'AE_model')
Building_model_dir=os.path.join(base_dir,'Building_model')
Floor_model_dir=os.path.join(base_dir,'Floor_model')
model_dir=os.path.join(base_dir,'Model_dir') 
result_dir = os.path.join(base_dir,'results') + '\\'


#************My code
#import os
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"
#import keras_nlp
from keras import layers
#keras.mixed_precision.set_global_policy("mixed_float16")
keras.mixed_precision.set_global_policy("float32")


embed_dim = 32  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer



Train_AE=True # False #

class TrainingModel(object):
    normY = data_helper.NormY()
    def __init__(self, save_model_name, x, dataset_name="UJIIndoorLoc"):
        self.epoch_AE=6
        self.epoch_floor=6
        self.epoch_position=6
        self.epoch_building=6
        self.loss='mean_squared_error'
        self.opt='adam'
        self.lr=0.0001
        self.dropout=0.7
        self.patience=3
        self.b=2.71828
        self.input_size =  x.shape[1]  # 200 #
        self.save_name = save_model_name 
        self.input = Input((self.input_size,)) 
        self.AE_floor_bottleneck = '256_128-64'
        self.model_save_path=os.path.join(model_dir,save_model_name + '.keras')
        self.dataset_name = dataset_name

#################AE_model
    def fnBuildAEModel(self):
        self.encode_layer = Dense(256, activation='relu', name='en1')(self.input)
        self.encode_layer = Dropout(0.2, name='dropout_encoder_1')(self.encode_layer)
        self.encode_layer = Dense(128, activation='relu', name='en2')(self.encode_layer)
        self.encode_layer = Dropout(0.2, name='dropout_encoder_2')(self.encode_layer)
        # self.encode_layer = Dense(64, activation='relu', name='en4')(self.encode_layer)
        # self.encode_layer = Dropout(0.2, name='dropout_encoder_4')(self.encode_layer)
        self.encode_layer = Dense(64, activation='relu', name='en3')(self.encode_layer)

        decode_layer = Dense(128, activation='relu', name='de-1')(self.encode_layer)
        decode_layer = Dropout(0.2, name='dropout_decoder_1')(decode_layer)
        decode_layer = Dense(256, activation='relu', name='de-2')(decode_layer) 
        decode_layer = Dropout(0.2, name='dropout_decoder_2')(decode_layer)
        # decode_layer = Dense(256, activation='relu', name='de-4')(decode_layer)
        # decode_layer = Dropout(0.2, name='dropout_decoder_4')(decode_layer)
        decode_layer = Dense(self.input_size, activation='relu', name='de-3')(decode_layer) #*********************465, 520, elu
        self.encoder_model_256_128= Model(inputs=self.input, outputs=decode_layer)
        self.bottleneck_model_256_128= Model(inputs=self.input, outputs=self.encode_layer)

    
    def fnBuildLocationModel(self, kernel_size=3):
                 
        if (not Train_AE) & (
        not os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck' + self.AE_floor_bottleneck + '_' + self.save_name+ '.keras')))):
            self.position_base_model = self.bottleneck_model_256_128
        else:
            self.position_base_model = load_model(
                os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck +'_'+ self.save_name+ '.keras')))

        for layer in self.position_base_model.layers:
            layer.trainable = True
            # 正则化
            # layer.kernet_regularizer = keras.regularizers.l2(l=0.0001)

        position_net_input = Reshape((self.position_base_model.output_shape[1], 1))(self.position_base_model.output)

        position_net = Conv1D(64, kernel_size, padding='same', activation='relu')(position_net_input)
        position_net = Conv1D(32, kernel_size, padding='same', activation='relu')(position_net)
        # position_net = Conv1D(128, kernel_size, padding='same', activation='relu')(position_net)
        
        position_net = Flatten()(position_net)
        self.position_predict_output = Dense(2, activation='linear')(position_net)
        self.position_model = Model(inputs=self.position_base_model.input, outputs=self.position_predict_output)


    def _preprocess(self, x, y, valid_x, valid_y, test_x=None, k=7):

        self.normY.fit(y.iloc[:, 0].values, y.iloc[:, 1].values)
        self.longitude_normalize_y, self.latitude_normalize_y = self.normY.normalizeY(y.iloc[:, 0].values, y.iloc[:, 1].values)
        self.longitude_normalize_valid_y, self.latitude_normalize_valid_y = self.normY.normalizeY(valid_y.iloc[:, 0].values,
                                                                                                  valid_y.iloc[:, 1].values)

        self.normalize_x = data_helper.normalizeX(x,self.b)
        self.normalize_valid_x = data_helper.normalizeX(valid_x,self.b)
        if test_x is not None:
            self.normalize_test_x = data_helper.normalizeX(test_x,self.b)

        # self.apply_pca(n_components = 300)
        # self.apply_random_forest(n_features=64)

    def apply_pca(self, n_components=300, variance_threshold=0.95):

        if isinstance(n_components, str) and n_components == 'auto':
            # Find number of components for desired variance
            pca_temp = PCA()
            pca_temp.fit(self.normalize_x)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= 0.95) + 1
            print(f"Auto-selected {n_components} components for {0.95*100}% variance")
        
        self.pca = PCA(n_components, random_state=42)
        
        # if  not os.path.exists("train_x_"+ str(n_components) + ".csv"):
        self.normalize_x = self.pca.fit_transform(self.normalize_x)

        self.normalize_valid_x = self.pca.transform(self.normalize_valid_x)

        self.normalize_test_x = self.pca.transform(self.normalize_test_x)
       

    def apply_random_forest(self, n_features=50, method='importance'):
        """
        Apply Random Forest for feature selection
        
        Parameters:
        - n_features: number of features to select
        - method: 'importance' or 'permutation'
        """
        print(f"Applying Random Forest feature selection ({method})...")

        original_feature_count = self.normalize_x.shape[1]
           
        # Train Random Forest classifier
        self.rf_selector = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10  # Prevent overfitting
        )

        self.position=np.hstack([self.longitude_normalize_y,self.latitude_normalize_y])
        self.position_valid=np.hstack([self.longitude_normalize_valid_y,self.latitude_normalize_valid_y])
        
        print("Training Random Forest for feature selection...")
        self.rf_selector.fit(self.normalize_x, self.position)
        
        if method == 'importance':
            # Get feature importances
            importances = self.rf_selector.feature_importances_
            
            # Select top features by importance
            top_indices = np.argsort(importances)[-n_features:]
            
        elif method == 'permutation':
            from sklearn.inspection import permutation_importance
            
            # Calculate permutation importance (more robust but slower)
            print("Calculating permutation importance (this may take a while)...")
            perm_importance = permutation_importance(
                self.rf_selector, self.normalize_x, self.position,
                n_repeats=5, random_state=42, n_jobs=-1
            )
            
            importances = perm_importance.importances_mean
            top_indices = np.argsort(importances)[-n_features:]
        
        
        print(f"Selected {n_features} most important features out of {original_feature_count}")
        print(f"Random Forest accuracy on full features: {self.rf_selector.score(self.normalize_x, self.position):.4f}")
        
        # Apply feature selection to training, validation, and test sets
        self.normalize_x = self.normalize_x[:, top_indices]
        self.normalize_valid_x = self.normalize_valid_x[:, top_indices]
        self.normalize_test_x = self.normalize_test_x[:, top_indices]
        
               
        # Store importance scores and selected features
        self.feature_importances = importances
        self.selected_features = top_indices
        
        #return X_train_selected, X_test_selected, importances, top_indices

    def fit(self, x, y, valid_x, valid_y, kernel_size = 3, test_x=None):

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        early_stopping_ae = EarlyStopping(monitor='val_loss', patience=self.patience)
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        # early_stopping=myEarlyStopping(monitor='val_acc', patience=self.patience,baseline=0.975)
        # Data pre-processing
        self._preprocess(x, y, valid_x, valid_y)
        
        h_pos = None
        adam=Adam(learning_rate=0.001) #***
#############fit_AE
        if Train_AE:
            self.fnBuildAEModel()
            self.encoder_model=self.encoder_model_256_128
            self.bottleneck_model=self.bottleneck_model_256_128
            self.encoder_model.compile(
                loss=self.loss,
                optimizer=adam #self.opt

            )
            h_AE=self.encoder_model.fit(
                self.normalize_x, self.normalize_x,
                validation_data=(self.normalize_valid_x,self.normalize_valid_x), 
                epochs=self.epoch_AE,
                batch_size=64,
                callbacks=[early_stopping_ae])
            self.bottleneck_model.save(os.path.join(AE_model_dir,('AE_bottleneck_'+self.AE_floor_bottleneck+'_'+ self.save_name+'.keras')))
            self.encoder_model.save('AE_model/AE_'+self.AE_floor_bottleneck + '_'+ self.save_name + '.keras')


        self.fnBuildLocationModel(kernel_size = kernel_size)
        self.position=np.hstack([self.longitude_normalize_y,self.latitude_normalize_y])
        self.position_valid=np.hstack([self.longitude_normalize_valid_y,self.latitude_normalize_valid_y])
        self.position_model.compile(
            loss=self.loss,
            optimizer=self.opt
        )
        # self.position_model.summary()
        h_pos=self.position_model.fit(
            self.normalize_x, 
            self.position,
            validation_data=(self.normalize_valid_x,self.position_valid), # valid_x
            epochs=self.epoch_position, batch_size=64,
            callbacks=[early_stopping, save_best]
            )
            #self.position_model.save(self.model_save_path)

        return h_pos


    

    def predict(self, x):

        predict_longitude=[]
        predict_latitude=[]
        x = data_helper.normalizeX(x,self.b) # normal
        #x = np.hstack([x, self.wknn_test_y])
        # x = self.normalize_test_x # if use PCA

        self.position_model = load_model(self.model_save_path)

        # Get model predictions
        outputs = self.position_model.predict(x)
        
        # Handle different output formats
        if isinstance(outputs, list):
            # Multiple outputs (reconstruction, position)
            predict_Location = outputs[1]  # Position output
        else:
            # Single output
            predict_Location = outputs
        
       # Debug: Check what type of tensor we have
        print(f"Prediction type: {type(predict_Location)}")
        print(f"Prediction shape: {predict_Location.shape}")
        
        # Handle RaggedTensor case
        if isinstance(predict_Location, tf.RaggedTensor):
            print("Warning: Model output is RaggedTensor, converting to dense")
            try:
                predict_Location = predict_Location.to_tensor()
            except Exception as e:
                print(f"Cannot convert to tensor: {e}")
                # Extract first coordinate from each sample
                predict_Location = tf.stack([
                    predict_Location[i][0] for i in range(predict_Location.nrows())
                ])
        
        # Ensure we have a 2D tensor with shape [batch_size, 2]
        if len(predict_Location.shape) == 1:
            predict_Location = tf.expand_dims(predict_Location, 0)
        
        # Extract coordinates safely
        if predict_Location.shape[-1] >= 2:
            predict_longitude = predict_Location[:, 0]
            predict_latitude = predict_Location[:, 1] 
        else:
            raise ValueError(f"Expected at least 2 coordinates, got shape: {predict_Location.shape}")
        
        # Apply reverse normalization
        predict_longitude, predict_latitude = self.normY.reverse_normalizeY(
            predict_longitude, predict_latitude
        )
        
        return predict_longitude, predict_latitude

        # không reverse
        # predict_longitude, predict_latitude = predict_Location[:, 0], predict_Location[:, 1]

    def build_CNN_model(self, input_shape, kernel_size=3):

        input = Input(shape=(input_shape,1))

        x = Conv1D(64, kernel_size, activation='relu')(input)
        x = Conv1D(32, kernel_size, activation='relu')(x)

        x = Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output = Dense(2, activation='linear')(x)

        position_model = Model(inputs=input, outputs=output)
        return position_model


    def build_CNN_Transformer_model(self,input_shape, kernel_size=3, embedding_dim=32, ff_dim=64, num_layer=1):
        input = Input(shape=(input_shape,1))
        x = Conv1D(64,kernel_size, activation='relu')(input)#(x)
        x = Conv1D(32,kernel_size, activation='relu')(x)

        x = WiFiRSSEmbedding(maxlen=input_shape, embed_dim=64)(x)
    
        # Custom transformer encoder
        transformer_encoder = TransformerEncoder(
                            num_layers=num_layer,  # Number of transformer layers
                            embed_dim=embedding_dim,
                            num_heads=4,
                            ff_dim=ff_dim,
                            rate=0.2
        )
        x = transformer_encoder(x)


        # Attention-based pooling instead of global average
        x = self._attention_pooling(x)
        
        #x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output = layers.Dense(2, activation='linear')(x)
        model = keras.Model(inputs=input, outputs=output)
        return model

    def build_AE_CNN_Trans_model(self, input_shape, kernel_size=3, embedding_dim=32, ff_dim=64, num_layer=1):
        
        if (not Train_AE) & (
        not os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '_' + self.save_name+ '.keras')))):
            self.position_base_model = self.bottleneck_model_256_128
        else:
            self.position_base_model = load_model( 
                os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '_' + self.save_name+ '.keras'))) #('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5'))) #
            
        for layer in self.position_base_model.layers:
            layer.trainable = True

        input = Reshape((self.position_base_model.output_shape[1], 1))(self.position_base_model.output)
                 

        x = Conv1D(64,3, activation='relu')(input)#(x)
        x = Conv1D(32,3, activation='relu')(x)
        
        x = WiFiRSSEmbedding(maxlen=input_shape, embed_dim=32)(x)
    
        # Custom transformer encoder
        transformer_encoder = TransformerEncoder(
                            num_layers=num_layer,  # Number of transformer layers
                            embed_dim=embedding_dim,
                            num_heads=4,
                            ff_dim=ff_dim,
                            rate=0.2
        )
        x = transformer_encoder(x)
        #'''

        #x = layers.GlobalAveragePooling1D()(x)

        # Attention-based pooling instead of global average
        x = self._attention_pooling(x)
        
        #x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output = layers.Dense(2, activation='linear')(x)
        self.position_model = Model(inputs=self.position_base_model.input, outputs=output)


    def build_AE_Trans_model(self, input_shape, embedding_dim=32, ff_dim=64, num_layer=1):
        if (not Train_AE) & (
        not os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '_' + self.save_name+ '.keras')))):
            self.position_base_model = self.bottleneck_model_256_128
        else:
            self.position_base_model = load_model( 
                os.path.join(AE_model_dir, ('AE_bottleneck_' + self.AE_floor_bottleneck + '_' + self.save_name+ '.keras'))) #('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5'))) #
            
        for layer in self.position_base_model.layers:
            layer.trainable = True

        input = Reshape((self.position_base_model.output_shape[1], 1))(self.position_base_model.output)        
        
        x = WiFiRSSEmbedding(maxlen=input_shape, embed_dim=32)(input)
    
        # Custom transformer encoder
        transformer_encoder = TransformerEncoder(
                            num_layers=num_layer,  # Number of transformer layers
                            embed_dim=embedding_dim,
                            num_heads=4,
                            ff_dim=ff_dim,
                            rate=0.2
        )
        x = transformer_encoder(x)
        #'''

        #x = layers.GlobalAveragePooling1D()(x)

        # Attention-based pooling instead of global average
        x = self._attention_pooling(x)
        
        #x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output = layers.Dense(2, activation='linear')(x)
        self.position_model = Model(inputs=self.position_base_model.input, outputs=output)

    def _attention_pooling(self, x):
        """Alternative pooling using attention mechanism"""
        # Compute attention weights for each access point
        attention_scores = layers.Dense(1, activation='tanh', name='attention_scores')(x)
        attention_weights = layers.Softmax(axis=1, name='attention_weights')(attention_scores)
        
        # Weighted sum of all access point representations
        pooled = keras.ops.sum(x * attention_weights, axis=1)
        return pooled

    def build_transformer_model(self, input_shape, embedding_dim=32, ff_dim=64, num_layer=1):
        inputs = layers.Input(shape=(input_shape,))
        
        # # Reshape input to 3D first: (batch, sequence, 1)
        x = layers.Reshape((input_shape, 1))(inputs)

        x = WiFiRSSEmbedding(maxlen=input_shape, embed_dim=32)(x)
    
        # Custom transformer encoder
        transformer_encoder = TransformerEncoder(
                            num_layers=num_layer,  # Number of transformer layers
                            embed_dim=embedding_dim,
                            num_heads=4,
                            ff_dim=ff_dim,
                            rate=0.2
        )
        x = transformer_encoder(x)
        x = self._attention_pooling(x)
        
        #x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output = layers.Dense(2, activation='linear')(x)
    
        model = keras.Model(inputs=inputs, outputs=output)
        return model

    def build_wifi_transformer_simple(self, num_access_points, embed_dim=128):
        """Simplified version using your custom components"""
        
        inputs = layers.Input(shape=(num_access_points,))
        
        # Reshape and embed
        x = layers.Reshape((num_access_points, 1))(x)
        x = WiFiRSSEmbedding(maxlen=num_access_points, embed_dim=embed_dim)(x)
        
        # Single transformer encoder
        x = TransformerEncoder(
            num_layers=3,
            embed_dim=embed_dim,
            num_heads=8,
            ff_dim=embed_dim * 4,
            rate=0.1
        )(x)
        
        # Pool and predict
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='gelu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(2, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_AE_MLP_model(self, hd_layers=[128, 64]):
        
        if (not Train_AE) & (
        not os.path.isfile(os.path.join(AE_model_dir, ('AE_bottleneck_'+self.AE_floor_bottleneck+'_'+ self.save_name+'.keras')))):
            self.position_base_model = self.bottleneck_model_256_128
        
        else:
            self.position_base_model = load_model( 
                os.path.join(AE_model_dir, ('AE_bottleneck_'+self.AE_floor_bottleneck+'_'+ self.save_name+'.keras'))) #('AE_bottleneck_' + self.AE_floor_bottleneck + '.h5'))) #
            
        for layer in self.position_base_model.layers:
            layer.trainable = True

        x = layers.Dense(hd_layers[0], activation='relu')(self.position_base_model.output)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(hd_layers[1], activation='relu')(x)

        output = layers.Dense(2, activation='linear')(x)
        self.position_model = Model(inputs=self.position_base_model.input, outputs=output)

    def build_encoder_MLP_model(self, input_shape):

        inputs = layers.Input(shape=(input_shape,))

        # Random encoder
        x = Dense(256, activation='relu')(inputs)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        features = Dense(64, activation='relu')(x)

        # MLP for position prediction
        x = layers.Dense(128, activation='relu')(features)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(2, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def fit_CNN(self,x, y,valid_x,valid_y):
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        self._preprocess(x, y,valid_x,valid_y)
        
        model=self.build_CNN_model(x.shape[1])

        self.position = np.hstack([self.longitude_normalize_y, self.latitude_normalize_y])
        self.position_valid = np.hstack([self.longitude_normalize_valid_y, self.latitude_normalize_valid_y])

        model.compile(
            #loss='mean_squared_error',
            loss=self.loss,
            optimizer=self.opt,
            #optimizer=adamm,
            metrics=['root_mean_squared_error']
        )
        h_floor = model.fit(self.normalize_x,
                            self.position,
                            epochs=self.epoch_position,
                            batch_size=64,
                            # shuffle=True,
                            validation_data=(
                                self.normalize_valid_x, self.position_valid),
                            # validation_split=0.2,
                            callbacks=[save_best,early_stopping]
                            )

        #model.save(self.model_save_path)
        return (h_floor)
    
    def build_encoder_MLP_model(self, input_shape):

        inputs = layers.Input(shape=(input_shape,))

        # Random encoder
        x = Dense(256, activation='relu')(inputs)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        features = Dense(64, activation='relu')(x)

        # MLP for position prediction
        x = layers.Dense(128, activation='relu')(features)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(2, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def fit_CNNTrans(self,x, y,valid_x,valid_y):
        
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        self._preprocess(x, y,valid_x,valid_y)
        #adamm = Adamax(learning_rate=self.adam_lr)
        
        model=self.build_CNN_Transformer_model(x.shape[1])

        self.position = np.hstack([self.longitude_normalize_y, self.latitude_normalize_y])
        self.position_valid = np.hstack([self.longitude_normalize_valid_y, self.latitude_normalize_valid_y])

        model.compile(
            #loss='mean_squared_error',
            loss=self.loss,
            optimizer=self.opt,
            #optimizer=adamm,
            metrics=['root_mean_squared_error']
        )

        
        h_floor = model.fit(self.normalize_x,
                            self.position,
                            epochs=self.epoch_position,
                            batch_size=64,
                            # shuffle=True,
                            validation_data=(
                                self.normalize_valid_x, self.position_valid),
                            # validation_split=0.2,
                            callbacks=[save_best,early_stopping]
                            )


        model.save(self.model_save_path)
        return (h_floor)

    def fit_transformer(self,x, y,valid_x,valid_y):
        
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        self._preprocess(x, y,valid_x,valid_y)
        
        model=self.build_transformer_model(x.shape[1])

        self.position = np.hstack([self.longitude_normalize_y, self.latitude_normalize_y])
        self.position_valid = np.hstack([self.longitude_normalize_valid_y, self.latitude_normalize_valid_y])

        model.summary()
        model.compile(
            #loss='mean_squared_error',
            loss=self.loss,
            optimizer=self.opt,
            metrics=['root_mean_squared_error']
        )

        
        h_floor = model.fit(self.normalize_x,
                            self.position,
                            epochs=self.epoch_position,
                            batch_size=64,
                            validation_data=(
                                self.normalize_valid_x, self.position_valid),
                            callbacks=[save_best,early_stopping]
                            )


        #model.save(self.model_save_path)
        return (h_floor)
    
    def fit_AECNNTrans(self, x, y, valid_x, valid_y):
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        early_stopping_ae = EarlyStopping(monitor='val_loss', patience=self.patience)
        # early_stopping=myEarlyStopping(monitor='val_acc', patience=self.patience,baseline=0.975)
        # Data pre-processing
        self._preprocess(x, y, valid_x, valid_y)
        h_pos = None
        opt=Adam(learning_rate=0.001)
#############fit_AE
        if Train_AE:
            self.fnBuildAEModel()
            self.encoder_model=self.encoder_model_256_128
            self.bottleneck_model=self.bottleneck_model_256_128
            self.encoder_model.compile(
                loss=self.loss,
                optimizer = opt
                #optimizer=self.opt, 

            )
            h_AE=self.encoder_model.fit(
                self.normalize_x,
                self.normalize_x,validation_data=(self.normalize_valid_x,self.normalize_valid_x), 
                epochs=self.epoch_AE,batch_size=64,
                callbacks=[early_stopping_ae]) #, save_best])
            self.bottleneck_model.save(os.path.join(AE_model_dir,('AE_bottleneck_'+self.AE_floor_bottleneck+'_'+ self.save_name+'.keras')))
            self.encoder_model.save('AE_model/AE_'+self.AE_floor_bottleneck + '_'+ self.save_name + '.keras')

#################fit_Location
        opt=Adamax(learning_rate=0.005)
        
        self.build_AE_CNN_Trans_model(x.shape[1])
        self.position=np.hstack([self.longitude_normalize_y,self.latitude_normalize_y])
        self.position_valid=np.hstack([self.longitude_normalize_valid_y,self.latitude_normalize_valid_y])
        self.position_model.compile(
            loss=self.loss,
            #optimizer = opt,
            optimizer=self.opt,
            #metrics=[keras.metrics.RootMeanSquaredError()],
            
        )
        self.position_model.summary()
        h_floor = self.position_model.fit(self.normalize_x,
                        self.position,
                        epochs=self.epoch_position,
                        batch_size=64,
                        validation_data=(
                        self.normalize_valid_x, self.position_valid),
                        # validation_split=0.2,
                        callbacks=[early_stopping, save_best]
                        )
        #self.position_model.save(self.model_save_path)
        return h_floor

    def fit_AETrans(self, x, y, valid_x, valid_y, embedding_dim=64, ff_dim=128, num_layer=1):
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        early_stopping_ae = EarlyStopping(monitor='val_loss', patience=self.patience)
        # early_stopping=myEarlyStopping(monitor='val_acc', patience=self.patience,baseline=0.975)
        # Data pre-processing
        self._preprocess(x, y, valid_x, valid_y)
        h_pos = None
        opt=Adam(learning_rate=0.001)
#############fit_AE
        if Train_AE:
            self.fnBuildAEModel()
            self.encoder_model=self.encoder_model_256_128
            self.bottleneck_model=self.bottleneck_model_256_128
            self.encoder_model.compile(
                loss=self.loss,
                optimizer = opt
                #optimizer=self.opt, 

            )
            h_AE=self.encoder_model.fit(
                self.normalize_x,
                self.normalize_x,validation_data=(self.normalize_valid_x,self.normalize_valid_x), 
                epochs=self.epoch_AE,batch_size=64,
                callbacks=[early_stopping_ae]) #, save_best])
            self.bottleneck_model.save(os.path.join(AE_model_dir,('AE_bottleneck_'+self.AE_floor_bottleneck+'_'+ self.save_name+'.keras')))
            self.encoder_model.save('AE_model/AE_'+self.AE_floor_bottleneck + '_'+ self.save_name + '.keras')

#################fit_Location
        opt=Adamax(learning_rate=0.001)
        
        self.build_AE_Trans_model(x.shape[1], num_layer = num_layer, embedding_dim=embedding_dim, ff_dim=ff_dim)
        self.position=np.hstack([self.longitude_normalize_y,self.latitude_normalize_y])
        self.position_valid=np.hstack([self.longitude_normalize_valid_y,self.latitude_normalize_valid_y])
        
        self.position_model.compile(
            loss=self.loss,
            optimizer=self.opt,
            #metrics=[keras.metrics.RootMeanSquaredError()],
            
        )

        self.position_model.summary()

        h_floor = self.position_model.fit(self.normalize_x,
                        self.position,
                        epochs=self.epoch_position,
                        batch_size=64,
                        validation_data=(
                        self.normalize_valid_x, self.position_valid),
                        # validation_split=0.2,
                        callbacks=[early_stopping, save_best]
                        )
            #self.position_model.save(self.model_save_path)
        return h_floor

    def fit_MLP(self, x, y, valid_x, valid_y, test_x, hd_layers=[128, 64]):
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        
        # early_stopping=myEarlyStopping(monitor='val_acc', patience=self.patience,baseline=0.975)
        # Data pre-processing
        self._preprocess(x, y, valid_x, valid_y, test_x)
        h_pos = None
        opt=Adam(learning_rate=0.005)

        x = layers.Dense(hd_layers[0], activation='relu')(self.input)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(hd_layers[1], activation='relu')(x)

        position_output = Dense(2, activation='linear', name='position_coordinates')(x)

        model = Model(inputs=self.input, outputs=position_output)

        model.compile(
            loss=self.loss, 
            optimizer=self.opt
        )

        self.position = np.hstack([self.longitude_normalize_y, self.latitude_normalize_y])
        self.position_valid = np.hstack([self.longitude_normalize_valid_y, self.latitude_normalize_valid_y])

        h = model.fit(self.normalize_x,
                        self.position,
                        epochs=self.epoch_position,
                        batch_size=64,
                        validation_data=(
                        self.normalize_valid_x, self.position_valid),
                        # validation_split=0.2,
                        callbacks=[early_stopping, save_best]
                        )
        #model.save(self.model_save_path)
        return h
    
    def fit_encoder_MLP(self, x, y, valid_x, valid_y):
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        
        # early_stopping=myEarlyStopping(monitor='val_acc', patience=self.patience,baseline=0.975)
        # Data pre-processing
        self._preprocess(x, y, valid_x, valid_y)
        h_pos = None
        opt=Adam(learning_rate=0.005)

        model = self.build_encoder_MLP_model(x.shape[1])

        model.compile(
            loss=self.loss, 
            optimizer=self.opt
        )

        self.position = np.hstack([self.longitude_normalize_y, self.latitude_normalize_y])
        self.position_valid = np.hstack([self.longitude_normalize_valid_y, self.latitude_normalize_valid_y])

        h = model.fit(self.normalize_x,
                        self.position,
                        epochs=self.epoch_position,
                        batch_size=64,
                        validation_data=(
                        self.normalize_valid_x, self.position_valid),
                        # validation_split=0.2,
                        callbacks=[early_stopping, save_best]
                        )
        #model.save(self.model_save_path)
        return h
    
    def fit_encoder_MLP(self, x, y, valid_x, valid_y):
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        
        # early_stopping=myEarlyStopping(monitor='val_acc', patience=self.patience,baseline=0.975)
        # Data pre-processing
        self._preprocess(x, y, valid_x, valid_y)
        h_pos = None
        opt=Adam(learning_rate=0.005)

        model = self.build_encoder_MLP_model(x.shape[1])

        model.compile(
            loss=self.loss, 
            optimizer=self.opt
        )

        self.position = np.hstack([self.longitude_normalize_y, self.latitude_normalize_y])
        self.position_valid = np.hstack([self.longitude_normalize_valid_y, self.latitude_normalize_valid_y])

        h = model.fit(self.normalize_x,
                        self.position,
                        epochs=self.epoch_position,
                        batch_size=64,
                        validation_data=(
                        self.normalize_valid_x, self.position_valid),
                        # validation_split=0.2,
                        callbacks=[early_stopping, save_best]
                        )
        #model.save(self.model_save_path)
        return h

    def fit_AE_MLP(self, x, y, valid_x, valid_y, hd_layers=[128, 64]):
        save_best = keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, monitor='val_loss', save_best_only=True)
        early_stopping_ae = EarlyStopping(monitor='val_loss', patience=10)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        # early_stopping=myEarlyStopping(monitor='val_acc', patience=self.patience,baseline=0.975)
        # Data pre-processing
        self._preprocess(x, y, valid_x, valid_y)
        h_pos = None
        opt=Adam(learning_rate=0.005)
#############fit_AE
        if Train_AE:
            self.fnBuildAEModel()
            self.encoder_model=self.encoder_model_256_128
            self.bottleneck_model=self.bottleneck_model_256_128
            self.encoder_model.compile(
                loss=self.loss,
                optimizer = opt
                #optimizer=self.opt, 

            )
            h_AE=self.encoder_model.fit(
                self.normalize_x,
                self.normalize_x,validation_data=(self.normalize_valid_x,self.normalize_valid_x), 
                epochs=self.epoch_AE,batch_size=64,
                callbacks=[early_stopping_ae])
            self.bottleneck_model.save(os.path.join(AE_model_dir,('AE_bottleneck_'+self.AE_floor_bottleneck+'_'+ self.save_name+'.keras')))
            self.encoder_model.save('AE_model/AE_'+self.AE_floor_bottleneck + '_'+ self.save_name + '.keras')

#################fit_Location
        #opt=Adamax(learning_rate=0.005)
        self.build_AE_MLP_model(hd_layers=hd_layers)
        self.position=np.hstack([self.longitude_normalize_y,self.latitude_normalize_y])
        self.position_valid=np.hstack([self.longitude_normalize_valid_y,self.latitude_normalize_valid_y])
        self.position_model.compile(
            loss=self.loss,
            optimizer=self.opt,
            #metrics=[keras.metrics.RootMeanSquaredError()],
            
        )
        self.position_model.summary()
        h_floor = self.position_model.fit(self.normalize_x,
                        self.position,
                        epochs=self.epoch_position,
                        batch_size=64,
                        validation_data=(self.normalize_valid_x, self.position_valid),
                        # validation_split=0.2,
                        callbacks=[early_stopping, save_best]
                        )
        return h_floor

    
    def save_to_excel_with_sheet(self,file_name, sheet_name, data):
        # Create a DataFrame from the provided data
        df = pd.DataFrame(data)

        # Specify the Excel writer with openpyxl engine
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='a' if os.path.exists(file_name) else 'w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Data saved to sheet '{sheet_name}' in file '{file_name}'.")

    def error(self, x, y):
        
        start_t = time.time()
        _y = self.predict(x)
        end=time.time()
        predict_time=end-start_t
        
        predict_long=np.reshape(_y[0],(1,len(_y[0])))
        predict_lati=np.reshape(_y[1],(1,len(_y[1])))

        predict_location = np.column_stack((_y[0], _y[1]))

        longitude = np.array(y.iloc[:, 0].values)
        latitude = np.array(y.iloc[:, 1].values)

        longitude_error = np.mean(np.sqrt(np.square(predict_long - longitude)))#y.iloc[:, 0].values)))
        latitude_error = np.mean(np.sqrt(np.square(predict_lati - latitude)))#y.iloc[:, 1].values)))

        #mean_error_all = np.sqrt(np.square(predict_long - y.iloc[:, 0].values)+np.square(predict_lati - y.iloc[:, 1].values))
        euclidean_errors = np.sqrt(np.square(predict_long - longitude) + np.square(predict_lati - latitude))
        
        # rmse = np.sqrt(np.mean(np.square(predict_long - longitude) + np.square(predict_lati - latitude)))
        mae_all = np.abs(predict_long - longitude) + np.abs(predict_lati - latitude)

        euclidean_error=np.mean(euclidean_errors)
        mse_manual = np.mean((predict_long - longitude) ** 2 + (predict_lati - latitude) ** 2)
        rmse_manual = np.sqrt(mse_manual)
        mae_manual = np.mean(mae_all)

        print(f'Mean Euclidean Error: {euclidean_error:.2f}')
        print(f'Root Mean Squared Error: {rmse_manual:.2f}')   
        print(f'Mean Absolute Error: {mae_manual:.2f}')

        data_to_save = {
            'Longitude': longitude.flatten(),
            'Latitude': latitude.flatten(),
        'Predict Longitude': predict_long.flatten(),
        'Predict Latitude': predict_lati.flatten(),
        'Mean error': euclidean_errors.flatten(), #np.full((predict_long.shape[1],), mean_error)
        'MAE': mae_all.flatten()
        }

        filename = result_dir + self.save_name + '.csv'
        # Create a DataFrame
        df = pd.DataFrame(data_to_save)

        # Save DataFrame to csv
        df.to_csv(filename, index=False)

        #Save data to excel
        #save_to_excel_with_sheet(filename, sheet_name, data_to_save)

        return longitude_error, latitude_error, euclidean_error, rmse_manual, mae_manual, predict_time




