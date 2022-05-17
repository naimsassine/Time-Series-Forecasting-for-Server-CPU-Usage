# let's put everything together in this code
import numpy as np
#%matplotlib inline 
import pandas as pd
import tensorflow as tf
from IPython.display import display, Image
import cv2
from PIL import Image
from tqdm import tqdm
from keras.applications.vgg19 import VGG19
from keras.utils.vis_utils import plot_model
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from yaml import ScalarEvent

# Useful functions : 
def train_generator(dataset, n_lags=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - n_lags - 1):
        a = dataset.iloc[i:(i+n_lags)].to_numpy()
        dataX.append(a)
        dataY.append(dataset.iloc[i + n_lags].to_numpy())

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    new_dataX = np.empty([dataX.shape[0], dataX.shape[1]])
    for i in range(len(dataX)) : 
        for j in range(dataX.shape[1]) :
            new_dataX[i][j] = dataX[i][j][0]

    new_dataY = np.empty(dataY.shape[0])
    for i in range(len(dataY)) : 
        new_dataY[i] = dataY[i][0]

    return (np.array(new_dataX), np.array(new_dataY))



# Preprocess_data function :
"""
This function takes as input :
- the dataset, the csv file
- the amount of days we want to go back into to predict the future (timesteps)
- the train test split
- outputs X_train, y_train, X_test, y_test

-> in this section possibly, the statistcs that Caroline did should be implemented
"""
def pre_process(dataset_name, timesteps, train_test_split) :
    # let's load the datasets
    if dataset_name == "azure" :
        df = pd.read_csv("./Azure_Dataset/azure.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df.drop('min cpu', inplace=True, axis=1)
        df.drop('max cpu', inplace=True, axis=1)

    elif dataset_name == "bit brains" : 
        df = pd.read_csv("./BitBrains_Dataset/1.csv", ";\t")
        df['Timestamp [ms]'] = pd.to_datetime(df['Timestamp [ms]'])
        df = df.set_index('Timestamp [ms]')
        df.drop('CPU cores', inplace=True, axis=1)
        df.drop('CPU usage [%]', inplace=True, axis=1)
        df.drop('CPU capacity provisioned [MHZ]', inplace=True, axis=1)
        df.drop('Memory capacity provisioned [KB]', inplace=True, axis=1)
        df.drop('Memory usage [KB]', inplace=True, axis=1)
        df.drop('Disk read throughput [KB/s]', inplace=True, axis=1)
        df.drop('Disk write throughput [KB/s]', inplace=True, axis=1)
        df.drop('Network received throughput [KB/s]', inplace=True, axis=1)
        df.drop('Network transmitted throughput [KB/s]', inplace=True, axis=1)

    elif dataset_name ==  "KSA" :
        df = pd.read_excel("./KSA_Dataset/KSA_test.xlsx")
        df.drop('Response Time', inplace=True, axis=1)
        df.drop('Class Name', inplace=True, axis=1) 
    
    else : 
        print("Wrong dataset name, please chose from : azure, bit brains and KSA")

    # create train test split
    train_length = round(len(df)*train_test_split)
    test_length = len(df) - train_length
    train = df.iloc[0:train_length]
    test = df.iloc[train_length:]

    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std

    # let's scale the values of the dataset
    scaler = MinMaxScaler(feature_range = (0,1)) #transform features by scaling each feature to a given range
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=df.columns)
    test_scaled = pd.DataFrame(scaler.fit_transform(test), columns=df.columns)


    X_train, y_train = train_generator(train_scaled, n_lags = timesteps)
    X_test_scaled, y_test_scaled = train_generator(test_scaled, n_lags=timesteps)
    X_test, y_test = train_generator(test, n_lags=timesteps)

    return X_train, y_train, X_test, y_test, X_test_scaled, y_test_scaled, scaler


# Train function : 
"""
This function takes as input : 
- the models that we want to train (3 options)
- the parameters of the training
- ouputs training results and graphs

"""
def train_models(list_of_models=[], print_summary=False, X_train=[], 
                    y_train=[], loss_function="mean_absolute_error", 
                    optimizer=tf.keras.optimizers.Adam(), epochs=200, validation_split=0.25,
                    batch_size=256, verbose=1, save__model_path="") :

    for model in list_of_models :
        if model not in ["neural network", "GRU", "LSTM"] :
            print("Model should be between the following : neural network, GRU, LSTM")   
            return None
        if model == "neural network" :
            # build a NN and train it
            model_NN = tf.keras.Sequential()
            model_NN.add(tf.keras.layers.Dense(32, input_dim=X_train.shape[1], activation='relu'))
            model_NN.add(tf.keras.layers.Dense(32, activation='relu'))
            model_NN.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            
            if print_summary == True : 
                print(model_NN.summary())
            
            model_NN.compile(loss=loss_function, optimizer=optimizer)
            es = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
            lr_red = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=0.0000001,)


            callbacks = [es, lr_red]
            history = model_NN.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=batch_size, verbose=verbose,
                                shuffle=False,
                                callbacks = callbacks)
            
            if save__model_path != "" :
                model_NN.save(save__model_path)

            # let's print some graphs

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            # why not also draw out the learning rate
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.ylabel('LR')
            plt.xlabel('epoch')
            plt.show()

            return model_NN

        if model == "GRU" : 
            # build an GRU and train it
            model_GRU = tf.keras.models.Sequential()
            model_GRU.add(tf.keras.layers.GRU(512,input_shape=(X_train.shape[1], 1),return_sequences=True))
            model_GRU.add(tf.keras.layers.GRU(512, return_sequences=False))
            model_GRU.add(tf.keras.layers.Dense(1))
            model_GRU.summary()
            
            if print_summary == True : 
                print(model_GRU.summary())
            
            model_GRU.compile(loss=loss_function, optimizer=optimizer)
            es = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
            lr_red = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=0.0000001,)


            callbacks = [es, lr_red]
            history = model_GRU.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=batch_size, verbose=verbose,
                                shuffle=False,
                                callbacks = callbacks)
            
            if save__model_path != "" :
                model_GRU.save(save__model_path)

            # let's print some graphs

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            # why not also draw out the learning rate
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.ylabel('LR')
            plt.xlabel('epoch')
            plt.show()

            return model_GRU

        if model == "LSTM" :
            # build an LSTM and train it
            model_lstm = tf.keras.models.Sequential()
            model_lstm.add(tf.keras.layers.LSTM(512,input_shape=(X_train.shape[1], 1),return_sequences=True))
            model_lstm.add(tf.keras.layers.LSTM(512, return_sequences=False))
            model_lstm.add(tf.keras.layers.Dense(1))
            model_lstm.summary()
            
            if print_summary == True : 
                print(model_lstm.summary())
            
            model_lstm.compile(loss=loss_function, optimizer=optimizer)
            es = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
            lr_red = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=0.0000001,)


            callbacks = [es, lr_red]
            history = model_lstm.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, batch_size=batch_size, verbose=verbose,
                                shuffle=False,
                                callbacks = callbacks)
            
            if save__model_path != "" :
                model_lstm.save(save__model_path)

            # let's print some graphs

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            # why not also draw out the learning rate
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.ylabel('LR')
            plt.xlabel('epoch')
            plt.show()

            return model_lstm



# Test function : 
"""
This function takes as input : 
- the model
- the test set
- outputs the test scores and the graphs we want
"""
def test_models(model, y_test, X_test_scaled, scaler):
    preds = model.predict(X_test_scaled)
    preds = scaler.inverse_transform(preds)
    plt.rcParams["figure.figsize"] = (32,12)
    TestY = pd.DataFrame(y_test, columns=['avg_cpu'])
    PredY = pd.DataFrame(preds, columns=['avg_cpu'])

    plot_avg = plt.figure(1)
    plt.plot(TestY['avg_cpu'])
    plt.plot(PredY['avg_cpu'])
    plt.show()

    testScore_1 = math.sqrt(mean_squared_error(y_test[:], preds[:]))
    print('Test Score: %.2f RMSE' % (testScore_1))

    testScore_2 = math.sqrt(mean_absolute_error(y_test[:], preds[:]))
    print('Test Score: %f MAE' % (testScore_2))

    testScore_3 = np.mean(np.abs(preds - y_test)/np.abs(y_test)*100)
    print('Test Score: %f MAPE' % (testScore_3))



if __name__ == "__main__":
    # let's load the data
    X_train, y_train, X_test, y_test, X_test_scaled, y_test_scaled, scaler = pre_process("azure", 10, 0.8)

    # let's train the models
    model_NN = train_models(["neural network"], False, X_train, y_train, "mean_absolute_error", 
                                    tf.keras.optimizers.Adam(), 400, 0.25, 256, 1, "")

    test_models(model_NN, y_test, X_test_scaled, scaler)