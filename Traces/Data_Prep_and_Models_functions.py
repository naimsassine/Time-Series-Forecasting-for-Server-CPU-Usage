# -*- coding: utf-8 -*-
"""
PARAMETERS
"""

import os
import re
import contextlib
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

#====================================================================
#                          PARAMETERS
#====================================================================

#----------- DATASET INFORMATION -------------------------
dataset_in = "azure.pckl"      # should be .csv or .pckl
dataset_out = "azure_all_features.pckl"

#----------- SAMPLING RATE and PREDICTION RANGE ----------
resample = True
sampling_rate = '5min' # choose between 5min, 10min, 15min, 30min, 1h

#----------- SCALER & TEST SIZE---------------------------
scaler = MinMaxScaler()
training_proportion = 0.8

#----------- MODEL PARAMETERS ----------------------------
features = ["cpu_max"] # list of the features to use

model_type = "LSTM" # LSTM, GRU (not implemented), ...
loss_function = "mean_absolute_error"
optimizer = tf.keras.optimizers.Adam()

n_rows_input = 100 # lag: number of observations necessary to predict the next value
batch_size = 256 # number of rows to predict at each epoch
n_epoch = 10 # nb of times we go through the whole training dataset
verbose = 2



#====================================================================
#                      FUNCTIONS
#====================================================================

def load_data(dataset_name, resample, sampling_rate) :
    print(f"\nImport data from {data_in_path}")

    if ".csv" in dataset_name :
        df = pd.read_csv(os.path.join(data_folder, data_in_path))
    elif ".pckl" in dataset_name :
        df = pd.read_pickle(os.path.join(data_folder, data_in_path))
    else :
        print("!!! FILE IN WRONG FORMAT !!!")
        
    cols = list(df.columns)

    if 'timestamp' in cols :
        print("> convert the timestamp column to index")
        df['timestamp'] = pd.to_datetime(df['timestamp']) # depending on dataset format --> to decide
        df = df.sort_values(axis = 0, by = "timestamp").set_index('timestamp')

    cols = list(df.columns)

    print("\n  IMPORTED DATASET")
    print(40 * "=")
    print(f"  {len(df)} observations")
    print(f"  {len(df.columns)} columnns:")
    print(f"  {cols}\n")
    print(df.info())
    print(40 * "=")

    if resample == True:
        print("\n> resample dataset")
        df = df.groupby([pd.Grouper(level = 'timestamp', freq = sampling_rate)]).max()
        df.index.freq = sampling_rate
        
        print(f"\nsampling rate: {sampling_rate}")
        print(f"==> {len(df)} observations\n")

    return df

def add_perc_change_features(col) :
    print("> percent changes")
    df[col + "_perc_change_1"] = df[col].pct_change() # dx/dt with dt = 1 sampling unit
    df[col + "_perc_change_2"] = df[col].pct_change(periods = 2)
    df[col + "_perc_change_4"] = df[col].pct_change(periods = 4)
    df[col + "_perc_change_acceleration"] = df["cpu_max_perc_change_1"].pct_change() # ddx/dt with dt = 1 sampling unit


def add_moving_average_features(col) :
    print("> moving averages")
    df[col + "_mov_avg_5"] = df[col].rolling(5).mean()
    df[col + "_mov_avg_10"] = df[col].rolling(10).mean()


def add_features(df, dataset_out) :
    print("\n--------- Add new features to dataset --------")
    df["weekday"] = df.index.weekday # 6 = Sunday
    one_hot_weekday = pd.get_dummies(df["weekday"], prefix = "weekday_", sparse=True)
    df = pd.concat([df, one_hot_weekday], axis = 1)

    df["hour"] = df.index.hour
    one_hot_hour = pd.get_dummies(df["hour"], prefix = "h_", sparse=True)
    df = pd.concat([df, one_hot_hour], axis = 1)

    if "cpu_max" in df.columns :
        add_perc_change_features("cpu_max")
        add_moving_average_features("cpu_max")

    data_out_path = os.path.join(data_folder, dataset_out)
    print(f"\nsave prepared dataset to {data_out_path}")
    df.to_pickle(os.path.join(data_folder, dataset_out))

    
def split_scale(training_proportion, features, scaler) :
    print("\n-------- train/test split --------")
    training_size = round(len(df) * training_proportion)
    df_subset = pd.DataFrame(df[features], columns = features)
    df_train = df_subset[:training_size]
    df_test = df_subset[training_size:]

    print(f"{len(df_train)} measures of training data ({round((len(df_train)/len(df))*100)}%) - dates from {df_train.index.min()} to {df_train.index.max()}")
    print(f"{len(df_test)} measures of testing data ({round((len(df_test)/len(df))*100)}%) - dates from {df_test.index.min()} to {df_test.index.max()}")

    print("\n-------- scale data --------")
    train_scaled = pd.DataFrame(scaler.fit_transform(df_train.values), columns = df_train.columns)
    test_scaled = pd.DataFrame(scaler.transform(df_test.values), columns = df_test.columns)

    print(train_scaled.head())
    
    print("\n-------- Build time series generator --------")
    train_tsgenerator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train_scaled,
                                      train_scaled["cpu_max"],
                                      length = n_rows_input,
                                      batch_size = batch_size)

    test_tsgenerator = tf.keras.preprocessing.sequence.TimeseriesGenerator(test_scaled,
                                      test_scaled["cpu_max"],
                                      length = n_rows_input,
                                      batch_size = batch_size)

    print(f"Generator: {len(train_tsgenerator)} times with {batch_size} observations/batch in training set of {len(df_train)} observations\n")
    print(f"Input of shape: {train_tsgenerator[0][0].shape}") # (batch_size, n_input, 1)
    print(f"output of shape {train_tsgenerator[0][1].shape}")
    
    return df_train, df_test, train_tsgenerator, test_tsgenerator
    

def build_compile_model(model_type, loss_function, optimizer) :
    n_features = train_tsgenerator[0][0].shape[2]
    print(f"\n-------- Build {model_type} model for {n_features} feature(s) --------")
    
    if model_type == "LSTM" :
        # define the keras model
        model = tf.keras.models.Sequential()
        # (6811, 100, 3)
        model.add(tf.keras.layers.LSTM(512, input_shape = (n_rows_input, n_features), return_sequences = True))
        model.add(tf.keras.layers.LSTM(512, return_sequences = False))
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(loss = loss_function, optimizer = optimizer)
        # early stopping ?
    
    elif model_type == "GRU" :
        pass

    else :
        print("\n!!! Wrong model type passed in !!!\n")

    print("\nModel summray:")
    print(model.summary())
    
    return model
    
    
def train_model(model, train_tsgenerator, n_epoch, batch_size) :
    print("\n-------- Train model --------")
    t1 = dt.now()
    print(f"{str(t1)} - start training model for {n_epoch} epochs with batch size of {batch_size} observations")
    print(f"{len(train_tsgenerator)} steps per epoch")
    history = model.fit(train_tsgenerator,
                        epochs = n_epoch,
                        batch_size = batch_size, verbose = verbose,
                        shuffle = False)

    t2 = dt.now()
    training_time = t2 - t1
    print(f"Training finished on {t2}")
    print(f"training time: {training_time}")
    
    return history, training_time

def evaluate_model(model, df_train, df_test, train_tsgenerator, test_tsgenerator) :
    print("\n-------- Model evaluation --------\n")
    trainPredict = scaler.inverse_transform(model.predict(train_tsgenerator))
    testPredict = scaler.inverse_transform(model.predict(test_tsgenerator))

    df_trainPredict = pd.DataFrame(trainPredict, index = df_train.index[n_rows_input:], columns = ["train_pred"])
    df_testPredict = pd.DataFrame(testPredict, index = df_test.index[n_rows_input:], columns = ["test_pred"])
    df_evaluation = test = pd.concat([df, df_trainPredict, df_testPredict], axis=1)

    print("\nResults on test dataset:")
    df_scores = df_evaluation[df_evaluation.test_pred.notnull()]
    RMSE = math.sqrt(mean_squared_error(df_scores["cpu_max"].values, df_scores["test_pred"].values))
    MAE = math.sqrt(mean_absolute_error(df_scores["cpu_max"].values, df_scores["test_pred"].values))
    MAPE = np.mean(np.abs(df_scores["test_pred"].values - df_scores["cpu_max"].values)/np.abs(df_scores["cpu_max"].values)*100)

    print(f"MAE:\t{MAE}")
    print(f"RMSE:\t{RMSE}")
    print(f"MAPE:\t{MAPE}")

    return df_trainPredict, df_testPredict, df_evaluation, df_scores, MAE, RMSE, MAPE


def write_results(result_file) :
    print(f"\n> writing results to {result_file}")

    with open(result_file, "w") as f:
        f.write(f"======================== DATA ===================================\n")
        f.write(f"dataset: {dataset_in}\n")
        f.write(f"{len(features)} feature(s):\t{features}\n")
        f.write(f"Sampling rate:\t{sampling_rate}\n")
        f.write(f"Training size:\t{len(df_train)}\n")
        f.write(f"Test size:\t{len(df_test)}\n")
        f.write("\n======================== MODEL ==================================\n")
        f.write(f"Model:\t{model_type}\n")
        f.write(f"Loss function:\t{history.model.loss}\n")
        f.write(f"Optimizer:\t{str_optimizer}\n\n")
        with contextlib.redirect_stdout(f):
            model.summary()
        f.write(f"\n{history.params['epochs']} epoch(s) with batch size of {batch_size} observations\n")
        f.write(f"--> {history.params['steps']} steps/epoch\n")
        f.write(f"\nTraining time total:\t{str(training_time)}\n")
        f.write(f"Training time/epoch:\t{str(training_time/history.params['epochs'])}\n")
        f.write(f"Training time/step:\t{str(training_time/ (history.params['epochs'] * history.params['steps']))}\n")
        f.write("\n======================== RESULTS ================================\n")
        f.write(f"MAE:\t{MAE}\n")
        f.write(f"RMSE:\t{RMSE}\n")
        f.write(f"MAPE:\t{MAPE}\n")
        f.close()


def plot(fig_path) :
    print(f"\n> outputting figures to {fig_path}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (16, 16))
    fig.suptitle(f"Results for {model_type} after {history.params['epochs']} epochs using {dataset_name} dataset", fontsize = 14, fontweight = 'bold')
    ax1.plot(range(1, len(history.history['loss']) +1), history.history['loss'])
    ax1.set_title(f"Training loss using {history.model.loss}")
    ax1.set_ylabel("loss")
    ax1.text(4* (history.params['epochs']/5),
             7* (max(history.history['loss'])/8),
             f"optimizer: {str_optimizer}\ntotal training time: {str(training_time)}\nbatch size: {batch_size}\n{history.params['steps']} steps/epoch")

    ax2.plot(df_scores.index[:1000], df_scores["cpu_max"][:1000], label = "real value")
    ax2.plot(df_scores.index[:1000], df_scores["test_pred"][:1000], label = "predictions")
    ax2.set_ylabel("CPU max")
    ax2.set_xlabel(f"MAE: {round(MAE, 3)}\nRMSE: {round(RMSE, 3)}\nMAPE: {round(MAPE, 3)}\n",
                   position = (0, 0),
                   horizontalalignment='left')
    ax2.legend()
    ax2.set_title(f"CPU max vs. model predictions in test dataset with sampling rate of {sampling_rate} (last 1000 observations)")
    plt.show()

    fig.savefig(fig_path)
    
#====================================================================
#                      MAIN
#====================================================================

# create directories
data_folder = os.path.abspath(os.path.join(os.path.join(os.getcwd(), ".."), "data"))
model_folder = os.path.abspath(os.path.join(os.path.join(os.getcwd(), ".."), "model"))

data_in_path = os.path.join(data_folder, dataset_in)
dataset_name = re.match("(.*)\.([a-zA-Z]*)", dataset_in).group(1)
model_name = f"{dataset_name}_{model_type}_input_{n_rows_input}_batch_s_{batch_size}_epoch_{n_epoch}.h5"
result_path = os.path.join(model_folder, model_name.replace(".h5", ".txt"))
fig_path = os.path.join(model_folder, model_name.replace(".h5", ".pdf"))


# import raw data
df = load_data(dataset_in, resample, sampling_rate)
    
# create new features (not used yet) and export dataset
add_features(df, dataset_out)

# split, scale, build time series generator and keep only features decided by user
df_train, df_test, train_tsgenerator, test_tsgenerator = split_scale(training_proportion, features, scaler)

# build model
model = build_compile_model(model_type, loss_function, optimizer)

# train model on time series generator
history, training_time = train_model(model, train_tsgenerator, n_epoch, batch_size)

# save model
model.save(os.path.join(model_folder, model_name))

# load model
#model = keras.models.load_model(os.path.join(model_folder, model_name))

# evaluate model
df_trainPredict, df_testPredict, df_evaluation, df_scores, MAE, RMSE, MAPE = evaluate_model(model, df_train, df_test, train_tsgenerator, test_tsgenerator) 

# formatting
str_optimizer = re.search('<(.*) obj.*', str(optimizer)).group(1)

# write results to text file
write_results(result_path)

# plot the results
plot(fig_path)
