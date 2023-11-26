import seaborn as sns
import pylab as rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib 
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional,GRU
from sklearn.preprocessing import MinMaxScaler
from  sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras_tuner as kt
import os
import warnings
from tcn import TCN
import math as m
project_path='/home/j/usfq/tesis/StockPredictionModels - Copy'
# %%
df=pd.read_csv(project_path+'/Data/Complete.csv')
df

df_dict={}
for key in df['ticker_symbol'].unique():
    df_dict[key]=df[df['ticker_symbol']==key]
    df_dict[key]=df_dict[key].drop(columns=['ticker_symbol'])
    df_dict[key]=df_dict[key].sort_values(by=['Date']).reset_index(drop=True)
    #df_dict[key]=df_dict[key].drop(columns=['Date'])



# %%
ticker='TSLA'
print(f'Working on {ticker}...')
# %%
df=df_dict[ticker].copy()


# %%
#putting the close column on the last position
df=df[['Date', 'p_sentiment', 'Open', 'High', 'Low',
    'Volume', 'unrate', 'psr', 'm2', 'dspic', 'pce', 'reer', 'ir', 'ffer',
    'tcs', 'indpro', 'ccpi', 'Close']]

# %%
dates = pd.to_datetime(df['Date'])

# %%
cols=list(df)[1:]


# %%
df_for_training = df[cols].astype(float)

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_for_training)

# %%
#split scaled data into training, val and testing
#train_data=scaled_data[0:1000,:]
#val_data=scaled_data[1000:1125,:]
#test_data=scaled_data[1125:,:]

# %%
n_future = 1 # Number of days we want to predict into the future
n_past = 7 # Number of past days we want to use to predict the future

# %%
X=[]
y=[]
for i in range(n_past, len(scaled_data) - n_future +1):
    X.append(scaled_data[i - n_past:i, 0:df_for_training.shape[1]])
    y.append(scaled_data[i + n_future - 1:i + n_future, len(cols)-1])

# %%
#shape of X_s and y_s
X, y = np.array(X), np.array(y)

early_stop=EarlyStopping(monitor='val_loss',patience=10)
def build_bigru(hp):
    hp_units=hp.Int('units',min_value=2,max_value=240,step=2)
    model=Sequential()
    model.add(Bidirectional(GRU(hp_units,activation='relu',input_shape=(7,17),return_sequences=False)))
    model.add(Dense(1))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),loss='mse')

    return model

def build_bilstm(hp):
    hp_units=hp.Int('units',min_value=2,max_value=240,step=2)
    model=Sequential()
    model.add(Bidirectional(LSTM(hp_units,activation='relu',input_shape=(7,17),return_sequences=False)))
    model.add(Dense(1))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),loss='mse')

    return model
def n_layers(ks):
    n=m.ceil(m.log2((((7-1)*(2-1))/(ks-1))+1))
    return n
def build_bitcn(hp):
    hp_ks=hp.Int('kernel_size',min_value=3,max_value=(7-1),step=1)
    hp_nb=hp.Int('nb_filters',min_value=16,max_value=240,step=16)
    hp_dp=hp.Choice('dropout_rate',[0.0,0.2,0.3])
    model=Sequential()
    model.add(Bidirectional(
            TCN(
                input_shape=(7,17),
                kernel_size=hp_ks,
                return_sequences=False,
                dilations=[2**i for i in range(0,n_layers(hp_ks))],
                activation='relu',
                nb_filters=hp_nb,
                padding='causal',
                dropout_rate=hp_dp,
                kernel_initializer='he_normal',
                use_skip_connections=True)))

    model.add(Dense(1))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),loss='mse')
    return model

def build_gru(hp):
    hp_units=hp.Int('units',min_value=2,max_value=240,step=2)
    model=Sequential()
    model.add(GRU(hp_units,activation='relu',input_shape=(7,17),return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),loss='mse')

    return model

def build_lstm(hp):
    hp_units=hp.Int('units',min_value=2,max_value=240,step=2)
    model=Sequential()
    model.add(LSTM(hp_units,activation='relu',input_shape=(7,17),return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),loss='mse')
    return model

def build_tcn(hp):
    hp_ks=hp.Int('kernel_size',min_value=3,max_value=(7-1),step=1)
    hp_nb=hp.Int('nb_filters',min_value=16,max_value=240,step=16)
    hp_dp=hp.Choice('dropout_rate',[0.0,0.2,0.3])
    model=Sequential()
    model.add(
            TCN(
                input_shape=(7,17),
                kernel_size=hp_ks,
                return_sequences=False,
                dilations=[2**i for i in range(0,n_layers(hp_ks))],
                activation='relu',
                nb_filters=hp_nb,
                padding='causal',
                dropout_rate=hp_dp,
                kernel_initializer='he_normal',
                use_skip_connections=True))

    model.add(Dense(1))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),loss='mse')
    return model

df_params=pd.DataFrame()
archs=['LSTM','BiLSTM','GRU','BiGRU','TCN','BiTCN']
builds=[build_lstm,build_bilstm,build_gru,build_bigru,build_tcn,build_bitcn]
df_params['Architectures']=archs
tickers=['AAPL','AMZN','GOOG','GOOGL','MSFT','TSLA']
Folders=['LSTM','GRU','TCN']
main_dir='/home/j/usfq/tesis/StockPredictionModels - Copy/Models'
for ticker in tickers:
    parms_per_tick=[]
    for i in range(len(archs)):
        for folder in Folders:
            if archs[i].__contains__(folder):
                dir=main_dir+'/'+folder+'/Tuning'
        tuning=ticker+'_'+archs[i].lower()+'_tuning'  
        print(dir)
        print(tuning)
        tuner = kt.GridSearch(builds[i],
                objective='val_loss',
                directory=dir,
                project_name=tuning,
                ) 
        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

        # Build the model with the best hyperparameters
        best_hps = best_trial.hyperparameters
        model=tuner.hypermodel.build(best_hps)
        model.fit(X,y,epochs=1,callbacks=[early_stop],verbose=0)

        print(archs[i])
        print(ticker)
        print(model.summary())
        parm=model.count_params()
        parms_per_tick.append(parm)
        del tuner
        del model
        del best_hps
        del best_trial
        del parm
    df_params[ticker]=parms_per_tick

df_params.to_csv('/home/j/usfq/tesis/StockPredictionModels - Copy/'+'Results/params.csv',index=False)


