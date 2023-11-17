# %%
import seaborn as sns
import pylab as rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib 
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Bidirectional, GlobalAveragePooling1D,GRU,Flatten
from sklearn.preprocessing import MinMaxScaler
from  sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import EarlyStopping
from tcn import TCN
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
import math
import keras_tuner as kt

# %% [markdown]
# 

# %%
df=pd.read_csv('/home/j/usfq/tesis/StockPredictionModels/Data/Complete.csv')
df

# %%
#turn date into unix time
#df['Date'] = pd.to_datetime(df['Date'])
#df['Date'] = df['Date'].apply(lambda x: x.timestamp())
#df

# %%
#generate new dataframes for each ticker_symbol
df_dict={}
for key in df['ticker_symbol'].unique():
    df_dict[key]=df[df['ticker_symbol']==key]
    df_dict[key]=df_dict[key].drop(columns=['ticker_symbol'])
    df_dict[key]=df_dict[key].sort_values(by=['Date']).reset_index(drop=True)
    #df_dict[key]=df_dict[key].drop(columns=['Date'])
    print(key,df_dict[key].shape)
    print(df_dict[key].head(-1))

# %%
ticker='TSLA'

# %%
df=df_dict[ticker].copy()
df.head()

# %%
#putting the close column on the last position
df=df[['Date', 'p_sentiment', 'Open', 'High', 'Low',
       'Volume', 'unrate', 'psr', 'm2', 'dspic', 'pce', 'reer', 'ir', 'ffer',
       'tcs', 'indpro', 'ccpi', 'Close']]

# %%
dates = pd.to_datetime(df['Date'])

# %%
cols=list(df)[1:]
cols

# %%
df_for_training = df[cols].astype(float)

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_for_training)

# %%
print(scaled_data)

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
print(X.shape)
print(y.shape)

# %%
import math as m
def n_layers(ks):
    n=m.ceil(m.log2((((n_past-1)*(2-1))/(ks-1))+1))
    return n

# %%
def build_model(hp):
    hp_ks=hp.Int('kernel_size',min_value=2,max_value=(n_past-1),step=1)
    hp_nb=hp.Int('nb_filters',min_value=2,max_value=256,step=2)
    hp_dp=hp.Float('dropout_rate',min_value=0.0,max_value=0.3,step=0.05)
    hp_gp=hp.Choice('globalpooling',[True,False])
    hp_skp=hp.Choice('skip_connections',[True,False])
    hp_acti=hp.Choice('activation',['relu','sigmoid'])
    model=Sequential()
    model.add(Bidirectional(
            TCN(
                input_shape=(X.shape[1],X.shape[2]),
                kernel_size=hp_ks,
                return_sequences=hp_gp,
                dilations=[2**i for i in range(0,n_layers(hp_ks))],
                activation=hp_acti,
                nb_filters=hp_nb,
                padding='causal',
                dropout_rate=hp_dp,
                kernel_initializer='he_normal',
                use_skip_connections=hp_skp)))
    if hp_gp:
        model.add(GlobalAveragePooling1D())
    model.add(Dense(y.shape[1]))
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),loss='mse')
    return model

early_stop=EarlyStopping(monitor='val_loss',patience=20)

# %%

tuner = kt.GridSearch(build_model,
                     objective='val_loss',
                     project_name='bi_tcn_tuning',
                     )

# %%
tuner.search(X, y, epochs=1000, validation_split=0.2, callbacks=[early_stop])

# %%
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)
