# %%
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
#generate new dataframes for each ticker_symbol]
metric_labels=['Testing-MSE','Validation-MSE','testing-MAE','validation-MAE','testing-SMAPE','validation-SMAPE','testing-Forecast Bias','validation-Forecast Bias']
metrics_df=pd.DataFrame()
metrics_df['Metrics']=metric_labels
#save dataframe as csv
metrics_df.to_csv('/home/j/usfq/tesis/StockPredictionModels/Results/gru_metrics.csv',index=False)
metrics=[]
df_dict={}
for key in df['ticker_symbol'].unique():
    df_dict[key]=df[df['ticker_symbol']==key]
    df_dict[key]=df_dict[key].drop(columns=['ticker_symbol'])
    df_dict[key]=df_dict[key].sort_values(by=['Date']).reset_index(drop=True)
    #df_dict[key]=df_dict[key].drop(columns=['Date'])
mse_t_p=[]
mae_t_p=[]
smape_t_p=[]
forecast_bias_t_p=[]
mse_v_p=[]
mae_v_p=[]
smape_v_p=[]
forecast_bias_v_p=[]

overall_mse_train=[]
overall_mse_val=[]
overall_std_train=[]
overall_std_val=[]

for ticker_symbol in df_dict.keys():
# %%
    ticker=ticker_symbol
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

    # %%

    def build_model(hp):
        hp_units=hp.Int('units',min_value=2,max_value=240,step=2)
        model=Sequential()
        model.add(GRU(hp_units,activation='relu',input_shape=(X.shape[1],X.shape[2]),return_sequences=False))
        model.add(Dense(y.shape[1]))
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),loss='mse')

        return model

    early_stop=EarlyStopping(monitor='val_loss',patience=20)

    # %%
    tuner = kt.GridSearch(build_model,
                        objective='val_loss',
                        directory='/home/j/usfq/tesis/StockPredictionModels/Models/GRU/Tuning',
                        project_name='gru_tuning',
                        )

    # %%
    tuner.search(X, y, epochs=1000, validation_split=0.2, callbacks=[early_stop])

    # %%
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    # %%
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))
    def forecast_bias(y_true, y_pred):
        return 100 * (np.mean(y_pred) - np.mean(y_true)) / np.mean(y_true)
    def mse(y_true, y_pred):
        return np.mean(np.square(y_pred - y_true))

    # %%
    dir='/home/j/usfq/tesis/StockPredictionModels/Graphs/GRU/'+ticker
    tscv = TimeSeriesSplit(n_splits=10)
    t_mses=[]
    v_mses=[]
    t_maes=[]
    v_maes=[]
    t_smapes=[]
    v_smapes=[]
    t_forecast_biases=[]
    v_forecast_biases=[]
    counter_fold=1
    val_losses=[]
    train_losses=[]

    for train_index, test_index in tscv.split(X):
        Fold='Fold_'+str(counter_fold)
        counter_fold+=1

        print(f'{Fold} Started...')
        #create dir for each fold if it doesn't exist
        if not os.path.exists(os.path.join(dir,Fold)):
            os.makedirs(os.path.join(dir,Fold))
        X_tmp, X_test = X[train_index], X[test_index]
        y_tmp, y_test= y[train_index], y[test_index]
        val_split = int(len(train_index) * 0.8)  # Adjust the validation split percentage as needed
        X_train, x_val = X_tmp[:val_split], X_tmp[val_split:]
        y_train, y_val = y_tmp[:val_split], y_tmp[val_split:]

        model=model = tuner.hypermodel.build(best_hps)
        history=model.fit(X_train,y_train,epochs=1000,validation_data=(x_val,y_val),shuffle=False, callbacks=[early_stop],verbose=False)
        val_losses.append(history.history['val_loss'])
        train_losses.append(history.history['loss'])

        plt.figure()
        plt.plot(history.history['loss'],label='train')
        plt.plot(history.history['val_loss'],label='val')
        plt.legend()
        plt.savefig(os.path.join(dir,Fold,'loss.png'))
        plt.close()

        y_test=np.repeat(y_test,X.shape[2],axis=1)
        y_test=scaler.inverse_transform(y_test)[:,-1] 
        y_val=np.repeat(y_val,X.shape[2],axis=1)
        y_val=scaler.inverse_transform(y_val)[:,-1]

        y_val_pred=model.predict(x_val,verbose=False)
        val_pred=np.repeat(y_val_pred,X.shape[2],axis=1)
        val_pred=scaler.inverse_transform(val_pred)[:,-1]

        y_pred=model.predict(X_test,verbose=False)
        pred=np.repeat(y_pred,X_train.shape[2],axis=1)
        pred=scaler.inverse_transform(pred)[:,-1]
        
        v_time=range(X_train.shape[0],X_train.shape[0]+len(y_val))

        t_time=range(X_train.shape[0]+len(y_val),X_train.shape[0]+len(y_val)+len(y_test))

        r_time=range(X_train.shape[0],X_train.shape[0]+len(y_val)+len(y_test))


        t_mse=mse(pred,y_test)
        t_mses.append(t_mse)

        v_mse=mse(val_pred,y_val)
        v_mses.append(v_mse)
        
        t_mae=mae(pred,y_test)
        t_maes.append(t_mae)

        v_mae=mae(val_pred,y_val)
        v_maes.append(v_mae)
 
        t_smape=smape(pred,y_test)
        t_smapes.append(t_smape)
        
        v_smape=smape(val_pred,y_val)
        v_smapes.append(v_smape)
        

        t_forecast_bias=forecast_bias(pred,y_test)
        t_forecast_biases.append(t_forecast_bias)

        v_forecast_bias=forecast_bias(val_pred,y_val)
        v_forecast_biases.append(v_forecast_bias)


        mse_t_p.append(t_mse)
        mse_v_p.append(v_mse)
        mae_t_p.append(t_mae)
        mae_v_p.append(v_mae)
        smape_t_p.append(t_smape)
        smape_v_p.append(v_smape)
        forecast_bias_t_p.append(t_forecast_bias)
        forecast_bias_v_p.append(v_forecast_bias)


        real=np.concatenate((y_val,y_test))
        plt.figure(figsize=(20,10))
        plt.plot(r_time,real,label='real',color='blue')
        plt.plot(v_time,val_pred,label='val_pred',color='green')
        plt.plot(t_time,pred, label='test_pred',color='orange') 
        plt.legend()
        plt.savefig(os.path.join(dir,Fold,'real_vs_pred.png'))
        plt.close()

        print(f'{Fold} Done')

    max_train_epochs = max(len(losses) for losses in train_losses)
    max_val_epochs = max(len(losses) for losses in val_losses)


    padded_train_losses = [np.pad(losses, (0, max_train_epochs - len(losses)), mode='constant', constant_values=np.nan)
                        for losses in train_losses]

    padded_val_losses = [np.pad(losses, (0, max_val_epochs - len(losses)), mode='constant', constant_values=np.nan)
                        for losses in val_losses]


    mean_train_loss = np.nanmean(padded_train_losses, axis=0)
    std_train_loss = np.nanstd(padded_train_losses, axis=0)

    overall_mse_train.append(np.mean(mean_train_loss))
    overall_std_train.append(np.std(mean_train_loss))

    mean_val_loss = np.nanmean(padded_val_losses, axis=0)
    std_val_loss = np.nanstd(padded_val_losses, axis=0)

    overall_mse_val.append(np.mean(mean_val_loss))
    overall_std_val.append(np.std(mean_val_loss))

    plt.figure(figsize=(20,20))
    plt.errorbar(range(1, max_train_epochs + 1), mean_train_loss, yerr=std_train_loss, label='Mean Training Loss', marker='o')
    plt.errorbar(range(1, max_val_epochs + 1), mean_val_loss, yerr=std_val_loss, label='Mean Validation Loss', marker='o')


    plt.xlabel('Epochs')
    plt.ylabel('Loss')  
    plt.title('Average Training and Validation Loss Across Folds')


    plt.legend()
    plt.savefig(os.path.join(dir,'mean_loss.png'))
    plt.close()
    

    # %%
    Taverage_mse=np.mean(t_mses)
    metrics.append(Taverage_mse)
    Vaverage_mse=np.mean(v_mses)
    metrics.append(Vaverage_mse)
    Taverage_mae=np.mean(t_maes)
    metrics.append(Taverage_mae)
    Vaverage_mae=np.mean(v_maes)
    metrics.append(Vaverage_mae)
    Taverage_smape=np.mean(t_smapes)
    metrics.append(Taverage_smape)
    Vaverage_smape=np.mean(v_smapes)
    metrics.append(Vaverage_smape)
    Taverage_forecast_bias=np.mean(t_forecast_biases)
    metrics.append(Taverage_forecast_bias)
    Vaverage_forecast_bias=np.mean(v_forecast_biases)
    metrics.append(Vaverage_forecast_bias)
    metrics_df[ticker_symbol]=metrics
    metrics=[]
    #save dataframe as csv
    metrics_df.to_csv('/home/j/usfq/tesis/StockPredictionModels/Results/gru_metrics.csv',index=False)
    print(f'{ticker} done')


#define dataframe for metrics
df_ps=pd.DataFrame()
df_ps['MSE']=mse_t_p
df_ps['MAE']=mae_t_p
df_ps['SMAPE']=smape_t_p
df_ps['Forecast Bias']=forecast_bias_t_p
df_ps.to_csv('/home/j/usfq/tesis/StockPredictionModels/Results/gru_hypothesis.csv',index=False)

df_loss=pd.DataFrame()
df_loss['Training Loss']=overall_mse_train
df_loss['Training Std']=overall_std_train
df_loss['Validation Loss']=overall_mse_val
df_loss['Validation Std']=overall_std_val
df_loss.to_csv('/home/j/usfq/tesis/StockPredictionModels/Results/gru_loss.csv',index=False)


 


