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
from keras import layers
import os
import warnings
from tcn import TCN

# %% [markdown]
# 
arch='transformer'
project_path='/home/j/usfq/Proyecto-Integrador/StockPredictionModels'
# %%
df=pd.read_csv(project_path+'/Data/Complete.csv')
df

# %%
#turn date into unix time
#df['Date'] = pd.to_datetime(df['Date'])
#df['Date'] = df['Date'].apply(lambda x: x.timestamp())
#df
tf.keras.utils.set_random_seed(
    42
)
# %%
#generate new dataframes for each ticker_symbol]
metric_labels=['Testing-MSE','Validation-MSE','testing-MAE','validation-MAE','testing-mape','validation-mape','testing-RMSE','validation-RMSE', 'testing-MPE','validation-MPE']
metrics_df=pd.DataFrame()
metrics_df['Metrics']=metric_labels
std_metrics_df=pd.DataFrame()
std_metrics_df['Metrics']=metric_labels
#save dataframe as csv
metrics_df.to_csv(project_path+f'/Results/metrics/{arch}_metrics.csv',index=False)
std_metrics_df.to_csv(project_path+f'/Results/metrics/{arch}_std_metrics.csv',index=False)
metrics=[]
df_dict={}
for key in df['ticker_symbol'].unique():
    df_dict[key]=df[df['ticker_symbol']==key]
    df_dict[key]=df_dict[key].drop(columns=['ticker_symbol'])
    df_dict[key]=df_dict[key].sort_values(by=['Date']).reset_index(drop=True)
    #df_dict[key]=df_dict[key].drop(columns=['Date'])
mse_t_p=[]
mae_t_p=[]
mape_t_p=[]
rmse_t_p=[]
mpe_t_p=[]
mse_v_p=[]
mae_v_p=[]
mape_v_p=[]
rmse_v_p=[]
mpe_v_p=[]

overall_mse_train=[]
overall_mse_val=[]
overall_std_train=[]
overall_std_val=[]

keys=df_dict.keys()
#to list
keys_list=list(keys)

for ticker_symbol in keys_list:
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
    def transformer_encoder(inputs, head_size, num_heads,
                        dropout=0, attention_axes=None):
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            attention_axes=attention_axes
            )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        res = x + inputs
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=3,padding='causal')(res)
        return x + res

    def build_model(hp) -> tf.keras.Model:
        hp_head=hp.Int('head_size',min_value=16,max_value=128,step=16)
        hp_num_heads=hp.Int('num_heads',min_value=2,max_value=6,step=2)
        num_trans_blocks=1
        hp_mlp_units=hp.Int('mlp_unit',min_value=16,max_value=128,step=16)
        dropout=0.2
        mlp_dropout=0.2

        n_timesteps, n_features, n_outputs = 7, 17, 1 
        inputs = tf.keras.Input(shape=(n_timesteps, n_features))
        x = inputs 
        for _ in range(num_trans_blocks):
            x = transformer_encoder(x, hp_head, hp_num_heads, dropout)
        
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in [hp_mlp_units]:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)

        outputs = layers.Dense(n_outputs, activation='relu')(x)
        model=tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),loss='mse')
        return model

    early_stop=EarlyStopping(monitor='val_loss',patience=10)

    # %%
    #arch variables to all caps
    archUp=arch.upper()
    tuner = kt.GridSearch(build_model,
                        objective='val_loss',
                        directory=project_path+f'/Models/{archUp}/Tuning',
                        project_name=ticker+f'_{arch}_tuning',
                        )

    # %%
    tuner.search(X, y, epochs=1000, validation_split=0.2, callbacks=[early_stop])

    # %%
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    
    
    # %%
    def mape(y_true, y_pred):
        return 100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true)))
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))
    def mse(y_true, y_pred):
        return np.mean(np.square(y_pred - y_true))
    def rmse(y_true, y_pred):
        return np.sqrt(mse(y_true,y_pred))
    def mpe(y_true, y_pred):
        return np.mean((y_pred - y_true) / y_true) * 100
    

    # %%
    early_stop=EarlyStopping(monitor='val_loss',patience=20)
    dir=project_path+f'/Graphs/{archUp}/'+ticker
    tscv = TimeSeriesSplit(n_splits=10)
    t_mses=[]
    v_mses=[]
    t_maes=[]
    v_maes=[]
    t_mapes=[]
    v_mapes=[]
    t_rmses=[]
    v_rmses=[]
    t_mpes=[]
    v_mpes=[]
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
 
        t_mape=mape(pred,y_test)
        t_mapes.append(t_mape)
        
        v_mape=mape(val_pred,y_val)
        v_mapes.append(v_mape)
        

        t_rmse=rmse(pred,y_test)
        t_rmses.append(t_rmse)

        v_rmse=rmse(val_pred,y_val)
        v_rmses.append(v_rmse)

        t_mpe=mpe(pred,y_test)
        t_mpes.append(t_mpe)

        v_mpe=mpe(val_pred,y_val)
        v_mpes.append(v_mpe)

        mse_t_p.append(t_mse)
        mse_v_p.append(v_mse)
        mae_t_p.append(t_mae)
        mae_v_p.append(v_mae)
        mape_t_p.append(t_mape)
        mape_v_p.append(v_mape)
        rmse_t_p.append(t_rmse)
        rmse_v_p.append(v_rmse)
        mpe_t_p.append(t_mpe)
        mpe_v_p.append(v_mpe)


        real=np.concatenate((y_val,y_test))
        plt.figure(figsize=(20,10))
        plt.plot(r_time,real,label='real',color='blue')
        plt.plot(v_time,val_pred,label='val_pred',color='green')
        plt.plot(t_time,pred, label='test_pred',color='orange') 
        plt.legend()
        plt.savefig(os.path.join(dir,Fold,'real_vs_pred.png'))
        plt.close()
        if counter_fold==11:
            model.save_weights(f'/home/j/usfq/Proyecto-Integrador/StockPredictionModels/Models/TRANSFORMER/Weights/model_{arch}_{ticker}.h5')
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
    Taverage_mape=np.mean(t_mapes)
    metrics.append(Taverage_mape)
    Vaverage_mape=np.mean(v_mapes)
    metrics.append(Vaverage_mape)
    Taverage_rmse=np.mean(t_rmses)
    metrics.append(Taverage_rmse)
    Vaverage_rmse=np.mean(v_rmses)
    metrics.append(Vaverage_rmse)
    Taverage_mpe=np.mean(t_mpes)
    metrics.append(Taverage_mpe)
    Vaverage_mpe=np.mean(v_mpes)
    metrics.append(Vaverage_mpe)
    metrics_df[ticker_symbol]=metrics
    metrics=[]
    #save dataframe as csv
    metrics_df.to_csv(project_path+f'/Results/metrics/{arch}_metrics.csv',index=False)
    print(f'{ticker} done')

    Tstd_mse=np.std(t_mses)
    Tstd_mae=np.std(t_maes)
    Tstd_mape=np.std(t_mapes)
    Tstd_rmse=np.std(t_rmses)
    Vstd_mse=np.std(v_mses)
    Vstd_mae=np.std(v_maes)
    Vstd_mape=np.std(v_mapes)
    Vstd_rmse=np.std(v_rmses)
    Tstd_mpe=np.std(t_mpes)
    Vstd_mpe=np.std(v_mpes)
    std_metrics=[Tstd_mse,Tstd_mae,Tstd_mape,Tstd_rmse,Vstd_mse,Vstd_mae,Vstd_mape,Vstd_rmse,Tstd_mpe,Vstd_mpe]
    std_metrics_df[ticker_symbol]=std_metrics
    std_metrics_df.to_csv(project_path+f'/Results/metrics/{arch}_std_metrics.csv',index=False)



#define dataframe for metrics
df_ps=pd.DataFrame()
df_ps['MSE']=mse_t_p
df_ps['MAE']=mae_t_p
df_ps['MAPE']=mape_t_p
df_ps['RMSE']=rmse_t_p
df_ps['MPE']=mpe_t_p
df_ps.to_csv(project_path+f'/Results/metrics/{arch}_hypothesis.csv',index=False)

df_loss=pd.DataFrame()
df_loss['Training Loss']=overall_mse_train
df_loss['Training Std']=overall_std_train
df_loss['Validation Loss']=overall_mse_val
df_loss['Validation Std']=overall_std_val
df_loss.to_csv(project_path+f'/Results/metrics/{arch}_loss.csv',index=False)



 


