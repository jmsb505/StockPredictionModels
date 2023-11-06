import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss,MAE
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic, augment_calendar_df
from neuralforecast.models import Autoformer    

df=pd.read_csv('/home/j/usfq/tesis/StockPredictionModels/Data/Complete.csv')
#Rename columns date to ds, close to y, and ticker_symbol to unique_id
df = df.rename(columns={'Date':'ds','Close':'y','ticker_symbol':'unique_id'})
#set unique_id,ds and y as the first columns
df1 = df[['unique_id','ds','y','p_sentiment', 'Open', 'High', 'Low', 'unrate',
       'psr', 'm2', 'dspic', 'pce', 'reer', 'ir', 'ffer', 'tcs', 'indpro',
       'ccpi']]
#only the unique_id==TSLA
exogen_cols=['p_sentiment', 'Open', 'High', 'Low', 'unrate',
       'psr', 'm2', 'dspic', 'pce', 'reer', 'ir', 'ffer', 'tcs', 'indpro',
       'ccpi']
df2 = df1[df1['unique_id']=='TSLA']

Y_train_df = df2[df2.ds<df2['ds'].values[-30]] # 132 train
Y_test_df = df2[df2.ds>=df2['ds'].values[-30]].reset_index(drop=True) # 12 test
Y_train_df['ds'] = pd.to_datetime(Y_train_df['ds'])
Y_test_df['ds'] = pd.to_datetime(Y_test_df['ds'])

model = Autoformer(h=29,
                 input_size=10,
                 hidden_size = 16,
                 conv_hidden_size = 32,
                 n_head=2,
                 loss=MAE(),
                 futr_exog_list=exogen_cols,
                 scaler_type='robust',
                 learning_rate=1e-3,
                 max_steps=300,
                 val_check_steps=50,
                 early_stop_patience_steps=2)

nf = NeuralForecast(
    models=[model],
    freq='M'
)
nf.fit(df=Y_train_df, val_size=30)

forecasts = nf.predict(futr_df=Y_test_df)

Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

if model.loss.is_distribution_output:
    plot_df = plot_df[plot_df.unique_id=='TSLA'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['Autoformer-median'], c='blue', label='median')
    plt.fill_between(x=plot_df['ds'][-12:], 
                    y1=plot_df['Autoformer-lo-90'][-12:].values, 
                    y2=plot_df['Autoformer-hi-90'][-12:].values,
                    alpha=0.4, label='level 90')
    plt.grid()
    plt.legend()
    plt.plot()
else:
    plot_df = plot_df[plot_df.unique_id=='TSLA'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['Autoformer'], c='blue', label='Forecast')
    plt.legend()