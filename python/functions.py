#! /usr/bin/env python3

import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def get_stonk(symbol):
    stonk = yf.Ticker(symbol)

    if(stonk.info["regularMarketPrice"] == None):
        return False

    # Daily OHCL, volume, market cap
    data = stonk.history(period='max', interval='1d')

    # Infer any missing values
    data = data.interpolate(method='time')

    stonk_data = {
        "info": stonk.info,
        "ewma": {
            "12": data['Close'].ewm(span=12, adjust=False).mean(),
            "26": data['Close'].ewm(span=26, adjust=False).mean(),
            "50": data['Close'].ewm(span=50, adjust=False).mean(),
            "200": data['Close'].ewm(span=200, adjust=False).mean()
        },
        "data": data.iloc[:, 0:5]
    }

    return stonk_data


def knn_ts(data, span, offset, k, n_preds, weighted_mse=True):
    if(type(data) is np.ndarray):
        data = data.flatten()
        data = pd.Series(data, name="Close")
    span = int(span)
    offset = int(offset)
    k = int(k)
    n_preds = int(n_preds)

    min_max_scaler = preprocessing.MinMaxScaler()

    # Slice out target
    target = data[-span:]
    # Min/max normalize
    x = target.values.reshape(-1,1)
    x_scaled = min_max_scaler.fit_transform(x).reshape(span,)
    norm = pd.Series(x_scaled, index=target.index, name="norm")
    # Add index position from original data
    index = pd.Series(np.arange(len(target)-span, len(target), 1), index=target.index, name="index")
    target = pd.concat([target, norm, index], axis=1)

    # For weighted MSE
    y_step = 1/span
    y_weights = np.arange(1, 2, y_step)
    y_weights = y_weights**2

    # Create dict of neighbors
    # Keys are RMSE and values are df containing window and index
    end = len(data)-((len(data)-span)%offset)-span*2+1
    vecs = dict()
    # Create window vectors
    for i in range(0, end, offset):
        # Slice out window
        window = data[i:i+span]
        # Min/max normalize
        x = window.values.reshape(-1,1)
        x_scaled = min_max_scaler.fit_transform(x).reshape(span,)
        window = pd.Series(x_scaled, index=window.index, name="Close")
        
        # Add index position from original data
        index = pd.Series(np.arange(i, i+span, 1), index=window.index, name="index")
        df = pd.concat([window, index], axis=1)

        if weighted_mse:
            # Calc weighted MSE
            mse = np.average((df["Close"].values-target["norm"].values)**2, axis=0, weights=y_weights)
        else:
            # Calc sum squared error (SSE)
            mse = mean_squared_error(df["Close"], target["norm"])

        # Assign keys as RMSE and values as window data
        vecs[mse] = df
    
    top_k = sorted(vecs.keys())[:k]
    
    preds = pd.DataFrame()
    for i in range(k):
        index = list(vecs[top_k[i]]["index"])
        win_raw = data[index[0]:index[-1]]
        max_raw = max(win_raw)
        min_raw = min(win_raw)
        x = data[index[-1]+1:index[-1]+1+n_preds]
        n_pred_norm = (x - min_raw) / (max_raw - min_raw)
        preds[f"k_{i}"] = n_pred_norm.to_numpy()
    
    preds["mean"] = preds.mean(axis=1)
    t_max = max(target["Close"])
    t_min = min(target["Close"])
    preds["predictions"] = (preds["mean"] * (t_max-t_min))+t_min

    return preds

def ensemble_lstm(start, end, ncol=6):
    nvda_all = get_stonk("nvda")
    nvda = nvda_all["data"]
    nvda = nvda[start:end]
    nvda["ewma_diff"] = nvda_all["ewma"]["26"] - nvda_all["ewma"]["12"]
    nvda.index.name = "date"
    nvda.reset_index(inplace=True)
    nvda["year_day"] = pd.to_datetime(nvda['date'].astype(str)).dt.day_of_year
    nvda = nvda.set_index(["date"])
    nvda = nvda[["Open", "High", "Low", "Close", "ewma_diff", "year_day"]]

    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(nvda)

    # Shape data for model
    x_data, y_data = create_dataset(data)
    
    predictions = list()
    nvda_less_adamax_ES_b32 = load_model("../reports/lstm_models/nvda_less_adamax_ES_b64.h5")
    nvda_less_adamax_e30_b64 = load_model("../reports/lstm_models/nvda_less_adamax_e30_b64.h5")
    models = [nvda_less_adamax_ES_b32, nvda_less_adamax_e30_b64]
    for model in models:
        pred = model.predict(x_data)
        pred_2d = np.zeros(shape=(len(pred), ncol))
        pred_2d[:,0] = pred[:,0]
        predictions.append(scaler.inverse_transform(pred_2d)[:,0])

    return sum(predictions) / len(predictions)

def create_dataset(df, win=30, t=5, tar_col=3):
    x, y = [], []
    for i in range(win, df.shape[0]):
        if i < df.shape[0]-t:
            x.append(df[i-win:i, :])
            y.append(df[i+t, tar_col])
    x, y = np.array(x), np.array(y)
    return x,y
