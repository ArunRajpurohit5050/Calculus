import numpy as np
import os
import requests
import pandas as pd
import time
from datetime import timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.title("apple stock predictor")
st.write("Built for macondo hack club")

learn_rate = 0.1
epochs = 5000
scale_factor = 1000
window_size = 4

@st.cache_data
def load_data():
    return pd.read_csv("appl_matrix.csv", index_col=0, parse_dates=True)

data = load_data()

raw_price = data["Close"].values
data["scaled_close"] = data["Close"] / scale_factor

def make_matrix(df, window_size):
    features = df[["scaled_close","sent"]].values
    ans = df["scaled_close"].values
    x= []
    y= []
    for i in range(len(features)-window_size):
        window = features[i:i+window_size].flatten()
        answer = ans[i+window_size]
        today = ans[i + window_size -1]
        change = answer - today
        x.append(window)
        y.append(change)
    return np.array(x), np.array(y)


def make_matrix_ud(df, window_size):
    features = df[["scaled_close","sent"]].values
    ans = df["scaled_close"].values
    y_dir = []
    for i in range(len(features)-window_size):
        window = features[i:i+window_size].flatten()
        answer = ans[i+window_size]
        dir = ans[i+window_size-1]
        if answer > dir:
             direction = 1
        else:
             direction = 0
        y_dir.append(direction)
    return np.array(y_dir)

x, y = make_matrix(data,window_size)
y_dir = make_matrix_ud(data, window_size)
total_rows = len(x)
split = int(len(x) *0.8)

#data split ---
x_train = x[:split]
y_train = y[:split]

x_test = x[split:]
y_test = y[split:]
total_train_rows = len(x_train)

y_train_dir = y_dir[:split]
y_test_dir  = y_dir[split:]

#train direction ---
def sigmod(z):
     return 1/(1+ np.exp(-z))

weight_dir = np.random.rand(x_train.shape[1])
bias_dir = 0.0
learn_dir = 0.1

#train---
weights = np.random.rand(8)
bias = np.random.rand(1)

for epoch in range(epochs):
    prediction = np.dot(x_train,weights)+ bias
    prediction_dir = sigmod(np.dot(x_train,weight_dir)+ bias_dir)
    error_dir = y_train_dir - prediction_dir
    error = y_train - prediction
    adj_dir = np.dot(x_train.T, error_dir) * (learn_dir / total_train_rows)
    bias_adj_dir = np.sum(error_dir) * (learn_dir / total_train_rows)
    adj = np.dot(error, x_train)* ( learn_rate/total_train_rows)
    bias_adj = np.sum(error) * (learn_rate/total_train_rows)
    weights = weights + adj
    bias = bias + bias_adj
    weight_dir = weight_dir + adj_dir
    bias_dir = bias_dir + bias_adj_dir
    if epoch % 1000 == 0:
        st.write(f"echop: {epoch}")
st.write(" train successful")
#final day direction ---
final_dir_raw = sigmod(np.dot(x_test,weight_dir)+ bias_dir)
final_dir = (final_dir_raw>=0.5).astype(int)
accuracy = np.mean(final_dir == y_test_dir) * 100
st.metric(label="price direction accuracy: ", value= accuracy)
print("train")
#final day price ---
raw_new_price = np.array([310.03,316.83])
new_price = raw_new_price / scale_factor

final_day_raw = np.dot(x_test,weights)+bias
final_day = final_day_raw
final_error = y_test - final_day_raw
final_loss = np.mean(final_error ** 2)
st.metric(label="bias: ", value=bias)
st.metric(label="loss: ", value=final_loss)


np.save("weights.npy", weights)
np.save("bias.npy", bias)

price_day = x_test[:,-2]
final_day_change = (price_day + final_day)* scale_factor
y_plot = (y_test + price_day)*scale_factor

#graph ---
plt.plot(y_plot, label="actual apple price", color="blue")
plt.plot(final_day_change, label="predicted apple price", color="red", linestyle="dashed")
for i in range(len(final_dir)):
     if final_dir[i] == y_test_dir[i]:
          plt.scatter(i, y_plot[i], color="green", s=20)
     else:
          plt.scatter(i, y_plot[i], color="red", s=20)
plt.scatter([],[], color="green", label="correct")
plt.scatter([],[], color="red", label="wrong")
plt.title("ai vs actual:AAPL")
plt.xlabel("test days")
plt.ylabel("price ($)")
plt.legend()
st.pyplot(plt)

     
