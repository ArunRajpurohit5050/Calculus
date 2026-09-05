import numpy as np
import os
import requests
import pandas as pd
import time
from datetime import timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

tick_name = "JNJ"
tick_matrix = f"{tick_name}5y_matrix.csv"

st.title(f"{tick_name} stock predictor")
st.write("Built for macondo hack club")

#check latest date



print("matrix is: ", tick_matrix)
#comare and get data
analyze = SentimentIntensityAnalyzer()

key =  st.secrets["poly_api"]
raw_day_score={}

ticker = yf.Ticker(tick_name)
@st.cache_data(ttl=86400)
def update_stock_data():
    matrix = tick_matrix

    data = pd.read_csv(matrix , index_col= 0,parse_dates=True )
    last_date = data.index[-1]
    print("last date: ", last_date)

    today_date = pd.Timestamp.now().normalize()
    yesterday_date = today_date - pd.Timedelta(days=1)
    print(f"today and yester day date:{today_date} and {yesterday_date}")
    start_date = last_date + pd.Timedelta(days=1)
    tick = tick_name
    start = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end = yesterday_date.strftime("%Y-%m-%dT23:59:59Z")
    if yesterday_date > last_date:
        print("data is left")
        missing_data = (yesterday_date - last_date).days
        print("days missing:", missing_data)
        
        latest_price = ticker.history(start= start_date.strftime("%Y-%m-%d"))
        latest_price.index = pd.to_datetime(latest_price.index).tz_localize(None).normalize()
        print("latest :", latest_price)
        url = (f"https://api.massive.com/v2/reference/news?ticker={tick}&published_utc.gte={start}&published_utc.lte={end}&limit=1000&apiKey={key}")
        print("got the prices")
        while url:
            try:
                response = requests.get(url, timeout=10)
            except requests.exceptions.RequestException as e:
                st.error(f"error is {e}")
                print("error: ", e)
                break
            if response.status_code == 200:
                print("getting the news")
                data_res = response.json()
                articles = data_res.get("results", [])
            

                for article in articles:

                    title = article.get("title","no title")

                    summ = article.get("description","no summary")


                    date_cut = article["published_utc"][:10]    

                    title_score = analyze.polarity_scores(title)["compound"]

                    if summ:

                        summ_score = analyze.polarity_scores(summ)["compound"]

                        final_score = (title_score+summ_score) /2

                    else:

                        final_score = title_score

                    if date_cut not in raw_day_score:

                        raw_day_score[date_cut] =[]

                    raw_day_score[date_cut].append(final_score)

                next = data_res.get("next_url")

                if next:

                        url = next + f"&apiKey={key}"

                        time.sleep(15)

                else:

                        st.write("nah, no more. Done getting the news")

                        break
            
            else:
                 print("there is an error response code diff")
                 print(response.status_code)
        daily_score ={}
        if raw_day_score:
            for date, score in raw_day_score.items():
                daily_score[date] = sum(score) /len(score)
            sent_pd = pd.DataFrame.from_dict(daily_score, orient= "index", columns=["sent"])
            sent_pd.index = pd.to_datetime(sent_pd.index).tz_localize(None).normalize()
            latest_price = latest_price.join(sent_pd)
        print("the scores are : ", daily_score)

        latest_price["sent"] = latest_price.get("sent", pd.Series(dtype=float)).fillna(0.0)

        data = pd.concat([data, latest_price])
        data.to_csv(matrix)
        print("cvs updated")
    else:
        print("got all data")
    pass

update_stock_data()
learn_rate = 0.002
epochs = 5000
scale_factor = 100
window_size = 10

@st.cache_data
def load_data(mat):
    return pd.read_csv(mat, index_col=0, parse_dates=True)

data = load_data(tick_matrix)

total_rows = len(data)
non_zero_sent = (data["sent"] != 0.0).sum()
print(f"total rows: {total_rows}, non-zero sentiment rows: {non_zero_sent}")

raw_price = data["Close"].values
data["scaled_close"] = data["Close"] / scale_factor
data["vol"] = data["Volume"] / 10000000
data["ma_5"] = data["scaled_close"].rolling(window=5).mean()
data = data.dropna()

answer_thing = "scaled_close"
print("the gussing item is: ",answer_thing)
def make_matrix(df, window_size):
    features = df[["scaled_close","sent","ma_5", "vol"]].values
    ans = df[answer_thing].values
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
    features = df[["scaled_close","sent","ma_5","vol"]].values
    ans = df[answer_thing].values
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

weight_dir = np.random.randn(x_train.shape[1]) * 0.01
bias_dir = 0.0
learn_dir = 0.01

#train---
weights = np.random.randn(x_train.shape[1]) * 0.01
bias = np.zeros(1)
price_day = x_test[:,-4]


for epoch in range(epochs):
    prediction = np.dot(x_train,weights)+ bias
    prediction_dir = sigmod(np.dot(x_train,weight_dir)+ bias_dir)
    error_dir = y_train_dir - prediction_dir
    error = y_train - prediction
    adj_dir = np.dot(x_train.T, error_dir) * (learn_dir / total_train_rows)
    bias_adj_dir = np.sum(error_dir) * (learn_dir / total_train_rows)
    adj = np.dot(x_train.T, error)* ( learn_rate/total_train_rows)
    bias_adj = np.sum(error) * (learn_rate/total_train_rows)
    weights = weights + adj
    bias = bias + bias_adj
    weight_dir = weight_dir + adj_dir
    bias_dir = bias_dir + bias_adj_dir
    if epoch % 1000 == 0:
        st.write(f"echop: {epoch}")
        print(f"echop: {epoch}")
st.write(" train successful")
print("done training")
#final day direction ---
final_dir_raw = sigmod(np.dot(x_test,weight_dir)+ bias_dir)
final_dir = (final_dir_raw>=0.5).astype(int)
ai_money = 10000
stock_money = 0
average_price = 0
stock_own = 0
print(y_test_dir)
print(final_dir)
y_test_scaled = (y_test + price_day) * scale_factor
price_scaled = price_day * scale_factor
print("prices", y_test_scaled)

if len(final_dir) != len(y_test_dir):
        print("lengths are not equal")
        print(len(final_dir), len(y_test_dir))
else:
        print("lengths are equal")

st.write("simulating ai trading")
st.write("initial ai money: ", ai_money)

for i in range(len(final_dir)):
     print(i)
     price_fluc = y_test_scaled[i] - price_scaled[i]
     if final_dir[i] == 1:
          print("go in")
          ai_money = ai_money - price_scaled[i]
          average_price = (average_price + y_test_scaled[i]) / 2 
          stock_money = stock_money + price_scaled[i]
          stock_own = stock_own + 1
          if y_test_dir[i] == 1:
                stock_money = stock_money + (price_fluc * stock_own)
                print("profit")
          elif y_test_dir[i] == 0 and stock_money > 0:
                ai_money = ai_money + stock_money + (price_fluc * stock_own)
                stock_money = 0
                stock_own = 0
                print("loss")
                if ai_money < 0:
                    print("ai money is negative, game over")
                    print("time to exit", i)
                    break
     if final_dir[i] == 0 and stock_money > 0:
            print("go out")
            ai_money = ai_money + stock_money + (price_fluc * stock_own)
            stock_money = 0
            stock_own = 0
     print(f"ai money: {ai_money}, stock money: {stock_money}")
print("final ai money: ", ai_money)
print("final stock money: ", stock_money)
print("final stock own: ", stock_own)
final_money = ai_money + stock_money + (price_fluc * stock_own)
print("final summary", final_money)
accuracy = np.mean(final_dir == y_test_dir) * 100

st.write("final ai money: ", ai_money)
st.write("final money on stocks: ", stock_money)
st.write("final stocks owned: ", stock_own)
st.write("final sum: ", final_money)

st.metric(label="price direction accuracy: ", value= accuracy)


print("train")
#final day price ---

final_day_raw = np.dot(x_test,weights)+bias
final_day = np.clip(final_day_raw,-0.05,0.05)
final_error = y_test - final_day_raw
final_loss = np.mean(final_error ** 2)
st.metric(label="bias: ", value=float(bias[0]))
st.metric(label="loss: ", value=final_loss)


np.save("weights.npy", weights)
np.save("bias.npy", bias)


final_day_change = (price_day + final_day)* scale_factor
y_plot = (y_test + price_day)*scale_factor

#graph ---
plt.plot(y_plot, label=f"actual {tick_name} price", color="blue")
plt.plot(final_day_change, label=f"predicted {tick_name} price", color="red", linestyle="dashed")
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

     