import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import os

start_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1825)
ticker_main = "BA"
tick = yf.Ticker(ticker_main)
end = pd.Timestamp.now().normalize()
key = os.environ.get("poly_api")
analyze = SentimentIntensityAnalyzer()
raw_day_score = {}
matrix = f"{ticker_main}5y_matrix.csv"
start_str = start_date.strftime("%Y-%m-%d")
end_str = end.strftime("%Y-%m-%d")

latest_price = tick.history(start= start_date.strftime("%Y-%m-%d"))
latest_price.index = pd.to_datetime(latest_price.index).tz_localize(None).normalize()
print("latest :", latest_price)
url = (f"https://api.massive.com/v2/reference/news?ticker={ticker_main}&published_utc.gte={start_str}&published_utc.lte={end_str}&limit=1000&apiKey={key}")
print("got the prices")
while url:
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"error is {e}")
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

        else:

                print("nah, no more. Done getting the news")

                break
    
    else:
            print("there is an error response code diff")
            print(response.status_code)
            break
daily_score ={}
if raw_day_score:
    for date, score in raw_day_score.items():
        daily_score[date] = sum(score) /len(score)
    sent_pd = pd.DataFrame.from_dict(daily_score, orient= "index", columns=["sent"])
    sent_pd.index = pd.to_datetime(sent_pd.index).tz_localize(None).normalize()
    latest_price = latest_price.join(sent_pd)
print("the scores are : ", daily_score)

latest_price["sent"] = latest_price.get("sent", pd.Series(dtype=float)).fillna(0.0)

columns = ["Open","High","Low","Close","Volume","Dividends","Stock Splits","sent"]
data = latest_price[columns]

data.to_csv(matrix)
print("cvs updated")

