**Calculus**

An agent made using numpy which extracts a stock price and news and then uses it to get the next day price prediction.Nothing related to calculus just thought it sounded cool

**How it works**

>It uses yfinance and polygon (currently called massive) to get the price data and the news data

>It then uses the vaderSentiment library to give a sentiment to the news

>It then use pandas library to make or update the data
 or use the extracter script to extract data for any period of time

>Finally it uses numpy to train the prediction model and make the predictions for the stock

>It then uses the results to do a simulation trade

**To use**
open webiste "https://calculus-uyiv.onrender.com/"

or

run this command "pip install numpy pandas requests vaderSentiment yfinance matplotlib streamlit"
and then run this command in the terminal "streamlit run calculus.py"

**note**: make a .env file and add your massive.com api key in this format "poly_api = your api key here" to run the file correctly

**note**: To change the stock to predict just the change the tick_name in the calculus.py 

**The website may take a little long to load please bear and also if the website is sleeping it will wake up in sometime and also it should not sleep but if**

**You can also check the test run for some of the more famous stock in the accuracy_stock.txt file**

