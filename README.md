# stock_application

This is an app which I have put together for analysis of Stock Price. The app has been built on Streamlit and the cloud instance has been used to 
host the app. The app can be accessed at https://ammihir-stocks-crypto-analyses-full-site-im72jn.streamlit.app/

Home :  Introduction Page <br />
Analysis : Analysis for a stock ticker with all the fundamentals. The data is pulled real time from Yahoo Finance. It will give a view of current price, 52 high/lows and the market cap.<br />
Portfolio Engineering: This widget objective is help a user simulate the future value of the portfolio using Monte Carlo simulations.<br />
Forecasting: Forecasting widget is used predict the price of a stock based on the current and historical prices. This widget uses the prophet package developed by Facebook.<br />
Portfolio Optmization: The widget will simulate the portfolio and will map the risk to reward ratio, and give the best weights of stocks for a portfolio such that the portfolio will yeild the best return for least risk possible.<br />
Chatbox: Makes use of a simple library to help answer simple questions. eg (what is bitcoin?)<br />
Crypto: The widget aims to pull crypto(Bitcoin related data from coingram). Embedding a iframe within the streamlit component.<br />
