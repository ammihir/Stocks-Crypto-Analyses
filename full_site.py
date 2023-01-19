import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import cv2
import pandas as pd
#from st_aggrid import AgGrid
import plotly.express as px
import io 

#Below is MC##
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
    #import alpaca_trade_api as tradeapi
####
    
    
with st.sidebar:
    choose = option_menu("Qauntics", ["Home", "Analysis", "Portfolio Engineering", "Forecasting", "Contact"],
                         icons=['house', 'kanban', 'gear', 'activity','person lines fill'],
                         menu_icon="quora", default_index=1,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
    
    
logo = Image.open(r'C:\Users\Mihir\Desktop\test.jpg')
profile = Image.open(r'C:\Users\Mihir\Desktop\test.jpg')
if choose == "Home":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">About the Creator</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )
    
    st.write("Our objective to create a world class application to help beginners understand the naunces of investing in equity markets")    
    st.image(profile, width=700 )
    
    
elif choose == "Forecasting":
        
    import streamlit as st
    from datetime import date

    import yfinance as yf
    from prophet import Prophet
    from prophet.plot import plot_plotly
    from plotly import graph_objs as go

    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title('Stock Forecast App')

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','idfc.ns')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365


    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data


    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
            

elif choose == "Portfolio Engineering":

    def montecarlo(df):

        daily_returns = df.pct_change()

        #st.write(daily_returns)


        avg_daily_returns = {}
        for i in range(0,len(selected_stock)):
            avg_daily_returns['avg_daily_returns_'+selected_stock[i]] = daily_returns.mean()[selected_stock[i]+'_df']


        #st.write(avg_daily_returns)

        std_daily_returns = {}
        for i in range(0,len(selected_stock)):
            std_daily_returns['std_daily_returns_'+selected_stock[i]] = daily_returns.std()[selected_stock[i]+'_df']    

        num_simulations = 100
        num_trading_days = 252

        last_prices = {} 
        for i in range(0,len(selected_stock)):     
            last_prices['last_price_' + selected_stock[i] ] = df[selected_stock[i]+'_df'][-1]

        portfolio_cumulative_returns = pd.DataFrame()

        for n in range(num_simulations):

            simulated_price_df = pd.DataFrame()

            simulated_prices = {}
            for i in range(0,len(selected_stock)):
                simulated_prices['simulated_prices_'+selected_stock[i]] = [last_prices['last_price_' + selected_stock[i]]]


            #st.write(simulated_prices)


            single_sim_price = {}

            for j in range(0,len(selected_stock)):
                for i in range(num_trading_days):


                    single_sim_price["single_sim_price"+selected_stock[j]] = simulated_prices['simulated_prices_'+selected_stock[j]][-1] * (1 + np.random.normal(avg_daily_returns['avg_daily_returns_'+selected_stock[j]], std_daily_returns['std_daily_returns_'+selected_stock[j]]))


                    simulated_prices['simulated_prices_'+selected_stock[j]].append(single_sim_price["single_sim_price"+selected_stock[j]])



            simulated_price_dict={}
            for i in range(0,len(selected_stock)):
                simulated_price_dict['simulated_price_'+selected_stock[i]] = pd.Series(simulated_prices['simulated_prices_'+selected_stock[i]])

            simulated_price_df = pd.concat(simulated_price_dict.values(), axis = 1).set_axis(b_dictionary, axis=1)    

            simulated_daily_returns = simulated_price_df.pct_change()        

            weights = list(a_dictionary.values())
            base = 100
            weights = [x /100 for x in weights]
            #weights  = [0.25,0.25,0.25,0.25]


            portfolio_daily_returns = simulated_daily_returns.dot(weights)        

            portfolio_cumulative_returns[f"Simulation {n+1}"] = (1 + portfolio_daily_returns.fillna(0)).cumprod()

        st.line_chart(portfolio_cumulative_returns)

        ending_cumulative_returns = portfolio_cumulative_returns.iloc[-1, :]
        ending_cumulative_returns.head()
        ending_cumulative_returns.plot(kind="hist", bins=10) 


        confidence_interval = ending_cumulative_returns.quantile(q=[0.025, 0.975])

        initial_investment = 10000

    # Calculate investment profit/loss of lower and upper bound cumulative portfolio returns
        investment_pnl_lower_bound = initial_investment * confidence_interval.iloc[0]
        investment_pnl_upper_bound = initial_investment * confidence_interval.iloc[1]

    # Print the results
        print(f"There is a 95% chance that an initial investment of $10,000 in the portfolio"
          f" over the next 252 trading days will end within in the range of"
          f" ${investment_pnl_lower_bound} and ${investment_pnl_upper_bound}")

        st.write('There is a 95% chance that an initial investment of $10,000 in the portfolio over next year/252 will be in range',investment_pnl_lower_bound,'and',investment_pnl_upper_bound)





    st.title('Monte Carlo Simulations')
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','idfc.ns')
    selected_stock = st.multiselect('Create a portfolio', stocks)
    st.write("Total stocks seleted",len(selected_stock))


    #st.write('Input something')


    a_dictionary = {}
    for i in range(0,len(selected_stock)):
        a_dictionary[selected_stock[i]]  = st.number_input('Allocation for '+selected_stock[i],key=i, min_value =10, max_value =100, step =10)



    st.write(a_dictionary)


    end_date = pd.to_datetime('today').date()
    start_date = end_date - timedelta(365)



    b_dictionary = {}
    for i in range(0,len(selected_stock)):
        b_dictionary[selected_stock[i]+'_df']  = yf.download(stocks[i], start_date, end_date)['Adj Close']

    #st.write(b_dictionary)

    try:
    #we are converting the dictionaries into a df, with the keys as the column names
        df = pd.concat(b_dictionary.values(), axis = 1).set_axis(b_dictionary, axis=1)

    except:
        st.write('PLEASE ENTER A STOCK')
    #st.write(df)




    if sum(a_dictionary.values()) == 100:
        st.write("Thanks! Please wait for the calculations")
        montecarlo(df)

    else:
        st.write("Allocations should be 100%")









            
            
elif choose == "Analysis":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Financial Dashboard</p>', unsafe_allow_html=True)

    #st.subheader('Equity Data')
    st.markdown('The data will be pulled from Yahoo finances')

    import streamlit as st

    import pandas as pd
    import yfinance as yf

    st.title('Fundamental Analysis')

    #st.header('This is the header below')

    tickers = ('idfc.ns','HDFCBANK.NS' ,'HDB','^NSEI','ABB')

    dropdown = st.selectbox('Pick your stock', tickers)



    start  =st.date_input('Start',value = pd.to_datetime('2021-01-01'))
    end = st.date_input('End',value = pd.to_datetime('today'))

    ticker = yf.Ticker(dropdown).info


    import datetime
    end_52 = pd.to_datetime('today').date()
    start_52 = end_52 - datetime.timedelta(365)

    df_52 = yf.download(dropdown, start_52, end_52)


    st.write('52-high',round(df_52['Adj Close'].min(),1))
    st.write('Current Price', ticker['currentPrice'])
    st.write('52-low',round(df_52['Adj Close'].max(),1))
    st.write('PEG ratio', ticker['pegRatio'])

    st.write('Market Cap', ticker['marketCap'])




    #st.text(hdb['sector'])

    def relativeret(df):
        rel = df.pct_change()
        cumret = (1+rel).cumprod()-1
        cumret = cumret.fillna(0)
        return cumret

    if len(dropdown)  > 0:
        #df = yf.download(dropdown, start, end)['Adj Close']

        df = relativeret(yf.download(dropdown, start, end)['Adj Close'])
        st.write("**BELOW IS RETURNS**")   
        st.line_chart(df)

    st.write("**BELOW IS CLOSING PRICE**")    

    st.line_chart(yf.download(dropdown, start, end)['Adj Close'])



