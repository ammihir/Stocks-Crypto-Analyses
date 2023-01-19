import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
#import cv2
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
    choose = option_menu("Qauntics", ["Home", "Analysis", "Portfolio Engineering", "Forecasting", "Portfolio Optimization","Chatbot", "crypto"],
                         icons=['house', 'kanban', 'gear', 'activity','person lines fill', 'house', 'question'],
                         menu_icon="quora", default_index=3,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
    
    
#logo = Image.open(r'quant_image.jpg')
profile = Image.open(r'finplan.png')
if choose == "Home":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'calibri'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">About Us</p>', unsafe_allow_html=True)    
    #with col2:               # To display brand log
        #st.image(logo, width=100 )
    
    st.subheader("Our objective to create a world class app to help beginners understand the naunces of investing in equity markets")    
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

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','idfc.ns','HDFCBANK.NS')
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

    st.subheader('Historical Data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()
    
    #adding this to address issues in streamlit#### 
    def remove_timezone(dt):
        return dt.replace(tzinfo=None)
    ###############
    
    # Predict forecast with Prophet.
    df_train = data[['Date','Close']]
    df_train['Date'] = df_train['Date'].apply(remove_timezone) # this is also extra
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    #st.write(f'Forecast plot for {n_years} years')
    
    fig1 = plot_plotly(m, forecast)
    fig1.layout.update(title_text='Time Series data with Rangeslider for '+ str(n_years) + ' years', xaxis_rangeslider_visible=True)
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

        st.markdown('*There is a 95% chance that an initial investment of $10,000 in the portfolio over next year will be in range of*  '+str(round(investment_pnl_lower_bound))+' *and* '+str(round((investment_pnl_upper_bound))))

        #st.latex('There is a 95% chance that an initial investment of $10'+str(45))



    st.title('Monte Carlo Simulations')
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','idfc.ns')
    selected_stock = st.multiselect('Create a portfolio', stocks)
    st.metric("Total stocks seleted",len(selected_stock))


    #st.write('Input something')


    a_dictionary = {}
    for i in range(0,len(selected_stock)):
        a_dictionary[selected_stock[i]]  = st.number_input('Allocation for '+selected_stock[i],key=i, min_value =10, max_value =100, step =10)



    #st.write(a_dictionary)


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
        st.header("Please select stonks")
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
    #st.markdown('<p class="font">Financial Dashboard</p>', unsafe_allow_html=True)

    
    

    import streamlit as st

    import pandas as pd
    import yfinance as yf

    st.title('Fundamental Analysis')
    #st.subheader('Data Source: Yahoo Finance')
    #st.markdown('**Data Source: Yahoo Finance**')
    
    
    #st.header('This is the header below')

    tickers = ('idfc.ns','HDFCBANK.NS' ,'HDB','^NSEI','ABB','AMZN')

    dropdown = st.selectbox('Enter stock ticker', tickers)



    start  =st.date_input('Start',value = pd.to_datetime('2021-01-01'))
    end = st.date_input('End',value = pd.to_datetime('today'))

    ticker = yf.Ticker(dropdown).info


    import datetime
    end_52 = pd.to_datetime('today').date()
    start_52 = end_52 - datetime.timedelta(365)

    df_52 = yf.download(dropdown, start_52, end_52)
    
    st.markdown('**Industry** '+ticker['sector'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric('52 High',round(df_52['Adj Close'].min(),1),delta_color="normal")
    col2.metric('52 Low',round(df_52['Adj Close'].max(),1))
    col3.metric('Current Price', ticker['currentPrice'])
    
    bol1,bol2 =st.columns(2)
    
    bol1.metric('PEG ratio', ticker['pegRatio'])
    bol2.metric('Market Cap', ticker['marketCap'])

    


    
    
    
    
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


elif choose == "Chatbot":
    
    
    
    import people_also_ask
    import json
    
    

    def ask_abot():
        question = st.text_input('Insert question below')
        if question:
            try:
                answer = people_also_ask.get_answer(question)
                st.write(answer['response'])
            except:
                st.write('Ask a different question')

    
    ask_abot()
    
    
        
elif choose == "Portfolio Optimization":
    
    ##Solution:: https://github.com/damianboh/portfolio_optimization/blob/main/portfolio_optimization_streamlit/app.py
    
    import streamlit as st
    from pandas_datareader.data import DataReader
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns
    from pypfopt import plotting
    import copy
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    from io import BytesIO

    def plot_cum_returns(data, title):    
        daily_cum_returns = 1 + data.dropna().pct_change()
        daily_cum_returns = daily_cum_returns.cumprod()*100
        fig = px.line(daily_cum_returns, title=title)
        return fig

    def plot_efficient_frontier_and_max_sharpe(mu, S): 
        # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
        ef = EfficientFrontier(mu, S)
        fig, ax = plt.subplots(figsize=(6,4))
        ef_max_sharpe = copy.deepcopy(ef)
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
        # Find the max sharpe portfolio
        ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
        ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
        # Generate random portfolios
        n_samples = 1000
        w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
        sharpes = rets / stds
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
        # Output
        ax.legend()
        return fig

    #st.set_page_config(page_title = "Bohmian's Stock Portfolio Optimizer", layout = "wide")
    st.title("Efficient Frontier Portfolio Optimizer")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date",datetime(2013, 1, 1))

    with col2:
        end_date = st.date_input("End Date") # it defaults to current date

    #tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas \ WITHOUT spaces, e.g. "MA,FB,V,AMZN,JPM,BA"', '').upper()
    #tickers = tickers_string.split(',')

    
    
    
    
    
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','idfc.ns')
    selected_stock = st.multiselect('Create a portfolio', stocks)
    st.metric("Total stocks seleted",len(selected_stock))  
    b_dictionary = {}
    for i in range(0,len(selected_stock)):
        b_dictionary[selected_stock[i]+'_df']  = yf.download(stocks[i], start_date, end_date)['Adj Close']

    #st.write(b_dictionary)

    
    
    try:
    #we are converting the dictionaries into a df, with the keys as the column names
        df = pd.concat(b_dictionary.values(), axis = 1).set_axis(b_dictionary, axis=1)    
    except:
        st.header("Please select stonks")
    
    
    

        
    
    
    
    
    
    try:
        # Get Stock Prices using pandas_datareader Library
        #stocks_df = DataReader(tickers, 'yahoo', start = start_date, end = end_date)['Adj Close']
        # Plot Individual Stock Prices
        fig_price = px.line(df, title='Price of Individual Stocks')
        # Plot Individual Cumulative Returns
        fig_cum_returns = plot_cum_returns(df, 'Cumulative Returns of Individual Stocks Starting with $100')
        
        
        
        
        # Calculatge and Plot Correlation Matrix between Stocks
        corr_df = df.corr().round(2)
        fig_corr = px.imshow(df, text_auto=True, title = 'Correlation between Stocks')

        # Calculate expected returns and sample covariance matrix for portfolio optimization later
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        # Plot efficient frontier curve
        fig = plot_efficient_frontier_and_max_sharpe(mu, S)
        fig_efficient_frontier = BytesIO()
        fig.savefig(fig_efficient_frontier, format="png")

        # Get optimized weights
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=0.02)
        weights = ef.clean_weights()
        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
        weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
        weights_df.columns = ['weights']

        # Calculate returns of portfolio with optimized weights
        df['Optimized Portfolio'] = 0
        for ticker, weight in weights.items():
            df['Optimized Portfolio'] += df[ticker]*weight

        # Plot Cumulative Returns of Optimized Portfolio
        fig_cum_returns_optimized = plot_cum_returns(df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')

        # Display everything on Streamlit
        #st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))	
        st.plotly_chart(fig_cum_returns_optimized)

        st.subheader("Optimized Max Sharpe Portfolio Weights")
        st.dataframe(weights_df)

        st.subheader("Optimized Max Sharpe Portfolio Performance")
        st.image(fig_efficient_frontier)

        st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
        st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
        st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))

        st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
        st.plotly_chart(fig_price)
        st.plotly_chart(fig_cum_returns)



    except:
        st.header("Please select stonks")

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    

    
elif choose == "crypto":
    
    import streamlit.components.v1 as components
    
    
    st.image("https://alternative.me/crypto/fear-and-greed-index.png")
    
    st.title('Dominance')
    components.iframe("https://coingram.com/en/widget/dominance-chart", height = 400)
    
    st.title('Top 10 Coins')
    components.iframe("https://coingram.com/en/widget/ticker?hcolor=ffffff&tcolor=000000", height = 500)
    
