import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import pandas as pd
from plotly import graph_objs as go 

# Page title and config
st.set_page_config(
    page_title='Return'
)
st.title('Return & Risk of the Stock')

# setting up structure to retrive data
ticker = st.sidebar.text_input('Ticker')
comp_list = []  # Create an empty list for the companies
comp_list.append(ticker)
# ticker2 = st.sidebar.text_input('Second Ticker (ideally broader market index)')
# stock_name2 = 'DJIA'
stock_name2 = st.sidebar.text_input('Ticker2')
comp_list.append(stock_name2) 

if ticker and stock_name2 is not None:
# Extracting data
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    for stock in comp_list:
        globals()[stock] = yf.download(stock, start, end)
        
    company_list = [eval(x) for x in comp_list]
    company_name = comp_list
    #["Company1", "DJIA"]

    for company, com_name in zip(company_list, company_name):
        company["company_name"] = com_name
    df = pd.concat(company_list, axis=0) 
    df=df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d').dt.date
    # st.dataframe(df[df['company_name'] == ticker].tail(10))


    # Using pct_change to find the percent change for each day
    for company in company_list:
        company['Daily Return'] = company['Adj Close'].pct_change()

    # Plotting the daily return percentage
    fig, axes = plt.subplots(nrows=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    # daily returns charts
    def ticker1_daily():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = company_list[0].index,y = company_list[0]['Daily Return'],name = f'{ticker} Daily Returns'))
        fig.layout.update(title_text = f'{ticker} Daily Returns',xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    ticker1_daily()


    def ticker2_daily():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = company_list[1].index,y = company_list[1]['Daily Return'],name = f'{stock_name2} Daily Returns'))
        fig.layout.update(title_text = f'{stock_name2} Daily Returns',xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    ticker2_daily()



    company_list[0]['Daily Return'].plot(ax=axes[0], legend=True, linestyle='--', marker='o')
    axes[0].set_title(ticker)

    company_list[1]['Daily Return'].plot(ax=axes[1], legend=True, linestyle='--', marker='o')
    axes[1].set_title(stock_name2)
    fig.tight_layout()

    # 
    st.subheader(f'{ticker} and {stock_name2} Returns Histogram')
    two_subplot_fig = plt.figure(figsize=(12, 9))

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Daily Return'].hist(bins=50)
        plt.xlabel('Daily Return')
        plt.ylabel('Counts')
        plt.title(f'{company_name[i - 1]}')
        
    plt.tight_layout()
    st.pyplot(two_subplot_fig) 

    # Grab all the closing prices for the tech stock list into one DataFrame

    closing_df = pdr.get_data_yahoo(comp_list, start=start, end=end)['Adj Close']

    # Make a new tech returns DataFrame
    comp_rets = closing_df.pct_change()
    comp_rets.head()

    rets = comp_rets.dropna()

    area = np.pi * 20

    plt.figure(figsize=(10, 8))
    plt.scatter(rets.mean(), rets.std(), s=area)
    plt.xlabel('Expected return')
    plt.ylabel('Risk')

    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                    arrowprops=dict(arrowstyle='-', color='purple', connectionstyle='arc3,rad=-0.3'))
else:
    st.write('Input Tickers')

