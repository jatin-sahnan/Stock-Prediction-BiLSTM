import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from keras.models import Sequential
from keras.layers import Dense, LSTM
yf.pdr_override()
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objs as go 
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Stock Forecast')
# setting up structure to retrive data
ticker = st.sidebar.text_input('Ticker')
comp_list = []  # Create an empty list for the companies
comp_list.append(ticker)

st.write(f'{ticker} Data')

start_date = yf.Ticker(ticker).history(period="max").index[0]

# Get the stock quote
df = pdr.get_data_yahoo(ticker, start = '2012-01-01', end=datetime.now())

st.write(df.tail()) 

def plot_ticker():
    plt.figure(figsize=(16,6))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df.index,y = df['Close'],name = f'{ticker} Close'))
    fig.layout.update(title_text = f'{ticker} Historical Close',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 
plot_ticker() 

# Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .9 ))

st.write('Length of Training Data', training_data_len) 

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape



# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=True))
#model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
st.cache()
model.fit(x_train, y_train, batch_size=1, epochs=1)
st.cache()

# Create the testing data set
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
st.write('RMSE: ',rmse)

# Plot the data
train = data[:training_data_len]
validate = data[training_data_len:]
validate['Predictions'] = predictions

#def LSTM_model():
#    fig = go.Figure()
#    fig.add_trace(go.Scatter(x = train.index,y = train['Close'],name = f'{ticker} Train'))
#   fig.add_trace(go.Scatter(x = validate.index,y = validate[['Close','Predictions']], name = f'{ticker} Validate Close'))
#    fig.layout.update(title_text = f'{ticker} LSTM Model Prediction',xaxis_rangeslider_visible=True)
#    st.plotly_chart(fig)
#LSTM_model()

st.cache()
plt.figure(figsize=(16,6))
plt.title('LSTM Model Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adj. Close Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(validate[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot()

def mape(actual, predicted):
    mask = actual != 0
    return (np.fabs(actual - predicted) / actual)[mask].mean() * 100
mape_val = mape(y_test,predictions)

st.write(f"MAPE on validation set: {mape_val:.2f}%") 

# Show the predicted prices vs actual
st.write(validate.head(10)) 


#Forecast for next 30 days 
n_steps1=len(np.concatenate((x_train[1:], x_test[1:])))
fut_inp = np.concatenate((x_train[1:], x_test[1:]))
fut_inp = fut_inp.reshape(1,-1)
tmp_inp = list(fut_inp)
fut_inp.shape

#Creating list of the last 100 data
tmp_inp = tmp_inp[0].tolist()

#Predicting next 30 days price suing the current data
#It will predict in sliding window manner (algorithm) with stride 1
lst_output=[]
n_steps=n_steps1
i=0
while(i<30):
    if(len(tmp_inp)>=n_steps):
        fut_inp = np.array(tmp_inp[-n_steps:])
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
    else:
        fut_inp = np.array(tmp_inp)
        fut_inp = fut_inp.reshape((1, len(tmp_inp), 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
    i=i+1

#Creating final data for plotting
Next_30days_forecast = scaler.inverse_transform(lst_output).tolist()
Next_30days_forecast=pd.DataFrame(Next_30days_forecast, columns=['30 Days Forecast'])
st.write(Next_30days_forecast.head(30)) 

