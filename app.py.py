import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st


import yfinance as yf

start = "2010-01-01"
end = "2022-12-31"

st.title("Stock Trend Predcitions")

ticker = st.text_input("enter Stock Ticker" , 'TSLA' )

# Fetch historical stock data for Tesla (TSLA)
df = yf.download(ticker, start=start, end=end)


#Describing 
st.subheader("Data from 2010 - 2022")
st.write(df.describe())

##vISUALISATION

st.subheader("Closing Price Vs Time-Chart")
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price Vs Time-Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price Vs Time-Chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100 , 'r')
plt.plot(ma200 , 'b')
plt.plot(df.Close , 'g')
st.pyplot(fig)



# training and testing set
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# #spliiting the data into traing and testing set
# x_train = []
# y_train = []

# for i in  range(100 ,data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100 : i])
#     y_train.append(data_training_array[i , 0])
    
    
# x_train , y_train = np.array(x_train), np.array(y_train)


#load model
model = load_model('keras_model.h5')



#predictions 

past_100_Days = data_training.tail(100)
final_df = past_100_Days.append(data_testing  , ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test =[]

for i in range(100 , input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i , 0 ])

x_test ,y_test = np.array(x_test), np.array(y_test)


y_pred = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_pred= y_pred *scale_factor
y_test = y_test*scale_factor


#final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test , 'b' , label = 'Original Price')
plt.plot(y_pred , 'r' , label = "Predicted Price ")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
# Print the fetched data
# print(df.head())


# # Fetch historical stock data for Apple (AAPL)
# ticker = "AAPL"
# data = yf.download(ticker, start="2010-01-01", end="2022-12-31")

# # Print the fetched data
# print(data)
