#!/usr/bin/env python
# coding: utf-8

# In[10]:


#from nsepy import get_history as gh
from jugaad_data.nse import stock_df
from datetime import date
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[13]:


start = date(2022,1,1)
end = date(2022,12,31)
#stk_data = gh(symbol='SBIN',start=start,end=end)
stk_data = stock_df(symbol="SBIN", from_date=date(2017,1,1),
            to_date=date(2022,12,31), series="EQ")
stk_data.head


# In[17]:


plt.figure(figsize=(14,14))
plt.plot(stk_data['DATE'], stk_data['CLOSE'])
plt.title('Historical Stock Value')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()


# In[23]:


stk_data['Date'] = stk_data.index
data2 = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
data2['Date'] = stk_data['DATE']
data2['Open'] = stk_data['OPEN']
data2['High'] = stk_data['HIGH']
data2['Low'] = stk_data['LOW']
data2['Close'] = stk_data['CLOSE']


# In[24]:


train_set = data2.iloc[:, 1:2].values
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train_set)
X_train = []
y_train = []
for i in range(60, 1482):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0]) 
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))


# In[25]:


regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))


# In[26]:


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 15, batch_size=32)


# In[27]:


testdataframe= stock_df(symbol="SBIN", from_date=date(2022,4,1), to_date=date(2022,12,18), series="EQ")
testdataframe['Date'] = testdataframe.index
testdata = pd.DataFrame(columns = ['Date', 'Open', 'High', 'Low', 'Close'])
testdata['Date'] = testdataframe['DATE']
testdata['Open'] = testdataframe['OPEN']
testdata['High'] = testdataframe['HIGH']
testdata['Low'] = testdataframe['LOW']
testdata['Close'] = testdataframe['CLOSE']
real_stock_price = testdata.iloc[:, 1:2].values
dataset_total = pd.concat((data2['Open'], testdata['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(testdata) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 235):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))


# In[28]:


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[29]:


plt.figure(figsize=(20,10))
plt.plot(real_stock_price, color = 'green', label = 'SBI BANK Stock Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted SBI BANK Stock Price')
plt.title('SBI BANK Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI BANK Stock Price')
plt.legend()
plt.show()

