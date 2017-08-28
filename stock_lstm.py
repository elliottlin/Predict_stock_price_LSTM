import numpy as np
import matplotlib.pyplot as plt
import math, time, itertools, datetime
import pandas as pd

from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM


def get_stock_data(stock_name, normalized=0):
    def get_ma_day(df, index, days):
        #return np.round(df[index].rolling(window = days, center = False).mean(), 2)
        # df need to be a DataFrame
        if not isinstance(df, pd.DataFrame):
            return None
        col = df[index]
        l = len(col)
        return [ col[i-days+1:i+1].mean() for i in range(l)] # first days-1 will be None because of the indexing handling

    def get_price_change(df):
        close_price = df['Close']
        return  np.log(close_price) - np.log(close_price.shift(1))
    
    url="http://www.google.com/finance/historical?q="+stock_name+"&startdate=Jul+12%2C+2013&enddate=Aug+11%2C+2017&num=30&ei=rCtlWZGSFN3KsQHwrqWQCw&output=csv"

    # Get
    col_names = ['Date','Open','High','Low','Close','Volume']
    stocks = pd.read_csv(url, header=0, names=col_names) 
    # reverse cuz it was backward
    stocks = stocks[::-1]
    
    stocks['MA5'] = get_ma_day(stocks,'Close',5)
    stocks['MA10']= get_ma_day(stocks,'Close',10)
    stocks['MA20']= get_ma_day(stocks,'Close',20)

    stocks['VMA5'] = get_ma_day(stocks,'Volume',5)
    stocks['VMA10'] = get_ma_day(stocks,'Volume',10)
    stocks['VMA20'] = get_ma_day(stocks,'Volume',20)

    stocks['price_change'] = get_price_change(stocks)
    #print(stocks.head(10))
    
    # Drop
    stocks = stocks.drop(['Date','Low'], axis=1)
    
    # Normalize
    df = pd.DataFrame(stocks)
    if normalized:
        df = df/df.mean() -1
    
    # drop first 19 NaN rows caused by MA/VMA
    return df[20:]

    # Normalize
    '''
    # not useful because the unbalanced range of each dimension.
    # lots of information would be masked
    
    from sklearn.preprocessing import StandardScaler,Normalizer
    X = StandardScaler().fit_transform(stocks)
    df=pd.DataFrame(X)
    print("x mean : ", X.mean(axis = 0))
    print("x.std : ", X.std(axis = 0))
    
    return df
    '''

def load_data(stock, seq_len, ratio=0.9):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() 
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)    # (len(), seq, cols) contains newest date
    
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    #np.random.shuffle(train)
    
    x_train = train[:, :-1]      # (len(), 10, 4) drop last row(), because last row contain the label
    y_train = train[:, -1][:,2] # with last row, and only keep "close" column @ [Open, High,"Close", Volume,...]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,2]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  
    
    return [x_train, y_train, x_test, y_test]

    '''
    In general, use following to split train/test sets
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df, df_labels, test_size=0.2, random_state=0)
    
    '''

def build_model(layers):
    d = 0.2
    model = Sequential()
    
    # now model.output_shape == (None, 128)
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    
    # for subsequent layers, no need to specify the input size:
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(d))
    
    # fully connected layer
    model.add(Dense(16,kernel_initializer='uniform',activation='relu'))        
    model.add(Dense(1,kernel_initializer='uniform',activation='linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model