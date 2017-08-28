# Predict_stock_price_LSTM
A LSTM model to predict stock price using Keras

### step 1. get data from Goole finance, and return a pandas Dataframe

```python
df = get_stock_data(stock_name,normalized=1)
```

### step 2. load the data into sequence with the seq length

```python
sequence_len = 25
X_train, y_train, X_test, y_test = load_data(df, sequence_len)
```

### step 3. build a 2 stacked LSTM with 2 FCL with Keras, and return a Keras.Sequential

```python
model = build_model([X_train.shape[-1],sequence_len])
```

### step 4. train the model
```python
history = model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=500,
    validation_split=0.1,
    verbose=0)
```

### step 5. test on test set
```python
p = model.predict(X_test)
```

### step 6. plot with matplotlib
```python
plt.plot(p,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')
plt.legend(loc='best')
plt.show()
```
