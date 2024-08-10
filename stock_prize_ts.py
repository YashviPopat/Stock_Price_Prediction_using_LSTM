import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data =  pd.read_csv(r"C:\\Users\\yashv\\Documents\\Nettyfy_Technologies\\Project_3(Time_series)\\Stock_Prize_Prediction.csv",encoding="utf-8")

# # # data.plot(figsize=(12, 5))
# # # plt.show()
# # data = data[['Open']]
# # data = data.dropna()

# # train_size = int(len(data) * 0.8)
# # train_data  = data[:train_size] 
# # test_data   = data[train_size:]

# # # plt.figure(figsize=(12, 6))
# # # plt.plot(train_data, label='Train Data')
# # # plt.plot(test_data, label='Test Data')
# # # plt.title('Train and Test Data')
# # # plt.xlabel('Date')
# # # plt.ylabel('Close Price')
# # # plt.legend()
# # # plt.show()

# # model = ARIMA(train_data, order=(1,1,1))
# # model_fit = model.fit()
# # print(model_fit.summary())

# # forecast = model_fit.forecast(steps=len(test_data))
# # forecast = pd.Series(forecast, index=test_data.index)
# # forecast = forecast.dropna()
# # print(forecast)

# # plt.figure(figsize=(12, 6))
# # plt.plot(train_data, label='Train Data')
# # plt.plot(test_data, label='Test Data')
# # plt.plot(forecast, label='Prediction')
# # plt.title('ARIMA Model - Prediction vs Actual')
# # plt.xlabel('Date')
# # plt.ylabel('Close Price')
# # plt.legend()
# # plt.show()


# # mse = mean_squared_error(test_data, forecast)
# # rmse = np.sqrt(mse)
# # print(f'Root Mean Squared Error: {rmse}')

# # plt.figure(figsize=(12, 6))
# # plt.plot(train_data, label='Train Data')
# # plt.plot(test_data, label='Test Data')
# # plt.plot(forecast, label='Prediction')
# # plt.title('ARIMA Model - Prediction vs Actual')
# # plt.xlabel('Date')
# # plt.ylabel('Close Price')
# # plt.legend()
# # plt.show()

# # Doing Predictions using LSTM Model
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LeakyReLU

print("Column names:", data.columns)

data = data[['Close']]
data = data.dropna()

# Convert the dataframe to a numpy array
dataset = data.values

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Split the data into training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create the datasets with time steps
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# Define the model
model = Sequential()

# First LSTM layer with Leaky ReLU
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LeakyReLU(alpha=0.01))

# Second LSTM layer with Leaky ReLU
model.add(LSTM(50, return_sequences=False))
model.add(LeakyReLU(alpha=0.01))

# Dense layers with Leaky ReLU
model.add(Dense(25))
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, batch_size=1, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stop])

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
print(test_predict)

train_rmse = np.sqrt(np.mean(((train_predict - scaler.inverse_transform([y_train]))**2)))
test_rmse = np.sqrt(np.mean(((test_predict - scaler.inverse_transform([y_test]))**2)))

print('Train RMSE: ', train_rmse)
print('Test RMSE: ', test_rmse)


# plt.figure(figsize=(12, 6))
# plt.plot(train_data, label='Train Data')
# plt.plot(test_data, label='Test Data')
# plt.plot(test_predict, label='Prediction')
# plt.title('LSTM Model - Prediction vs Actual')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.legend()
# plt.show()

# Generate predictions for the next 30 months
future_steps = 2 * 20  # 2 months * 20 trading days per month = 40 steps

# Start with the last observed data
last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
# print(last_sequence.shape)

future_predictions = []
for _ in range(future_steps):
    next_prediction = model.predict(last_sequence)
    # print(next_prediction.shape)
    future_predictions.append(next_prediction[0, 0])
    
    # Update the last_sequence with the new prediction
    next_prediction = np.array(next_prediction).reshape(1, 1, 1)  
    last_sequence = np.append(last_sequence[:, 1:, :], next_prediction, axis=1)
# Transform predictions back to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# print(future_predictions.shape)

# Generate dates for future predictions
last_date = '28-11-2018'
future_dates =  pd.date_range(start=last_date, periods=40, freq='M')
# # print(future_dates)
# print(future_dates.shape)

# Create a DataFrame with future predictions
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted': future_predictions.flatten()
})

# Save the future predictions to a CSV file
future_df.to_csv('future_stock_predictions.csv', index=False)

# print('Future predictions saved to future_stock_predictions.csv')
plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_predictions, marker='o', linestyle='-')
plt.title('Future Predictions')
plt.xlabel('Date')
plt.ylabel('Predictions')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




