#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

#################################### Preprocess the Data ######################################
# Reading from csv file
train_df = pd.read_csv('train.csv')

# Deal with the overall sale of items in all of the stores, 
# hence disregard the columns representing the Store ID and Item ID:
df = train_df.drop(['store','item'], axis=1)

# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'])

# Train models to predict the sales in the next month.
# So, we need to convert our date to a period of ‘Month’ 
# and then sum the number of items sold in each month
df['date'] = df['date'].dt.to_period('M')
monthly_sales = df.groupby('date').sum().reset_index()
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()

# Making data stationary
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()

# Setting dataset for training
supervised_data = monthly_sales.drop(['date','sales'], axis=1)
# Considering previous 12 months' data for prediction
for i in range(1,13):
    col_name = 'month_' + str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)

train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
print('Train Data Shape:', train_data.shape)
print('Test Data Shape:', test_data.shape)

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

X_train, y_train = train_data[:,1:], train_data[:,0:1]
X_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()

sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)

act_sales = monthly_sales['sales'][-13:].to_list()

############################################## Linear Regression ############################################
# Model
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)
linreg_pred = linreg_model.predict(X_test)

# Converting predictions back to original scale
linreg_pred = linreg_pred.reshape(-1,1)
linreg_pred_test_set = np.concatenate([linreg_pred,X_test], axis=1)
linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)

# Converting unscaled sale difference predictions to item sale predictions 
# and append the result to predict data frame
result_list = []
for index in range(0, len(linreg_pred_test_set)):
    result_list.append(linreg_pred_test_set[index][0] + act_sales[index])
linreg_pred_series = pd.Series(result_list,name='linreg_pred')
predict_df = predict_df.merge(linreg_pred_series, left_index=True, right_index=True)

# Performance Analysis
linreg_rmse = np.sqrt(mean_squared_error(predict_df['linreg_pred'], monthly_sales['sales'][-12:]))
linreg_mae = mean_absolute_error(predict_df['linreg_pred'], monthly_sales['sales'][-12:])
linreg_r2 = r2_score(predict_df['linreg_pred'], monthly_sales['sales'][-12:])
print('Linear Regression RMSE: ', linreg_rmse)
print('Linear Regression MAE: ', linreg_mae)
print('Linear Regression R2 Score: ', linreg_r2)

# Learning Curve Plot
train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_sizes_abs = (np.linspace(0.1, 1.0, 10) * len(X_train)).astype(int)
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_scores_mean, label='Training Error', marker='o')
plt.errorbar(train_sizes_abs, test_scores_mean, label='Testing Error', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve for Linear Regression')
plt.legend()
plt.grid()

############################################## XGBoost ############################################
# Model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Converting predictions back to original scale
xgb_pred = xgb_pred.reshape(-1,1)
xgb_pred_test_set = np.concatenate([xgb_pred,X_test], axis=1)
xgb_pred_test_set = scaler.inverse_transform(xgb_pred_test_set)

# Converting unscaled sale difference predictions to item sale predictions 
# and append the result to predict data frame
result_list = []
for index in range(0, len(xgb_pred_test_set)):
    result_list.append(xgb_pred_test_set[index][0] + act_sales[index])
xgb_pred_series = pd.Series(result_list, name='xgb_pred')
predict_df = predict_df.merge(xgb_pred_series, left_index=True, right_index=True)

# Performance Analysis
xgb_rmse = np.sqrt(mean_squared_error(predict_df['xgb_pred'], monthly_sales['sales'][-12:]))
xgb_mae = mean_absolute_error(predict_df['xgb_pred'], monthly_sales['sales'][-12:])
xgb_r2 = r2_score(predict_df['xgb_pred'], monthly_sales['sales'][-12:])
print('XG Boost MSE: ', xgb_rmse)
print('XG Boost MAE: ', xgb_mae)
print('XG Boost R2 Score: ', xgb_r2)

# Learning Curve Plot
train_sizes, train_scores, test_scores = learning_curve(
    XGBRegressor(), X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_sizes_abs = (np.linspace(0.1, 1.0, 10) * len(X_train)).astype(int)
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_scores_mean, label='Training Error', marker='o')
plt.errorbar(train_sizes_abs, test_scores_mean, label='Testing Error', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve for XGBoost')
plt.legend()
plt.grid()

############################################## Random Forest ############################################
# Model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Converting predictions back to original scale
rf_pred = rf_pred.reshape(-1,1)
rf_pred_test_set = np.concatenate([rf_pred,X_test], axis=1)
rf_pred_test_set = scaler.inverse_transform(rf_pred_test_set)

# Converting unscaled sale difference predictions to item sale predictions 
# and append the result to predict data frame
result_list = []
for index in range(0, len(rf_pred_test_set)):
    result_list.append(rf_pred_test_set[index][0] + act_sales[index])
rf_pred_series = pd.Series(result_list, name='rf_pred')
predict_df = predict_df.merge(rf_pred_series, left_index=True, right_index=True)

rf_rmse = np.sqrt(mean_squared_error(predict_df['rf_pred'], monthly_sales['sales'][-12:]))
rf_mae = mean_absolute_error(predict_df['rf_pred'], monthly_sales['sales'][-12:])
rf_r2 = r2_score(predict_df['rf_pred'], monthly_sales['sales'][-12:])
print('Random Forest RMSE: ', rf_rmse)
print('Random Forest MAE: ', rf_mae)
print('Random Forest R2 Score: ', rf_r2)

# Learning Curve Plot
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestRegressor(), X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_sizes_abs = (np.linspace(0.1, 1.0, 10) * len(X_train)).astype(int)
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_scores_mean, label='Training Error', marker='o')
plt.errorbar(train_sizes_abs, test_scores_mean, label='Testing Error', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve for Random Forest')
plt.legend()
plt.grid()

############################################ Model Comparison ##########################################
# Comparison Plot
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.plot(predict_df['date'], predict_df['linreg_pred'])
plt.plot(predict_df['date'], predict_df['xgb_pred'])
plt.plot(predict_df['date'], predict_df['rf_pred'])
plt.title("Customer Sales Forecast Model Comparisons")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Actual Sales", "Predicted Sales using Linear Regression", "Predicted Sales using XGBoost", "Predicted Sales using Random Forest"])
plt.grid()

linreg_stats = [linreg_rmse, linreg_mae, linreg_r2]
xgb_stats = [xgb_rmse, xgb_mae, xgb_r2]
rf_stats = [rf_rmse, rf_mae, rf_r2]
plt.figure(figsize=(15,7))
plt.plot(linreg_stats)
plt.plot(xgb_stats)
plt.plot(rf_stats)
plt.title("Model Comparison between Linear Regression, XGBoost, and Random Forest")
plt.xticks([0,1,2], labels=['RMSE','MAE','R2 Score'])
plt.legend(["Linear Regression", "XGBoost", "Random Forest"])
plt.grid()

plt.show()
