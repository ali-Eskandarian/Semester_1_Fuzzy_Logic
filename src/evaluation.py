import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler


def evaluate_model(regressor, data, input_columns, output_column, fuzzy_type='type1', scale="Minmax", num_iterations=2):
    best_rmse_result = float('inf')
    best_r2_score = float('-inf')
    avg_rmse = 0
    for _ in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(data[input_columns], data[output_column], test_size=0.2, random_state=None)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        sk_regressor = XGBRegressor()
        sk_regressor.fit(X_train, y_train)

        if scale == "Minmax":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif scale == "Standard":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif scale == "Robust":
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train, columns=input_columns)
        X_test = pd.DataFrame(X_test, columns=input_columns)
        
        regressor.train(X_train, y_train)
        y_pred = regressor.predict(X_test)
        if fuzzy_type == 'type2':
            y_pred = [(pred[0] + pred[1])/2 for pred in y_pred]
            try:
                y_pred = np.where(np.isnan(y_pred), np.nanmean(y_pred[~np.isnan(y_pred)]), y_pred)
            except:
                y_pred = np.where(np.isnan(y_pred), 10, y_pred)

        r2_score = 1 - (mean_squared_error(y_test, y_pred) / abs(np.var(y_test))) 
        current_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        avg_rmse += current_rmse
        if current_rmse < best_rmse_result:
            best_rmse_result = current_rmse
            best_predictions = y_pred
            sk_pred = sk_regressor.predict(X_test)
            best_actuals = y_test
            
        if r2_score > best_r2_score:
            best_r2_score = r2_score

    return best_rmse_result, best_predictions, best_actuals, regressor.rules, regressor.fuzzy_vars, avg_rmse/num_iterations, sk_pred, best_r2_score 