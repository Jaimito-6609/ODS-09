# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:23:22 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 1 ODS 09
#==============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Suppose we have process industry data including variables such as operating costs, production times, maintenance indicators, and sustainability parameters
# Columns: Operating_Cost, Production_Time, Maintenance_Indicator, Sustainability_Parameter, Performance, Profitability

# Load the process industry data
process_data = pd.read_csv('process_industry_data.csv')

# Separate the independent variables (X) from the target variables (y)
X = process_data.drop(['Performance', 'Profitability'], axis=1)
y = process_data[['Performance', 'Profitability']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Predictive maintenance model using a random forest regressor
maintenance_model = RandomForestRegressor(n_estimators=100)
maintenance_model.fit(X_train_scaled, y_train['Maintenance_Indicator'])

# Production planning optimization model
# Define an objective function that models the operating cost and performance
def optimization_objective(optimization_vars):
    operating_cost, production_time = optimization_vars
    # We model performance as inversely proportional to both cost and production time
    performance = 1 / (operating_cost * production_time)
    return -performance  # Negative because we want to maximize
# Production constraints (e.g., limits on costs and production times)
limits = [(1000, 5000),  # Operating costs between 1000 and 5000
          (1, 24)]       # Production time between 1 and 24 hours

# Optimization of production planning
optimization_result = minimize(optimization_objective, x0=[2000, 12], bounds=limits, method='SLSQP')

# Performance and profitability prediction model using regression
prediction_model = RandomForestRegressor(n_estimators=100)
prediction_model.fit(X_train_scaled, y_train)

# Predict performance and profitability on the test set
predictions = prediction_model.predict(X_test_scaled)

# Evaluate the accuracy of the model
mae_performance = mean_absolute_error(y_test['Performance'], predictions[:, 0])
mae_profitability = mean_absolute_error(y_test['Profitability'], predictions[:, 1])

print(f"Production Planning Optimization: Operating Cost = {optimization_result.x[0]}, Production Time = {optimization_result.x[1]}")
print(f"MAE Performance: {mae_performance}, MAE Profitability: {mae_profitability}")

# Note: This code is a basic framework and needs to be adjusted and validated with real data and detailed technical specifications of the production system.
