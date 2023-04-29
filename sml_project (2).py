import pandas as pd

# Read in the dataset
df = pd.read_csv("/content/Steel_industry_data.csv")

# Convert the "date" column to a datetime data type
df["date"] = pd.to_datetime(df["date"])

# Group the data by the date and calculate the mean for each day
daily_mean = df.groupby(df["date"].dt.date).mean()

# Save the daily mean as a csv file
daily_mean.to_csv("daily_mean.csv", index=True)

df.head()

daily_mean

from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(daily_mean)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(daily_mean.drop('Usage_kWh', axis=1), daily_mean['Usage_kWh'], test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
print('Linear Regression RMSE:', mean_squared_error(y_test, y_pred, squared=False))

# Train Ridge regression
ridge = Ridge(alpha=0.1)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)
print('Ridge Regression RMSE:', mean_squared_error(y_test, y_pred, squared=False))

# Train Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
print('Lasso Regression RMSE:', mean_squared_error(y_test, y_pred, squared=False))

# Train SVM regression
svr = SVR(C=1.0, epsilon=0.2)
svr.fit(X_train_scaled, y_train)
y_pred = svr.predict(X_test_scaled)
print('SVM Regression RMSE:', mean_squared_error(y_test, y_pred, squared=False))

# Train Decision Tree regression
dtr = DecisionTreeRegressor(max_depth=5, random_state=42)
dtr.fit(X_train_scaled, y_train)
y_pred = dtr.predict(X_test_scaled)
print('Decision Tree RMSE:', mean_squared_error(y_test, y_pred, squared=False))

# Train Random Forest regression
rfr = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rfr.fit(X_train_scaled, y_train)
y_pred = rfr.predict(X_test_scaled)
print('Random Forest RMSE:', mean_squared_error(y_test, y_pred, squared=False))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(daily_mean.drop('Usage_kWh', axis=1), daily_mean['Usage_kWh'], test_size=0.2, random_state=42)

# Train linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_rmse = mean_squared_error(y_test, y_pred_lr, squared=False)
lr_r2 = r2_score(y_test, y_pred_lr)

# Train decision tree regressor model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_rmse = mean_squared_error(y_test, y_pred_dt, squared=False)
dt_r2 = r2_score(y_test, y_pred_dt)

# Train random forest regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
rf_r2 = r2_score(y_test, y_pred_rf)

# Print results
print('Linear Regression RMSE:', lr_rmse)
print('Linear Regression R2 Score:', lr_r2)
print('Decision Tree Regressor RMSE:', dt_rmse)
print('Decision Tree Regressor R2 Score:', dt_r2)
print('Random Forest Regressor RMSE:', rf_rmse)
print('Random Forest Regressor R2 Score:', rf_r2)

# Plot predicted vs actual values for random forest regressor
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Actual Usage_kWh')
plt.ylabel('Predicted Usage_kWh')
plt.title('Random Forest Regressor: Predicted vs Actual Usage_kWh')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Splitting the data into training and testing sets
train_size = int(0.8 * len(daily_mean))
train_X, train_y = daily_mean.iloc[:train_size]['Usage_kWh'], daily_mean.iloc[:train_size]['CO2(tCO2)']
test_X, test_y = daily_mean.iloc[train_size:]['Usage_kWh'], daily_mean.iloc[train_size:]['CO2(tCO2)']

# Reshaping the data
train_X = np.array(train_X).reshape(-1, 1)
train_y = np.array(train_y).reshape(-1, 1)
test_X = np.array(test_X).reshape(-1, 1)
test_y = np.array(test_y).reshape(-1, 1)

# Training the linear regression model
lr_model = LinearRegression()
lr_model.fit(train_X, train_y)

# Training the decision tree regression model
dt_model = DecisionTreeRegressor(max_depth=5)
dt_model.fit(train_X, train_y)

# Predicting on the test set using the trained models
lr_pred = lr_model.predict(test_X)
dt_pred = dt_model.predict(test_X)

# Plotting the predicted values against the actual values for the linear regression model
plt.figure(figsize=(10, 6))
plt.scatter(test_X, test_y, color='blue', label='Actual')
plt.plot(test_X, lr_pred, color='red', label='Predicted (Linear Regression)')
plt.xlabel('Usage_kWh')
plt.ylabel('CO2(tCO2)')
plt.legend()
plt.show()

# Plotting the predicted values against the actual values for the decision tree regression model
plt.figure(figsize=(10, 6))
plt.scatter(test_X, test_y, color='blue', label='Actual')
plt.plot(test_X, dt_pred, color='green', label='Predicted (Decision Tree Regression)')
plt.xlabel('Usage_kWh')
plt.ylabel('CO2(tCO2)')
plt.legend()
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print('RMSE:', rmse)
print('R^2 score:', r2)

from sklearn.model_selection import train_test_split

X = daily_mean.drop(['Usage_kWh'], axis=1)
y = daily_mean['Usage_kWh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt

# create a list of algorithms and their accuracy scores
algorithms = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']
accuracy_scores = [0.994, 1.0 , 0.996, 0.67]

# plot the bar chart
plt.bar(algorithms, accuracy_scores)
plt.ylim(0.6, 1.2) # set the y-axis limits
plt.title('Accuracy Scores of SML Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Read in the dataset
df = pd.read_csv("Steel_industry_data.csv")

# Convert the "date" column to a datetime data type
df["date"] = pd.to_datetime(df["date"])

# Group the data by the date and calculate the mean for each day
daily_mean = df.groupby(df["date"].dt.date).mean()

# Define the features and target variable
X = daily_mean.drop(columns=["Usage_kWh"])
y = daily_mean["Usage_kWh"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred, squared=False))

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Print results
print("RMSE:", rmse)
print("R2 Score:", r2)

import numpy as np

# Create a random array with the same shape as X_test
X_new = np.array([[4.285	,7.574,	0.008587,	79.74855,	98.8555,	42750.0]])

# Predict the usage using the trained random forest model
y_new_pred = rf_model.predict(X_new)

# Print the predicted usage
print(y_new_pred)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Read in the dataset
df = pd.read_csv("Steel_industry_data.csv")

# Convert the "date" column to a datetime data type
df["date"] = pd.to_datetime(df["date"])

# Group the data by the date and calculate the mean for each day
daily_mean = df.groupby(df["date"].dt.date).mean().reset_index()

# Select the variables for the model
X = daily_mean[['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
                'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM']]
y = daily_mean['Usage_kWh']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and test the models
models = {'Linear Regression': LinearRegression(),
          'Decision Tree': DecisionTreeRegressor(),
          'Random Forest': RandomForestRegressor(),
          'Support Vector Machine': SVR(),
          'Neural Network': MLPRegressor()}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results.append((name, rmse, r2))

# Create a streamlit app
st.title('Predicting Usage_kWh in Steel Industry')
st.write('Select a model to use:')

# Create a dropdown to select a model
options = [result[0] for result in results]
selected_model = st.selectbox('Choose a model', options)

# Show the RMSE and R2 score for the selected model
for result in results:
    if result[0] == selected_model:
        st.write('RMSE:', result[1])
        st.write('R2 score:', result[2])

# Predict the value of Usage_kWh using the selected model and user input
input_values = {'Lagging_Current_Reactive.Power_kVarh': st.number_input('Enter Lagging_Current_Reactive.Power_kVarh'),
                'Leading_Current_Reactive_Power_kVarh': st.number_input('Enter Leading_Current_Reactive_Power_kVarh'),
                'CO2(tCO2)': st.number_input('Enter CO2(tCO2)'),
                'Lagging_Current_Power_Factor': st.number_input('Enter Lagging_Current_Power_Factor'),
                'Leading_Current_Power_Factor': st.number_input('Enter Leading_Current_Power_Factor'),
                'NSM': st.number_input('Enter NSM')}

input_df = pd.DataFrame(input_values, index=[0])
selected_model_obj = [result[1] for result in results if result[0] == selected_model][0]
selected_model_obj.fit(X, y)
predicted_value = selected_model_obj.predict(input_df)[0]

# Show the predicted value
st.write('The predicted value of Usage_kWh is', predicted_value)
