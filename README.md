# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load & Split Data – Fetch California Housing dataset, select 3 features as 𝑋 and 2 targets as 𝑌, then split into train & test sets.
2. Standardize Data – Apply StandardScaler to normalize 𝑋 and 𝑌 for better SGD performance.
3. Train Model – Use SGDRegressor inside MultiOutputRegressor and fit it on training data.
4. Make Predictions – Predict target values for 𝑋_test and inverse transform to get original values.
5. Evaluate Model – Compute Mean Squared Error (MSE) and print.

## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Lathikeshwaran J
RegisterNumber: 212222230072
```
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data.data[:, :3]
Y = np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter = 1000, tol = 1e-3)

multi_output_regressor = MultiOutputRegressor(sgd)
multi_output_regressor.fit(X_train,Y_train)

#predict on test data
Y_pred = multi_output_regressor.predict(X_test)

Y_test = scaler_Y.inverse_transform(Y_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)

mse = mean_squared_error(Y_test,Y_pred)
print(f"Mean Squared Error: {mse}")

print("\nPredicted Values:",Y_pred[:5])

```

## Output:
![364864523-dae860d2-b4c3-4619-8c09-43a6c00e3d7e](https://github.com/user-attachments/assets/a78bebd7-2196-4cc4-b230-e0f0c91453b4)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
