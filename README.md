# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries and load the dataset.
2. Split data into training and testing sets.
3. Train the Linear Regression model and make predictions.
4. Plot results and calculate error metrics (MSE, MAE, RMSE)

 
 
 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: jagadheesh kumar T
RegisterNumber:  212225040139
*/
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Load the dataset
df = pd.read_csv("student_scores.csv")
# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())
# Display the last few rows of the dataset
print("Last 5 rows of the dataset:")
print(df.tail())
# Separate the independent (X) and dependent (Y) variables
X = df.iloc[:, :-1].values # Assuming the 'Hours' column is the first column
Y = df.iloc[:, 1].values
# Assuming the 'Scores' column is the second column
# Split the dataset into training and testing sets (1/3rd for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3,
random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
print("Predicted values:")
print(Y_pred)
print("Actual values:")
print(Y_test)
plt.scatter(X_train, Y_train, color="red", label="Actual Scores")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Fitted Line")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()
# Plot the Testing set results
plt.scatter(X_test, Y_test, color='green', label="Actual Scores")
plt.plot(X_train, regressor.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.show()
# Calculate and print error metrics
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print('Mean Squared Error (MSE) =', mse)
print('Mean Absolute Error (MAE) =', mae)
print('Root Mean Squared Error (RMSE) =', rmse)


```

## Output:
<img width="604" height="672" alt="Screenshot 2026-02-09 104300" src="https://github.com/user-attachments/assets/52bb9dbb-baf8-43f3-927f-ff488ea63542" />

<img width="604" height="672" alt="Screenshot 2026-02-09 104300 - Copy" src="https://github.com/user-attachments/assets/98152e4d-00e7-41f3-9664-745d56535989" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
