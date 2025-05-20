 Implementation-of-Logistic-Regression-Using-Gradient-Descent

#### AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

#### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

#### Algorithm
1.Import the packages required.
2.Read the dataset. 
3.Define X and Y array. 
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value.

#### Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Keziah.F
RegisterNumber:  212223040094
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1) 
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
#### Output:

## DATASET
![image](https://github.com/user-attachments/assets/229b1674-dd3f-46f5-b302-00a1896c4dcf)

## DATATYPES OF DATASET

![image](https://github.com/user-attachments/assets/d5b7519e-9d16-40bb-b5e7-0b9934c289f5)

## LABELED DATASET

![image](https://github.com/user-attachments/assets/bf10dcba-b3c7-46e2-a08a-2be9f39cbc56)

## Y VALUE(DEPENDENT VARIABLE)

![image](https://github.com/user-attachments/assets/a3f7bcce-577a-46b4-83a7-877ce1cbf9ab)

## ACCURACY

![image](https://github.com/user-attachments/assets/36cbc430-e754-4057-8ff0-7c96215b451c)

## Y VALUE

![image](https://github.com/user-attachments/assets/412d1da1-f5d1-4128-8e75-f993657d1b27)

## NEW Y PREDICTIONS

![image](https://github.com/user-attachments/assets/8c3fb8ca-4371-4fd0-8d3c-d1bed5fba93f)





#### Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

