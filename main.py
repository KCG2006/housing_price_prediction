import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"F:\dataset\housing_price_2.csv")
#take columns to change to 1 and 0
cols_to_map = ["mainroad",
               "guestroom",
               "basement",
               "hotwaterheating",
               "airconditioning",
               "prefarea"]
#change yes, no to 1, 0
for i in range(len(cols_to_map)):
    data[cols_to_map[i]] = data[cols_to_map[i]].map({"yes": 1, "no": 0})

#convert categorical columns (more than 2 types) using one hot coding
data[['is_semi','is_unfurnished']] = pd.get_dummies(data["furnishingstatus"], dtype=float, drop_first=True)
data = data.drop(["furnishingstatus"], axis=1)

#training set, validation set, and testing set as 80%, 10%, and 10%
data_len = len(data)
eighty_pct_index = int((80/100)*data_len)
ten_pct_index = int((10/100)*data_len)
training_set = data[:eighty_pct_index]
validation_set = data[eighty_pct_index:eighty_pct_index+ten_pct_index]
testing_set = data[eighty_pct_index+ten_pct_index:]

# take the shape
m, n = data.shape
#compute the cost function
def compute_cost(X, y, w, b):
    """return total_cost"""
    total_cost = 0
    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        loss = (f_wb - y[i])**2
        total_cost += loss
    total_cost /= (2*m)
    return total_cost

#compute the gradient
def compute_gradient(X, y, w, b):
    """return dj_dw, dj_db"""
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        error = f_wb - y[i]
        for j in range(n):
            dj_dw += error * X[i,j]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# compute the gradient descent:
def gradient_descent(X, y, w, b, alpha, num_iters):
    """return w and b"""
    dj_dw, dj_db = compute_gradient(X, y, w, b)
    for i in range(num_iters):
        w = w - (alpha* dj_dw)
        b = b - (alpha* dj_db)
        if (i%1000)==0:
            print(f"Cost: {compute_cost(X, y, w, b)}")
            print(f"w: {w}, b: {b}")
    return w, b

def predict(X, y, w, b):
    val_m = len(X)
    for i in range(val_m):
        f_wb = np.dot(w, X[i]) + b
        print(f"predicted value: {f_wb} | actual value: {y[i]}")

