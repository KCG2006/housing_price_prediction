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
training_set = data[:eighty_pct_index].copy()
validation_set = data[eighty_pct_index:eighty_pct_index+ten_pct_index].copy()
testing_set = data[eighty_pct_index+ten_pct_index:].copy()


#compute the cost function
def compute_cost(X, y, w, b):
    """return total_cost"""
    # for i in range(m):
    #     f_wb = np.dot(w, X.iloc[i]) + b
    #     loss = (f_wb - y.iloc[i])**2
    #     total_cost += loss
    f_wb = X @ w + b
    total_cost = np.sum(((f_wb - y)**2)/(2*m))
    return total_cost

#compute the gradient
def compute_gradient(X, y, w, b):
    """return dj_dw, dj_db"""
    # for i in range(m):
    #     f_wb = np.dot(w, X.iloc[i]) + b
    #     error = f_wb - y[i]
    #     for j in range(n):
    #         dj_dw[j] += error * X.iloc[i,j]
    #     dj_db += error
    error = (X @ w + b) - y
    dj_dw = (X.T @ error) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db

# compute the gradient descent:
def gradient_descent(X, y, w, b, alpha, num_iters):
    """return w and b"""
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - (alpha* dj_dw)
        b = b - (alpha* dj_db)
        if (i%1000)==0:
            print(f"Interation {i} |Cost: {compute_cost(X, y, w, b)}")
    return w, b

def predict(X, y, w, b):
    f_wb = X @ w + b
    return f_wb




# take the shape
y_train = training_set["price"]
training_set.drop('price', axis=1, inplace=True)

# define some nums
m, n = training_set.shape
w = np.zeros(n)
b = 0
num_iters = 10000
alpha = 0.01

# SCALE THE AREA
area = training_set['area']
area_mean = np.mean(area)
area_std = np.std(area)
training_set['area'] = (area - area_mean) / area_std
# print(training_set.head())
#Scale the price too!!
y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)
y_train = (y_train - y_train_mean)/y_train_std
# print(y_train.head())

w, b = gradient_descent(training_set, y_train, w, b, alpha, num_iters)


# for i in range(len(X)):
    # print(f"predicted value: {f_wb.iloc[i]} | actual value: {y.iloc[i]}")




