import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import the data
data = {
    "Size": [850, 900, 1000, 1200, 1500, 800, 1100, 950, 1300, 1400],
    "Bedrooms": [2, 2, 3, 3, 4, 2, 3, 2, 4, 3],
    "Age": [20, 15, 10, 5, 1, 25, 8, 12, 3, 2],
    "DistanceToCityCenter": [8.0, 7.0, 6.0, 5.0, 4.0, 10.0, 6.5, 7.5, 4.5, 4.0],
    "Price": [105, 115, 130, 150, 185, 95, 140, 120, 165, 175]
}

df = pd.DataFrame(data)

#training set, validation set, and testing set as 80%, 10%, and 10%
data_len = len(df)
eighty_pct_index = int((80/100)*data_len)
ten_pct_index = int((10/100)*data_len)
training_set = df[:eighty_pct_index].copy()
validation_set = df[eighty_pct_index:eighty_pct_index+ten_pct_index].copy()
testing_set = df[eighty_pct_index+ten_pct_index:].copy()


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

def predict(X, w, b):
    f_wb = X @ w + b
    return f_wb


# take the shape
# y_train = training_set["price"]
# training_set.drop('price', axis=1, inplace=True)

# define some nums
m, n = training_set.shape
w = np.zeros(n)
b = 0
num_iters = 5000
alpha = 0.01

# # plot the feature
# cols = training_set.columns
# # plt.show()
# for i in range(len(cols)):
#     plt.figure(i)
#     plt.scatter(training_set[cols[i]], y_train)
#     plt.xlabel(cols[i])
# plt.show()

# w, b = gradient_descent(training_set, y_train, w, b, alpha, num_iters)
# predicted_value = predict(training_set, w, b)






