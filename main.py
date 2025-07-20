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
    m, n = X.shape
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
    m, n = X.shape
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
J_history = []
def gradient_descent(X, y, w, b, alpha, num_iters):
    """return w and b"""
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - (alpha* dj_dw)
        b = b - (alpha* dj_db)
        if (i%1000)==0:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i} |Cost: {cost}")
            J_history.append(cost)
    return w, b

def predict(X, w, b):
    f_wb = X @ w + b
    return f_wb

def standardscaler(col):
    mean = np.mean(col)
    std = np.std(col)
    scaled_col = (col - mean)/std
    return scaled_col, mean, std


# take the shape
y_train = training_set["Price"]
training_set.drop('Price', axis=1, inplace=True)

# define some nums
w = np.zeros(training_set.shape[1])
b = 0
num_iters = 10000
alpha = 0.1

#take the columns
cols = training_set.columns

#scale the features
dct =  {}
for i in range(len(cols)):
    training_set[cols[i]], mean, std = standardscaler(training_set[cols[i]])
    dct[cols[i]] = (mean,std)

# plot the feature
# for i in range(len(cols)):
#     plt.figure(i)
#     plt.scatter(training_set[cols[i]], y_train)
#     plt.xlabel(cols[i])
# plt.show()

w, b = gradient_descent(training_set, y_train, w, b, alpha, num_iters)


# Evaluate model performance on the validation set
y_train = validation_set["Price"]
validation_set.drop('Price', axis=1, inplace=True)
#scale the validation set
validation_dct =  {}
#take the columns
cols = validation_set.columns
for i in range(len(cols)):
    validation_set[cols[i]] = (validation_set[cols[i]] - dct[cols[i]][0])/ dct[cols[i]][1]
predicted_value = predict(validation_set, w, b)

print(f"predicted value: {predicted_value}, actual value: {y_train}")


# plot the predicted value and actual value
# plt.figure(figsize=(8, 6))
# plt.scatter(predicted_value, y_train, color='blue')
# plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
# plt.ylabel('Actual Price ($1000s)')
# plt.xlabel('Predicted Price ($1000s)')
# plt.title('Actual vs Predicted Housing Prices')
# plt.grid(True)
# plt.show()






