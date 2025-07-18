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

