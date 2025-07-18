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

print(data.head())