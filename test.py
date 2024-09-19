import pandas as pd

file = "dataset_train.csv"

data = pd.read_csv(file)

# print(data)
print(data.describe())