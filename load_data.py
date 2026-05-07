import pandas as pd

data = pd.read_csv("FA-KES-Dataset.csv")

print(data.head())
print(data.columns)
print(data.shape)