import os

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error


model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

test_path = "data/test"

X_all = []
y_all = []

for file in os.listdir(test_path):
    df = pd.read_csv(os.path.join(test_path, file))
    X_all.append(df[["day_of_week", "price", "advertising"]])
    y_all.append(df["sales"])

X_test = pd.concat(X_all)
y_test = pd.concat(y_all)

X_scaled = scaler.transform(X_test)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=["0", "1", "2"]
)

pred = model.predict(X_scaled)

mse = mean_squared_error(y_test, pred)

print("Test MSE:", mse)
