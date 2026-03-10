import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


train_path = "data/train"

data = []

for file in os.listdir(train_path):
    df = pd.read_csv(os.path.join(train_path, file))
    data.append(df)

data = pd.concat(data)

X = data[["day_of_week", "price", "advertising"]]
y = data["sales"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")

pd.DataFrame(X_scaled).to_csv("data/train/X_train_scaled.csv", index=False)
y.to_csv("data/train/y_train.csv", index=False)

print("Preprocessing finished")
