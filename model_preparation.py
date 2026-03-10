import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

X = pd.read_csv("data/train/X_train_scaled.csv")
y = pd.read_csv("data/train/y_train.csv")

model = LinearRegression()

model.fit(X, y.values.ravel())

joblib.dump(model, "models/model.pkl")

print("Model trained")
