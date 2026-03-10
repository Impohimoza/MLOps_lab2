import numpy as np
import pandas as pd
import os

np.random.seed(42)

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)


def generate_sales_data(n):

    day_of_week = np.random.randint(1, 8, n)
    price = np.random.uniform(10, 100, n)
    advertising = np.random.uniform(0, 50, n)

    sales = (
        200
        - 1.5 * price
        + 2.5 * advertising
        + 5 * day_of_week
        + np.random.normal(0, 10, n)
    )

    df = pd.DataFrame({
        "day_of_week": day_of_week,
        "price": price,
        "advertising": advertising,
        "sales": sales
    })

    return df


for i in range(5):
    df = generate_sales_data(200)
    df.to_csv(f"data/train/train_{i}.csv", index=False)


for i in range(2):
    df = generate_sales_data(200)
    df.to_csv(f"data/test/test_{i}.csv", index=False)

print("Datasets created")
