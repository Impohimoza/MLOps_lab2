import numpy as np
import pandas as pd
import os

np.random.seed(42)

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)


def generate_sales_data(n, add_anomalies=False):
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
    
    if add_anomalies:
        n_anomalies = int(n * 0.1)  # 10% аномалий
        
        # Резкие скачки продаж
        anomaly_idx = np.random.choice(n, n_anomalies, replace=False)
        sales[anomaly_idx] = sales[anomaly_idx] * np.random.uniform(2, 5, n_anomalies)
        
        # Отрицательные продажи
        anomaly_idx2 = np.random.choice(n, n_anomalies//2, replace=False)
        sales[anomaly_idx2] = -np.abs(sales[anomaly_idx2])
        
        # Выбросы в рекламном бюджете
        advertising[anomaly_idx] = advertising[anomaly_idx] * np.random.uniform(3, 10, n_anomalies)

    df = pd.DataFrame({
        "day_of_week": day_of_week,
        "price": price,
        "advertising": advertising,
        "sales": sales
    })

    return df


for i in range(5):
    df = generate_sales_data(200, add_anomalies=False)
    df.to_csv(f"data/train/train_{i}.csv", index=False)


for i in range(2):
    df = generate_sales_data(200, add_anomalies=True)
    df.to_csv(f"data/test/test_{i}.csv", index=False)

print("Datasets created")