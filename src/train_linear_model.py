import numpy as np
import pandas as pd
from preprocessing import z_score_scaling
from sklearn.model_selection import train_test_split
from linear_model import LinearRegressionScratch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DATA_PATH = "data/credit_data.csv"

FEATURES = [
    "LIMIT_BAL",
    "PAY_0",
    "BILL_AMT1",
    "PAY_AMT1"
]


def main():

    df = pd.read_csv(DATA_PATH)

    df["OUTSTANDING_LOSS"] = np.maximum(0, df["BILL_AMT1"] - df["PAY_AMT1"])

    X = df[FEATURES].values
    y = df["OUTSTANDING_LOSS"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled, X_test_scaled, mean, std = z_score_scaling(X_train, X_test)

    print("Training samples:", X_train_scaled.shape)

    model = LinearRegressionScratch(
        learning_rate=0.01,
        num_iterations=3000,
        lambda_=0.1
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n===== Linear Model Evaluation =====")
    print(f"Test MSE: {mse:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    print("\nModel Coefficients (Feature Importance):")
    for feature, weight in zip(FEATURES, model.w):
        print(f"{feature}: {weight:.4f}")


if __name__ == "__main__":
    main()