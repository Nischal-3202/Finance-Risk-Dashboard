import numpy as np
import pandas as pd
from preprocessing import z_score_scaling
from sklearn.model_selection import train_test_split
from logistic_model import LogisticRegressionScratch
from linear_model import LinearRegressionScratch

DATA_PATH = "data/credit_data.csv"

LOGISTIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "BILL_AMT1",
    "PAY_AMT1"
]

LINEAR_FEATURES = [
    "LIMIT_BAL",
    "PAY_0",
    "BILL_AMT1",
    "PAY_AMT1"
]

TARGET = "default.payment.next.month"
DEFAULT_THRESHOLD = 0.35


def main():

    df = pd.read_csv(DATA_PATH)

    X_log = df[LOGISTIC_FEATURES].values
    y_log = df[TARGET].values

    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
    )

    X_train_log_scaled, X_test_log_scaled, mean_log, std_log = z_score_scaling(
        X_train_log, X_test_log
    )

    logistic_model = LogisticRegressionScratch(
        learning_rate=0.01,
        num_iterations=3000,
        lambda_=0.1
    )

    logistic_model.fit(X_train_log_scaled, y_train_log)

    df["OUTSTANDING_LOSS"] = np.maximum(0, df["BILL_AMT1"] - df["PAY_AMT1"])

    X_lin = df[LINEAR_FEATURES].values
    y_lin = df["OUTSTANDING_LOSS"].values

    X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
        X_lin, y_lin, test_size=0.2, random_state=42
    )

    X_train_lin_scaled, X_test_lin_scaled, mean_lin, std_lin = z_score_scaling(
        X_train_lin, X_test_lin
    )

    linear_model = LinearRegressionScratch(
        learning_rate=0.01,
        num_iterations=3000,
        lambda_=0.1
    )

    linear_model.fit(X_train_lin_scaled, y_train_lin)
    np.savez(
        "models/credit_risk_model.npz",
        w_logistic=logistic_model.w,
        b_logistic=logistic_model.b,
        mean_logistic=mean_log,
        std_logistic=std_log,
        logistic_features=LOGISTIC_FEATURES,
        threshold=DEFAULT_THRESHOLD,
        w_linear=linear_model.w,
        b_linear=linear_model.b,
        mean_linear=mean_lin,
        std_linear=std_lin,
        linear_features=LINEAR_FEATURES
    )

    print("\nModels exported successfully to models/credit_risk_model.npz")


if __name__ == "__main__":
    main()