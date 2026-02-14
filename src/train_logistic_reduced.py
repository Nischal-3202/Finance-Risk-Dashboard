import numpy as np
import pandas as pd
from preprocessing import train_test_split_data, z_score_scaling
from logistic_model import LogisticRegressionScratch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

DATA_PATH = "data/credit_data.csv"

REDUCED_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "BILL_AMT1",
    "PAY_AMT1"
]

TARGET = "default.payment.next.month"


def main():

    df = pd.read_csv(DATA_PATH)

    X = df[REDUCED_FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    X_train_scaled, X_test_scaled, mean, std = z_score_scaling(X_train, X_test)

    print("Training samples:", X_train_scaled.shape)
    print("Testing samples:", X_test_scaled.shape)

    print("\n==============================")
    print("Training Reduced Feature Model")
    print("==============================")

    model = LogisticRegressionScratch(
        learning_rate=0.01,
        num_iterations=3000,
        lambda_=0.1
    )

    model.fit(X_train_scaled, y_train)

    y_probs = model.predict_proba(X_test_scaled)
    DEFAULT_THRESHOLD = 0.35
    y_pred = (y_probs >= DEFAULT_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    print("\nTest Accuracy:", accuracy)
    print("ROC-AUC:", auc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()