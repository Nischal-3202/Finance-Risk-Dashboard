import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df

def split_features_target(df):
    X = df.drop(columns=["default.payment.next.month"]).values
    y = df["default.payment.next.month"].values
    return X, y


def train_test_split_data(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )


def z_score_scaling(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std