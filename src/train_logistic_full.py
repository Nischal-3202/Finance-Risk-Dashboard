import numpy as np
from preprocessing import load_data, split_features_target, train_test_split_data, z_score_scaling
from logistic_model import LogisticRegressionScratch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

DATA_PATH = "data/credit_data.csv"

def main():

    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    X_train_scaled, X_test_scaled, mean, std = z_score_scaling(X_train, X_test)

    print("Training samples:", X_train_scaled.shape)
    print("Testing samples:", X_test_scaled.shape)

    model = LogisticRegressionScratch(
        learning_rate=0.01,
        num_iterations=1000,
        lambda_=0.1
    )

    model.fit(X_train_scaled, y_train)

    y_probs = model.predict_proba(X_test_scaled)
    y_pred = (y_probs >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print("\nTest Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    print("\nROC-AUC Score:", auc_score)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Full Feature Model")
    plt.show()

    print("\nTesting different thresholds:\n")

    for threshold in [0.5, 0.4, 0.3, 0.2]:
        y_temp = (y_probs >= threshold).astype(int)
        acc = accuracy_score(y_test, y_temp)
        print(f"Threshold {threshold} -> Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()