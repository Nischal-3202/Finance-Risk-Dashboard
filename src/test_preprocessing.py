from preprocessing import load_data, split_features_target, train_test_split_data, z_score_scaling

DATA_PATH = "data/credit_data.csv"

def main():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    X_train_scaled, X_test_scaled, mean, std = z_score_scaling(X_train, X_test)

    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
    print("Number of features:", X_train_scaled.shape[1])

if __name__ == "__main__":
    main()