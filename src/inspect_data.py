import pandas as pd

DATA_PATH = "data/credit_data.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    print("✅ Dataset Loaded Successfully\n")

    print("Shape of dataset (rows, columns):")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nChecking for missing values:")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()