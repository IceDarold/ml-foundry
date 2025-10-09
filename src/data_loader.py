import pandas as pd
import pickle

def load_data(train_path, test_path, target_col):
    """
    Load train and test DataFrames using pandas, separate X and y from train, return X, y, X_test.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y = train_df[target_col]
    X = train_df.drop(target_col, axis=1)
    X_test = test_df

    return X, y, X_test

def save_processed_data(df, path):
    """
    Save a DataFrame to pickle for faster loading.
    """
    df.to_pickle(path)

def load_processed_data(path):
    """
    Load a DataFrame from pickle.
    """
    return pd.read_pickle(path)