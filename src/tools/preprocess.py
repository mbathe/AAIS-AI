import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    """
    Preprocess a given dataframe by handling missing values, converting date columns to datetime format.

    Parameters:
    df (pandas.DataFrame): The input dataframe to preprocess.

    Returns:
    pandas.DataFrame: The preprocessed dataframe with missing values handled and date columns converted to datetime format.
    """
    anomalies = {}
    df_clean = df.copy()

    missing_values = df_clean.isnull().sum()

    anomalies["number_of_missing_values"] = missing_values
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean = df_clean.dropna()

    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col == 'tif_path' or col == 'dataset':
            continue
        if df_clean[col].isnull().sum() > 0:
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value)

    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col == 'tif_path' or col == 'dataset':
            continue
        if df_clean[col].isnull().sum() > 0:
            mode_value = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_value)

    # Convert date columns to datetime format
    date_columns = ['SDate', 'HDate']
    for col in date_columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        df_clean[f'{col}_year'] = df_clean[col].dt.year
        df_clean[f'{col}_month'] = df_clean[col].dt.month
        df_clean[f'{col}_day'] = df_clean[col].dt.day

    return df_clean
