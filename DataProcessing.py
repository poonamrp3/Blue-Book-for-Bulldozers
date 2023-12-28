import pandas as pd

def extract_date_features(df):
    """
    Extracts date-related features from the 'saledate' column.

    Parameters:
    - df (DataFrame): Input DataFrame with 'saledate' column.

    Returns:
    None (modifies df in place).
    """
    df["saleyear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayofWeek"] = df.saledate.dt.dayofweek
    df["saleDayofYear"] = df.saledate.dt.dayofyear
    df.drop("saledate", axis=1, inplace=True)

def fill_missing_numeric_values(df):
    """
    Fills missing values in numeric columns with the median.

    Parameters:
    - df (DataFrame): Input DataFrame.

    Returns:
    None (modifies df in place).
    """
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label+"_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())

def fill_missing_categorical_values(df):
    """
    Fills missing values in categorical columns by converting to codes.

    Parameters:
    - df (DataFrame): Input DataFrame.

    Returns:
    None (modifies df in place).
    """
    for label, content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_missing"] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes + 1

def preprocess_data(df):
    """
    Combines date feature extraction and missing value filling for a DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: Preprocessed DataFrame.
    """
    extract_date_features(df)
    fill_missing_numeric_values(df)
    fill_missing_categorical_values(df)
    return df
