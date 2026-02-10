import pandas as pd
import numpy as np
import os
import pickle
import json

def validate_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates OHLCV data for completeness, consistency, and correctness.

    Args:
        df (pd.DataFrame): The input DataFrame containing OHLCV data.

    Returns:
        pd.DataFrame: The validated and cleaned DataFrame.

    Raises:
        ValueError: If critical validation checks fail.
    """
    if df.empty:
        raise ValueError("OHLCV DataFrame is empty.")

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values found in OHLCV data. Attempting to fill with forward fill then backward fill.")
        df = df.ffill().bfill() # Simple imputation strategy
        if df.isnull().sum().sum() > 0:
            raise ValueError("Critical missing values remain after imputation.")

    # Check data types and convert if necessary
    for col in ['open', 'high', 'low', 'close']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                raise ValueError(f"Column '{col}' cannot be converted to numeric.")
        if df[col].isnull().any():
             raise ValueError(f"Numeric conversion resulted in NaNs for column '{col}'.")

    if not pd.api.types.is_integer_dtype(df['volume']):
        try:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64') # Use Int64 for nullable integer
        except Exception:
            raise ValueError(f"Column 'volume' cannot be converted to integer.")
        if df['volume'].isnull().any():
             raise ValueError(f"Integer conversion resulted in NaNs for column 'volume'.")


    # Logical inconsistencies
    if (df['low'] > df['high']).any():
        raise ValueError("Inconsistency detected: 'low' price is greater than 'high' price.")
    if (df['close'] < df['low']).any() or (df['close'] > df['high']).any():
        print("Warning: 'close' price outside 'low'-'high' range detected in some rows.")
        # Depending on strictness, one might clean these or flag them more aggressively.
        # For now, we'll just warn.

    if (df['open'] < 0).any() or (df['high'] < 0).any() or (df['low'] < 0).any() or (df['close'] < 0).any():
        raise ValueError("Negative price detected.")
    if (df['volume'] < 0).any():
        raise ValueError("Negative volume detected.")

    print("OHLCV data validated successfully.")
    return df

print("data_processing.py created with validate_ohlcv_data function.")

def normalize_data_for_llm(ohlcv_df: pd.DataFrame, financial_statements: dict, indicators_df: pd.DataFrame) -> str:
    """
    Normalizes OHLCV data, financial statements, and technical indicators into a consistent
    text format suitable for LLM processing.

    Args:
        ohlcv_df (pd.DataFrame): DataFrame containing OHLCV data.
        financial_statements (dict): Dictionary of financial statements (income_statement, balance_sheet, cash_flow).
        indicators_df (pd.DataFrame): DataFrame containing calculated technical indicators.

    Returns:
        str: A comprehensive string summarizing the financial data for LLM analysis.
    """
    normalized_output = []

    if not ohlcv_df.empty:
        normalized_output.append("### Historical OHLCV Data (Last 5 Days):")
        # Select relevant columns and format dates
        ohlcv_summary = ohlcv_df[['open', 'high', 'low', 'close', 'volume']].tail(5)
        ohlcv_summary.index = ohlcv_summary.index.strftime('%Y-%m-%d')
        normalized_output.append(ohlcv_summary.to_markdown(numalign="left", stralign="left"))
        normalized_output.append("\n")

    if not indicators_df.empty:
        normalized_output.append("### Technical Indicators (Last 5 Days):")
        # Select a few key indicators and format dates
        indicators_summary = indicators_df.tail(5)
        indicators_summary.index = indicators_summary.index.strftime('%Y-%m-%d')
        # Try to include common indicators if they exist
        available_indicators = ['sma_20', 'ema_20', 'rsi_14', 'macd', 'signal_line', 'macd_histogram', 'upper_band', 'middle_band', 'lower_band']
        cols_to_include = [col for col in available_indicators if col in indicators_summary.columns]
        if cols_to_include:
            normalized_output.append(indicators_summary[cols_to_include].to_markdown(numalign="left", stralign="left"))
            normalized_output.append("\n")
        else:
            normalized_output.append("No common technical indicators found in the provided DataFrame.\n")

    if financial_statements:
        normalized_output.append("### Financial Statements:")
        for stmt_name, stmt_df in financial_statements.items():
            if not stmt_df.empty:
                normalized_output.append(f"#### {stmt_name.replace('_', ' ').title()}:")
                # Transpose for easier LLM reading, and select latest few periods
                stmt_df_formatted = stmt_df.iloc[:, :3].transpose() # Take top 3 most recent columns and transpose
                stmt_df_formatted.index = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else str(col) for col in stmt_df_formatted.index]
                normalized_output.append(stmt_df_formatted.to_markdown(numalign="left", stralign="left"))
                normalized_output.append("\n")
            else:
                normalized_output.append(f"No {stmt_name.replace('_', ' ').lower()} found.\n")

    if not normalized_output:
        return "No data available for normalization."

    return "\n".join(normalized_output)

print("normalize_data_for_llm function added to data_processing.py.")

def save_to_cache(data, key: str, cache_dir: str = 'cache/') -> None:
    """
    Saves data to a local cache directory.

    Args:
        data: The data to be cached (DataFrame, dict, etc.).
        key (str): A unique key to identify the cached data (e.g., ticker_period for OHLCV, ticker_financials for statements).
        cache_dir (str): The directory where cache files will be stored.
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, f"{key}.pkl")
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data for key '{key}' saved to cache at {file_path}")
    except Exception as e:
        print(f"Error saving data to cache for key '{key}': {e}")

def load_from_cache(key: str, cache_dir: str = 'cache/') -> object:
    """
    Loads data from a local cache directory.

    Args:
        key (str): The unique key for the cached data.
        cache_dir (str): The directory where cache files are stored.

    Returns:
        object: The deserialized data if found, otherwise None.
    """
    file_path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Data for key '{key}' loaded from cache at {file_path}")
            return data
        except Exception as e:
            print(f"Error loading data from cache for key '{key}': {e}")
            return None
    print(f"No cached data found for key '{key}' at {file_path}")
    return None

print("save_to_cache and load_from_cache functions added to data_processing.py.")
