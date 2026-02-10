import yfinance as yf
import pandas as pd
import numpy as np
from data_processing import load_from_cache, save_to_cache

def fetch_ohlcv_data(ticker_symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given stock ticker symbol.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
        period (str): The period for which to fetch data (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max').

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV data, or an empty DataFrame if fetching fails.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=period)
        if not data.empty:
            print(f"Successfully fetched OHLCV data for {ticker_symbol} for period {period}.")
            data.columns = [col.lower() for col in data.columns] # Standardize column names to lowercase
            return data
        else:
            print(f"No OHLCV data found for {ticker_symbol} for period {period}.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching OHLCV data for {ticker_symbol}: {e}")
        return pd.DataFrame()

print("data_ingestion.py created with fetch_ohlcv_data function.")

def fetch_financial_statements(ticker_symbol: str) -> dict:
    """
    Fetches financial statements (income statement, balance sheet, cash flow) for a given stock ticker symbol.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').

    Returns:
        dict: A dictionary containing 'income_statement', 'balance_sheet', and 'cash_flow' as pandas DataFrames.
              Returns empty DataFrames for any statement that cannot be fetched.
    """
    financial_statements = {
        "income_statement": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cash_flow": pd.DataFrame()
    }
    try:
        ticker = yf.Ticker(ticker_symbol)

        # Fetch income statement
        income_stmt = ticker.income_stmt
        if not income_stmt.empty:
            financial_statements["income_statement"] = income_stmt
            print(f"Successfully fetched income statement for {ticker_symbol}.")
        else:
            print(f"No income statement found for {ticker_symbol}.")

        # Fetch balance sheet
        balance_sheet = ticker.balance_sheet
        if not balance_sheet.empty:
            financial_statements["balance_sheet"] = balance_sheet
            print(f"Successfully fetched balance sheet for {ticker_symbol}.")
        else:
            print(f"No balance sheet found for {ticker_symbol}.")

        # Fetch cash flow statement
        cash_flow = ticker.cashflow
        if not cash_flow.empty:
            financial_statements["cash_flow"] = cash_flow
            print(f"Successfully fetched cash flow for {ticker_symbol}.")
        else:
            print(f"No cash flow statement found for {ticker_symbol}.")

    except Exception as e:
        print(f"Error fetching financial statements for {ticker_symbol}: {e}")

    return financial_statements

print("fetch_financial_statements function added to data_ingestion.py.")

def calculate_sma(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.DataFrame:
    """
    Calculates the Simple Moving Average (SMA) for a given column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data.
        window (int): The window size for the SMA calculation.
        column (str): The column name to calculate SMA on (e.g., 'close').

    Returns:
        pd.DataFrame: The DataFrame with the 'sma' column added.
    """
    if df.empty or column not in df.columns:
        print(f"DataFrame is empty or column '{column}' not found. Cannot calculate SMA.")
        return df
    if len(df) < window:
        print(f"Not enough data points ({len(df)}) to calculate SMA with window {window}.")
        df[f'sma_{window}'] = np.nan
        return df

    df[f'sma_{window}'] = df[column].rolling(window=window).mean()
    print(f"SMA with window {window} calculated.")
    return df

print("calculate_sma function added to data_ingestion.py.")

def calculate_ema(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.DataFrame:
    """
    Calculates the Exponential Moving Average (EMA) for a given column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data.
        window (int): The window size for the EMA calculation.
        column (str): The column name to calculate EMA on (e.g., 'close').

    Returns:
        pd.DataFrame: The DataFrame with the 'ema' column added.
    """
    if df.empty or column not in df.columns:
        print(f"DataFrame is empty or column '{column}' not found. Cannot calculate EMA.")
        return df
    if len(df) < window:
        print(f"Not enough data points ({len(df)}) to calculate EMA with window {window}.")
        df[f'ema_{window}'] = np.nan
        return df

    df[f'ema_{window}'] = df[column].ewm(span=window, adjust=False).mean()
    print(f"EMA with window {window} calculated.")
    return df

print("calculate_ema function added to data_ingestion.py.")

def calculate_rsi(df: pd.DataFrame, window: int = 14, column: str = 'close') -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI) for a given column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data.
        window (int): The window size for the RSI calculation.
        column (str): The column name to calculate RSI on (e.g., 'close').

    Returns:
        pd.DataFrame: The DataFrame with the 'rsi' column added.
    """
    if df.empty or column not in df.columns:
        print(f"DataFrame is empty or column '{column}' not found. Cannot calculate RSI.")
        return df
    if len(df) < window + 1:
        print(f"Not enough data points ({len(df)}) to calculate RSI with window {window}. Need at least {window + 1}.")
        df[f'rsi_{window}'] = np.nan
        return df

    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    print(f"RSI with window {window} calculated.")
    return df

print("calculate_rsi function added to data_ingestion.py.")

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'close') -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD).

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data.
        fast_period (int): The window size for the fast EMA.
        slow_period (int): The window size for the slow EMA.
        signal_period (int): The window size for the signal line EMA.
        column (str): The column name to calculate MACD on (e.g., 'close').

    Returns:
        pd.DataFrame: The DataFrame with 'macd', 'signal_line', and 'macd_histogram' columns added.
    """
    if df.empty or column not in df.columns:
        print(f"DataFrame is empty or column '{column}' not found. Cannot calculate MACD.")
        return df
    if len(df) < slow_period + signal_period:
        print(f"Not enough data points ({len(df)}) to calculate MACD with slow_period {slow_period} and signal_period {signal_period}.")
        df['macd'] = np.nan
        df['signal_line'] = np.nan
        df['macd_histogram'] = np.nan
        return df

    # Calculate Fast EMA
    ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()

    # Calculate Slow EMA
    ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD Line
    df['macd'] = ema_fast - ema_slow

    # Calculate Signal Line
    df['signal_line'] = df['macd'].ewm(span=signal_period, adjust=False).mean()

    # Calculate MACD Histogram
    df['macd_histogram'] = df['macd'] - df['signal_line']
    print(f"MACD calculated with fast_period {fast_period}, slow_period {slow_period}, signal_period {signal_period}.")
    return df

print("calculate_macd function added to data_ingestion.py.")

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std_dev: int = 2, column: str = 'close') -> pd.DataFrame:
    """
    Calculates Bollinger Bands (Middle Band, Upper Band, Lower Band).

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data.
        window (int): The window size for the moving average and standard deviation.
        num_std_dev (int): The number of standard deviations for the upper and lower bands.
        column (str): The column name to calculate Bollinger Bands on (e.g., 'close').

    Returns:
        pd.DataFrame: The DataFrame with 'middle_band', 'upper_band', and 'lower_band' columns added.
    """
    if df.empty or column not in df.columns:
        print(f"DataFrame is empty or column '{column}' not found. Cannot calculate Bollinger Bands.")
        return df
    if len(df) < window:
        print(f"Not enough data points ({len(df)}) to calculate Bollinger Bands with window {window}.")
        df['middle_band'] = np.nan
        df['upper_band'] = np.nan
        df['lower_band'] = np.nan
        return df

    # Calculate Middle Band (SMA)
    df['middle_band'] = df[column].rolling(window=window).mean()

    # Calculate Standard Deviation
    std_dev = df[column].rolling(window=window).std()

    # Calculate Upper and Lower Bands
    df['upper_band'] = df['middle_band'] + (std_dev * num_std_dev)
    df['lower_band'] = df['middle_band'] - (std_dev * num_std_dev)
    print(f"Bollinger Bands calculated with window {window} and {num_std_dev} standard deviations.")
    return df

print("calculate_bollinger_bands function added to data_ingestion.py.")



def fetch_ohlcv_data_with_cache(ticker_symbol: str, period: str = "1y", cache_dir: str = 'cache/') -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given stock ticker symbol, with caching.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
        period (str): The period for which to fetch data.
        cache_dir (str): Directory for cache files.

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV data, or an empty DataFrame if fetching fails.
    """
    cache_key = f"ohlcv_{ticker_symbol}_{period}"
    cached_data = load_from_cache(cache_key, cache_dir)

    if cached_data is not None:
        return cached_data

    print(f"Fetching OHLCV data for {ticker_symbol} from Yahoo Finance (no cache found).")
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=period)
        if not data.empty:
            data.columns = [col.lower() for col in data.columns] # Standardize column names
            save_to_cache(data, cache_key, cache_dir)
            print(f"Successfully fetched and cached OHLCV data for {ticker_symbol} for period {period}.")
            return data
        else:
            print(f"No OHLCV data found for {ticker_symbol} for period {period}.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching OHLCV data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def fetch_financial_statements_with_cache(ticker_symbol: str, cache_dir: str = 'cache/') -> dict:
    """
    Fetches financial statements with caching.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        cache_dir (str): Directory for cache files.

    Returns:
        dict: A dictionary containing 'income_statement', 'balance_sheet', and 'cash_flow' as pandas DataFrames.
    """
    cache_key = f"financials_{ticker_symbol}"
    cached_data = load_from_cache(cache_key, cache_dir)

    if cached_data is not None:
        return cached_data

    print(f"Fetching financial statements for {ticker_symbol} from Yahoo Finance (no cache found).")
    financial_statements = {
        "income_statement": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cash_flow": pd.DataFrame()
    }
    try:
        ticker = yf.Ticker(ticker_symbol)

        income_stmt = ticker.income_stmt
        if not income_stmt.empty: financial_statements["income_statement"] = income_stmt
        else: print(f"No income statement found for {ticker_symbol}.")

        balance_sheet = ticker.balance_sheet
        if not balance_sheet.empty: financial_statements["balance_sheet"] = balance_sheet
        else: print(f"No balance sheet found for {ticker_symbol}.")

        cash_flow = ticker.cashflow
        if not cash_flow.empty: financial_statements["cash_flow"] = cash_flow
        else: print(f"No cash flow statement found for {ticker_symbol}.")

        save_to_cache(financial_statements, cache_key, cache_dir)
        print(f"Successfully fetched and cached financial statements for {ticker_symbol}.")

    except Exception as e:
        print(f"Error fetching financial statements for {ticker_symbol}: {e}")

    return financial_statements

print("Caching logic integrated into data_ingestion.py.")

# Import caching functions from data_processing module
from src.data.data_processing import load_from_cache, save_to_cache

def fetch_ohlcv_data_with_cache(ticker_symbol: str, period: str = "1y", cache_dir: str = 'cache/') -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given stock ticker symbol, with caching.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
        period (str): The period for which to fetch data.
        cache_dir (str): Directory for cache files.

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV data, or an empty DataFrame if fetching fails.
    """
    cache_key = f"ohlcv_{ticker_symbol}_{period}"
    cached_data = load_from_cache(cache_key, cache_dir)

    if cached_data is not None:
        return cached_data

    print(f"Fetching OHLCV data for {ticker_symbol} from Yahoo Finance (no cache found).")
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=period)
        if not data.empty:
            data.columns = [col.lower() for col in data.columns] # Standardize column names
            save_to_cache(data, cache_key, cache_dir)
            print(f"Successfully fetched and cached OHLCV data for {ticker_symbol} for period {period}.")
            return data
        else:
            print(f"No OHLCV data found for {ticker_symbol} for period {period}.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching OHLCV data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def fetch_financial_statements_with_cache(ticker_symbol: str, cache_dir: str = 'cache/') -> dict:
    """
    Fetches financial statements with caching.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        cache_dir (str): Directory for cache files.

    Returns:
        dict: A dictionary containing 'income_statement', 'balance_sheet', and 'cash_flow' as pandas DataFrames.
    """
    cache_key = f"financials_{ticker_symbol}"
    cached_data = load_from_cache(cache_key, cache_dir)

    if cached_data is not None:
        return cached_data

    print(f"Fetching financial statements for {ticker_symbol} from Yahoo Finance (no cache found).")
    financial_statements = {
        "income_statement": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cash_flow": pd.DataFrame()
    }
    try:
        ticker = yf.Ticker(ticker_symbol)

        income_stmt = ticker.income_stmt
        if not income_stmt.empty: financial_statements["income_statement"] = income_stmt
        else: print(f"No income statement found for {ticker_symbol}.")

        balance_sheet = ticker.balance_sheet
        if not balance_sheet.empty: financial_statements["balance_sheet"] = balance_sheet
        else: print(f"No balance sheet found for {ticker_symbol}.")

        cash_flow = ticker.cashflow
        if not cash_flow.empty: financial_statements["cash_flow"] = cash_flow
        else: print(f"No cash flow statement found for {ticker_symbol}.")

        save_to_cache(financial_statements, cache_key, cache_dir)
        print(f"Successfully fetched and cached financial statements for {ticker_symbol}.")

    except Exception as e:
        print(f"Error fetching financial statements for {ticker_symbol}: {e}")

    return financial_statements

print("Caching logic integrated into data_ingestion.py.")
