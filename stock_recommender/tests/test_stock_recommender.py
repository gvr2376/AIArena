import pytest
import pandas as pd
import numpy as np
import os
import shutil

# Adjust imports to reflect the new file structure
from src.data.data_ingestion import fetch_ohlcv_data_with_cache, calculate_sma, calculate_ema
from src.data.data_ingestion import fetch_financial_statements_with_cache
from src.data.data_processing import save_to_cache, load_from_cache, validate_ohlcv_data, normalize_data_for_llm
# from recommendation_engine import generate_stock_recommendation # Not directly tested in integration, but good to have

# Define a cache directory for testing
TEST_CACHE_DIR = "./stock_recommender/cache/test_cache"

@pytest.fixture(autouse=True)
def cleanup_cache():
    """Fixture to clean up the test cache directory before and after each test."""
    if os.path.exists(TEST_CACHE_DIR):
        shutil.rmtree(TEST_CACHE_DIR)
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_CACHE_DIR):
        shutil.rmtree(TEST_CACHE_DIR)


def test_fetch_ohlcv_data_with_cache_unit():
    """Unit test for fetch_ohlcv_data_with_cache using mock data."""
    ticker_symbol = "TEST_OHLCV"
    period = "1y"
    cache_key = f"ohlcv_{ticker_symbol}_{period}"

    # a. Use mock OHLCV data
    mock_ohlcv_data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [102.0, 103.0, 104.0, 105.0, 106.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))

    # b. Pre-populate the cache
    save_to_cache(mock_ohlcv_data, cache_key, TEST_CACHE_DIR)

    # c. Call the function
    fetched_data = fetch_ohlcv_data_with_cache(ticker_symbol, period, TEST_CACHE_DIR)

    # d. Assertions
    assert not fetched_data.empty
    pd.testing.assert_frame_equal(fetched_data, mock_ohlcv_data)
    print(f"Unit test passed for {ticker_symbol} OHLCV data.")

def test_end_to_end_integration():
    """Integration test for a simplified end-to-end data processing flow."""
    ticker_symbol = "TEST_E2E"
    period = "1y"
    ohlcv_cache_key = f"ohlcv_{ticker_symbol}_{period}"
    financials_cache_key = f"financials_{ticker_symbol}"

    # a. Create mock OHLCV data
    mock_ohlcv_data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [102.0, 103.0, 104.0, 105.0, 106.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))

    # Create mock financial statements
    mock_income_stmt = pd.DataFrame({
        pd.to_datetime('2022-12-31'): [100000, 50000, 20000],
        pd.to_datetime('2021-12-31'): [90000, 45000, 18000]
    }, index=['Total Revenue', 'Gross Profit', 'Net Income'])
    mock_financials = {
        "income_statement": mock_income_stmt,
        "balance_sheet": pd.DataFrame(),
        "cash_flow": pd.DataFrame()
    }

    # b. Store mock data in cache
    save_to_cache(mock_ohlcv_data, ohlcv_cache_key, TEST_CACHE_DIR)
    save_to_cache(mock_financials, financials_cache_key, TEST_CACHE_DIR)

    # c. Call data ingestion functions to load from cache
    ohlcv_data = fetch_ohlcv_data_with_cache(ticker_symbol, period, TEST_CACHE_DIR)
    financial_statements = fetch_financial_statements_with_cache(ticker_symbol, TEST_CACHE_DIR)

    # d. Call validate OHLCV Data
    validated_ohlcv_data = validate_ohlcv_data(ohlcv_data.copy())
    assert not validated_ohlcv_data.empty

    # e. Call one or two calculate indicator functions
    indicators_df = validated_ohlcv_data.copy()
    indicators_df = calculate_sma(indicators_df)
    indicators_df = calculate_ema(indicators_df)
    assert 'sma_20' in indicators_df.columns
    assert 'ema_20' in indicators_df.columns

    # f. Call normalize_data_for_llm
    normalized_data = normalize_data_for_llm(validated_ohlcv_data, financial_statements, indicators_df)

    # g. Assert that the normalized data is a non-empty string
    assert isinstance(normalized_data, str)
    assert len(normalized_data) > 0
    assert "### Historical OHLCV Data" in normalized_data
    assert "### Technical Indicators" in normalized_data
    assert "### Financial Statements" in normalized_data
    print(f"Integration test passed for {ticker_symbol} end-to-end flow.")

print("test_stock_recommender.py created with unit and integration tests.")
