import os
import argparse
import pandas as pd

# Import all necessary modules
from src.data.data_ingestion import fetch_ohlcv_data_with_cache, fetch_financial_statements_with_cache, calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands
from src.data.data_processing import validate_ohlcv_data, normalize_data_for_llm
from src.model.recommendation_engine import generate_stock_recommendation
from src.model.llm_integration import generate_analysis_prompt
from src.validation.report_generator import generate_investor_report
from src.notion.notion_integration import initialize_notion_client, create_notion_page, append_report_to_page

def main():
    parser = argparse.ArgumentParser(description="Generate stock recommendations and reports, then sync to Notion.")
    parser.add_argument("ticker", type=str, help="The stock ticker symbol (e.g., AAPL).")
    parser.add_argument("--period", type=str, default="1y", help="Period for historical data (e.g., 1y, 5y, max).")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for data caching.")
    args = parser.parse_args()

    ticker_symbol = args.ticker.upper()
    period = args.period
    cache_dir = args.cache_dir

    # 0. Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n--- Starting analysis for {ticker_symbol} ---")

    # 1. Fetch OHLCV Data
    print("\n--- Fetching OHLCV data ---")
    ohlcv_data = fetch_ohlcv_data_with_cache(ticker_symbol, period=period, cache_dir=cache_dir)
    if ohlcv_data.empty:
        print(f"Could not fetch OHLCV data for {ticker_symbol}. Exiting.")
        return

    # 2. Validate OHLCV Data
    print("\n--- Validating OHLCV data ---")
    try:
        validated_ohlcv_data = validate_ohlcv_data(ohlcv_data.copy())
    except ValueError as e:
        print(f"OHLCV data validation failed: {e}. Exiting.")
        return

    # 3. Calculate Technical Indicators
    print("\n--- Calculating technical indicators ---")
    indicators_df = validated_ohlcv_data.copy()
    indicators_df = calculate_sma(indicators_df)
    indicators_df = calculate_ema(indicators_df)
    indicators_df = calculate_rsi(indicators_df)
    indicators_df = calculate_macd(indicators_df)
    indicators_df = calculate_bollinger_bands(indicators_df)

    # 4. Fetch Financial Statements
    print("\n--- Fetching financial statements ---")
    financial_statements = fetch_financial_statements_with_cache(ticker_symbol, cache_dir=cache_dir)

    # 5. Normalize Data for LLM
    print("\n--- Normalizing data for LLM ---")
    normalized_data = normalize_data_for_llm(validated_ohlcv_data, financial_statements, indicators_df)
    if "No data available" in normalized_data:
        print("No sufficient data to normalize for LLM. Exiting.")
        return

    # 6. Generate Recommendation and Analysis using LLM
    print("\n--- Generating stock recommendation ---")
    recommendation_output = generate_stock_recommendation(ticker_symbol, normalized_data)
    full_analysis = recommendation_output.pop("full_analysis") # Extract full analysis

    if recommendation_output["recommendation"] == "Hold" and recommendation_output["confidence"] == 50 and \
       ("Failed to get" in recommendation_output["rationale"] or "Could not parse" in recommendation_output["rationale"]):
        print(f"Failed to generate a valid recommendation for {ticker_symbol}. Exiting.")
        return

    # 7. Generate Investor Report using LLM
    print("\n--- Generating investor report ---")
    investor_report = generate_investor_report(ticker_symbol, full_analysis, recommendation_output)

    # 8. Sync to Notion
    print("\n--- Syncing to Notion ---")
    try:
        notion_client = initialize_notion_client()
        page_id = create_notion_page(notion_client, ticker_symbol, recommendation_output)
        if page_id:
            append_report_to_page(notion_client, page_id, investor_report)
        else:
            print(f"Skipping report append: Failed to create Notion page for {ticker_symbol}.")
    except ValueError as e:
        print(f"Notion integration setup error: {e}. Skipping Notion sync.")
    except Exception as e:
        print(f"An error occurred during Notion sync: {e}.")

    print(f"\n--- Analysis for {ticker_symbol} completed ---")
    print(f"Final Recommendation: {recommendation_output['recommendation']} (Confidence: {recommendation_output['confidence']}%) - {recommendation_output['rationale']}")

if __name__ == "__main__":
    main()
