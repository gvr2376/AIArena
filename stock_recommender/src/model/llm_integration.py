import os
import openai
from openai import OpenAI

def get_llm_response(prompt: str, model_name: str = "gpt-4-turbo-preview") -> str:
    """
    Gets a response from the OpenAI LLM given a prompt.

    Args:
        prompt (str): The prompt to send to the LLM.
        model_name (str): The name of the LLM model to use (default: 'gpt-4-turbo-preview').

    Returns:
        str: The LLM's response or an error message if the API call fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not set."

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except openai.APIConnectionError as e:
        return f"OpenAI API connection error: {e}"
    except openai.RateLimitError as e:
        return f"OpenAI API request exceeded rate limit: {e}"
    except openai.APIStatusError as e:
        return f"OpenAI API status error: {e.status_code} - {e.response}"
    except Exception as e:
        return f"An unexpected error occurred with the OpenAI API: {e}"

print("llm_integration.py created with get_llm_response function.")

def generate_analysis_prompt(normalized_data: str, ticker_symbol: str) -> str:
    """
    Constructs a detailed prompt for the LLM to perform financial analysis.

    Args:
        normalized_data (str): The preprocessed financial and technical data.
        ticker_symbol (str): The stock ticker symbol being analyzed.

    Returns:
        str: A comprehensive prompt for the LLM.
    """
    prompt = f"""You are a seasoned financial analyst. Your task is to provide a comprehensive financial analysis for {ticker_symbol}.

Here is the latest financial data and technical indicators for {ticker_symbol}:

{normalized_data}

Based on the provided data, perform a thorough financial analysis. Your analysis should cover the following aspects:
1.  **Key Insights from Historical OHLCV Data and Technical Indicators:** Discuss significant price movements, trends, volatility, and signals from indicators like SMA, EMA, RSI, MACD, and Bollinger Bands.
2.  **Key Insights from Financial Statements:** Analyze the income statement, balance sheet, and cash flow statement. Highlight trends in revenue, profitability, assets, liabilities, equity, and cash flow generation.
3.  **Overall Financial Health and Performance:** Synthesize your findings to describe the company's current financial standing and recent performance.
4.  **Emerging Trends and Potential Risks:** Identify any patterns, opportunities, or potential risks that stand out from the data.
5.  **Rationale for a Potential Recommendation:** Based *solely* on this analysis, provide a detailed rationale that *supports* a potential 'Buy', 'Sell', or 'Hold' recommendation. **DO NOT explicitly state the 'Buy', 'Sell', or 'Hold' recommendation itself yet.** Focus on the reasons and evidence from the data that would lead to such a recommendation.

Your response should be professional, data-driven, and clearly structured.
"""
    return prompt

print("generate_analysis_prompt function added to llm_integration.py.")
