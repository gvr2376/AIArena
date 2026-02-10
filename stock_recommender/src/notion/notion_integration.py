import os
from notion_client import Client

def initialize_notion_client() -> Client:
    """
    Initializes and returns a Notion client instance using an API token from environment variables.

    Returns:
        Client: An initialized Notion client.

    Raises:
        ValueError: If the NOTION_API_TOKEN environment variable is not set.
    """
    notion_api_token = os.getenv("NOTION_API_KEY")
    if not notion_api_token:
        raise ValueError("NOTION_API_KEY environment variable not set. Please set it securely.")

    try:
        notion_client = Client(auth=notion_api_token)
        print("Notion client initialized successfully.")
        return notion_client
    except Exception as e:
        raise Exception(f"Error initializing Notion client: {e}")

print("notion_integration.py created with initialize_notion_client function.")

def create_notion_page(client: Client, ticker_symbol: str, recommendation_details: dict) -> str:
    """
    Creates a new page in the Notion database with stock recommendation details.

    Args:
        client (Client): An initialized Notion client.
        ticker_symbol (str): The stock ticker symbol.
        recommendation_details (dict): Dictionary with 'recommendation', 'confidence', 'rationale'.

    Returns:
        str: The ID of the newly created page, or an empty string if creation fails.

    Raises:
        ValueError: If NOTION_DATABASE_ID environment variable is not set.
    """
    notion_database_id = os.getenv("NOTION_DATABASE_ID")
    if not notion_database_id:
        raise ValueError("NOTION_DATABASE_ID environment variable not set. Please set it securely.")

    recommendation = recommendation_details.get('recommendation', 'Hold')
    confidence = recommendation_details.get('confidence', 50)
    rationale = recommendation_details.get('rationale', 'No specific rationale provided.')

    try:
        new_page = client.pages.create(
            parent={
                "database_id": notion_database_id
            },
            properties={
                "Name": {
                    "title": [
                        {
                            "text": {
                                "content": f"{ticker_symbol} Stock Recommendation"
                            }
                        }
                    ]
                },
                "Ticker": {
                    "rich_text": [
                        {
                            "text": {
                                "content": ticker_symbol
                            }
                        }
                    ]
                },
                "Recommendation": {
                    "select": {
                        "name": recommendation
                    }
                },
                "Confidence": {
                    "number": confidence
                },
                "Rationale": {
                    "rich_text": [
                        {
                            "text": {
                                "content": rationale
                            }
                        }
                    ]
                }
            }
        )
        page_id = new_page["id"]
        print(f"Notion page created successfully for {ticker_symbol} with ID: {page_id}")
        return page_id
    except Exception as e:
        print(f"Error creating Notion page for {ticker_symbol}: {e}")
        return ""

print("create_notion_page function added to notion_integration.py.")

def append_report_to_page(client: Client, page_id: str, report_content: str) -> bool:
    """
    Appends the investor-friendly report content to an existing Notion page.

    Args:
        client (Client): An initialized Notion client.
        page_id (str): The ID of the Notion page to append content to.
        report_content (str): The full investor-friendly report content (Markdown formatted).

    Returns:
        bool: True if content was successfully appended, False otherwise.
    """
    blocks_to_append = []

    # Simple Markdown parsing to Notion blocks (this can be expanded for richer markdown)
    for line in report_content.split('\n'):
        line = line.strip()
        if not line: # Skip empty lines
            continue
        if line.startswith('### '):
            blocks_to_append.append({"object": "block", "type": "heading_3", "heading_3": {"rich_text": [{"type": "text", "text": {"content": line[4:].strip()}}]}})
        elif line.startswith('## '):
            blocks_to_append.append({"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:].strip()}}]}})
        elif line.startswith('# '):
            blocks_to_append.append({"object": "block", "type": "heading_1", "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]}})
        elif line.startswith('- '):
            blocks_to_append.append({"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]}})
        else:
            blocks_to_append.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]}})

    # Notion API has a limit of 100 blocks per append request. Split if necessary.
    # For simplicity, we'll append all in one go for now, assuming typical report length.
    # In a production system, this would need to be batched.
    try:
        if blocks_to_append:
            client.blocks.children.append(
                block_id=page_id,
                children=blocks_to_append
            )
        print(f"Report content successfully appended to Notion page {page_id}.")
        return True
    except Exception as e:
        print(f"Error appending report content to Notion page {page_id}: {e}")
        return False

print("append_report_to_page function added to notion_integration.py.")
