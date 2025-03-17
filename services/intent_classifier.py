from typing import Optional
from utils.langchain_client import LangChainClient

class IntentClassificationService:
    """
    Service for classifying user query intent using LangChainClient.
    
    For this application, we distinguish between:
      - "domain": Queries related to bank policies, credit cards, and other financial topics.
      - "non-domain": All other queries.
    """
    
    def __init__(self, vendor: str, model_name: str, api_key: str):
        """
        Initializes the IntentClassificationService with a LangChainClient.
        
        Parameters:
          - vendor: The LLM vendor (e.g., "openai").
          - model_name: The model name to be used.
          - api_key: The API key for the vendor.
        """
        self.langchain_client = LangChainClient(llm_vendor=vendor, model_name=model_name, api_key=api_key)
    
    async def classify_intent(self, query: str, prompt: Optional[str] = None) -> str:
        """
        Classifies the intent of the given query as either "domain" or "non-domain."
        
        Parameters:
          - query: The user query string.
          - prompt: Optional custom prompt to guide the classification. If not provided, a default prompt is used.
        
        Returns:
          - A string: either "domain" or "non-domain".
        """
        if prompt is None:
            prompt = (
            "# Intent Classification Task\n\n"
            "## Query to Classify :\n{query}\n\n"
            "## Classification Instructions\n"
            "Analyze the query and determine if it falls into one of these categories:\n\n"
            "### Greeting (Respond with 'greeting' only)\n"
            "- Hello, hi, hey, good morning/afternoon/evening\n"
            "- Initial conversation starters\n"
            "- Welcome messages\n"
            "- How are you/how's it going\n"
            "- Introductions\n"
            "### Domain (Respond with 'domain' only)\n"
            "- Banking products and services (accounts, loans, mortgages, credit cards)\n"
            "- Financial transactions and operations\n"
            "- Banking policies, fees, and rates\n"
            "- Financial regulations and compliance\n"
            "- Customer account inquiries and management\n"
            "- Banking technology and digital services\n\n"
            "### Non-Domain (Respond with 'non-domain' only)\n"
            "- General conversation and small talk\n"
            "- Questions about topics unrelated to banking/finance\n"
            "- Personal questions about the assistant\n"
            "- Technical support for non-banking systems\n"
            "- Requests that don't pertain to financial services\n\n"
            "## Output Format\n"
            "Respond with exactly one word: either 'domain' or 'non-domain'\n\n"
            "## Classification:\n"
        )
        # Call the LangChainClient's classify_intent method.
        intent = await self.langchain_client.classify_intent(query, prompt)
        return intent.strip()


    
