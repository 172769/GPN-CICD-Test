from typing import List, Optional
from utils.langchain_client import LangChainClient

class QueryReformulationService:
    """
    Service for query reformulation using LangChainClient.
    
    This service takes a user's query and a short-term memory (list of recent messages) 
    and reformulates the query to add context or improve its language for vector search retrieval.
    """
    
    def __init__(self, vendor: str, model_name: str, api_key: str):
        """
        Initializes the QueryReformulationService with a LangChainClient.
        
        Parameters:
          - vendor: The LLM vendor (e.g., "openai").
          - model_name: The model to be used.
          - api_key: The API key for the vendor.
        """
        self.langchain_client = LangChainClient(llm_vendor=vendor, model_name=model_name, api_key=api_key)
        
    async def reformulate_query(self, query: str, short_term_memory: List[str], prompt: Optional[str] = None) -> str:
        """
        Reformulates the given query by incorporating short term memory (recent conversation history).
        
        Parameters:
          - query: The original user query.
          - short_term_memory: A list of recent messages (could be empty if none).
          - prompt: Optional custom prompt to guide the reformulation process. If not provided, 
                    a default prompt is used.
                    
        Returns:
          - A string containing the reformulated query.
        """
        if prompt is None:
          prompt = (
              "# Query Reformulation Task\n\n"
              "## Context : \n"
              "You are an AI assistant helping to reformulate a user query to improve retrieval and search from a vector database.\n\n"
              "## Conversation History\n{history}\n\n"
              "## Original Query\n{query}\n\n"
              "## Instructions\n"
              "1. Analyze if this query is a follow-up question that depends on previous context\n"
              "2. If it's a follow-up, explicitly incorporate relevant entities and context from the conversation history, if it is not a follow up question do not add context from previous history\n"
              "3. Expand any ambiguous terms, acronyms, or pronouns (like 'it', 'they', 'this')\n"
              "4. Add specificity to improve vector search accuracy\n"
              "5. Maintain the original intent and core question\n"
              "6. Do not add unnecessary information that could dilute the search\n"
              "7. Keep the reformulation concise and focused\n\n"
              "**very important** :  Do not always add context from previous history if the new query is not related to the previous ones\n if the new query has different intent, do not add context from previous history\n"
              "## Reformulated Query : \n"
          )
        
        reformulated = await self.langchain_client.reformulate_query(query, short_term_memory, prompt)
        return reformulated



