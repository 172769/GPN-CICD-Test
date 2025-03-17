from typing import List, Optional
from utils.langchain_client import LangChainClient
import traceback

class HallucinationCheckService:
    """
    Service for detecting hallucinations in LLM-generated responses.
    
    This service compares a response against a given context and returns a percentage score
    (0 = completely hallucinated, 100 = fully consistent with context).
    """
    
    def __init__(self, vendor: str, model_name: str, api_key: str):
        """
        Initializes the HallucinationCheckService with a LangChainClient.
        
        Parameters:
          - vendor: The LLM vendor (e.g., "openai").
          - model_name: The model to use.
          - api_key: The API key for the vendor.
        """
        self.langchain_client = LangChainClient(llm_vendor=vendor, model_name=model_name, api_key=api_key)
    
    async def check_hallucination(self, query: str, response: str, context: List[str], prompt: Optional[str] = None) -> float:
        """
        Evaluates the factual consistency of the response against the provided context.
        
        Parameters:
        - query: The original user query.
        - response: The LLM-generated response.
        - context: A list of strings containing the context (retrieved documents).
        - prompt: Optional prompt to guide the hallucination detection. If not provided, a default prompt is used.
        
        Returns:
        - A percentage score (float) where 100 means the response is fully supported by the context,
            and 0 means it is entirely hallucinated.
        """
        if prompt is None:
            prompt = (
                "# Hallucination Detection Evaluation\n\n"
                "## Query\n"
                "{query}\n\n"
                "## Response to Evaluate\n"
                "{response}\n\n"
                "## Reference Context\n"
                "{context}\n\n"
                "## Evaluation Instructions\n"
                "Analyze the response and determine if it is factually consistent with the provided context:\n\n"
                "1. Identify all factual claims made in the response\n"
                "2. Check each claim against the provided context\n"
                "3. Claims directly supported by context get full points\n"
                "4. Claims that contradict context get zero points\n"
                "5. Claims not mentioned in context but are common knowledge may get partial points\n"
                "6. Determine what percentage of the response is supported by the context\n\n"
                "## Scoring Methodology\n"
                "- 90-100%: Response is fully supported by context with minimal to no unsupported claims\n"
                "- 70-89%: Response is mostly supported with minor unsupported claims\n"
                "- 50-69%: Response has a mix of supported and unsupported claims"
                "- 20-49%: Response contains mostly unsupported claims with some supported elements\n"
                "- 0-19%: Response is almost entirely or completely hallucinated\n\n"
                "## Output Format\n"
                "Return ONLY a single number between 0 and 100 (without the % symbol) representing the factual consistency score.\n\n"
                "## Factual Consistency Score:"
            )
        
        result_str = await self.langchain_client.hallucination_check(query, response, context, prompt)
        
        # More robust parsing logic
        try:
            # Extract all numeric characters from the result
            import re
            numeric_chars = re.findall(r'\d+\.?\d*', result_str)
            
            if numeric_chars:
                # Take the first number found and convert to float
                score = float(numeric_chars[0])
                
                # Ensure the score is within the valid range
                score = max(0.0, min(100.0, score))
            
        except Exception :
            traceback.print_exc()
        
        return score


