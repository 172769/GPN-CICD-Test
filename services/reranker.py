# services/reranker.py

from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Reranker:
    """
    Reranker service using a specified vendor to rank document relevance.
    
    Currently supports:
      - cohere (using the rerank API)
    
    Methods:
      - rerank(query: str, documents: List[str], top_n: Optional[int] = None)
          Ranks the documents by their relevance to the provided query and returns
          a list of dictionaries containing the document index and relevance score.
    """
    
    def __init__(self, vendor: str, api_key: str, model: Optional[str] = None):
        """
        Initializes the reranker client.
        
        Parameters:
          - vendor: The LLM vendor, e.g. "cohere".
          - api_key: The API key for the vendor.
          - model: Optional; the model to use. For Cohere, defaults to "rerank-v3.5".
        """
        self.vendor = vendor.lower()
        self.api_key = api_key
        
        if self.vendor == "cohere":
            import cohere
            # Using the ClientV2 for reranking functionality.
            self.client = cohere.ClientV2(api_key=self.api_key)
            self.model = model if model is not None else "rerank-v3.5"
        else:
            raise ValueError(f"Unsupported vendor: {vendor}")
    
    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Ranks the provided documents by relevance to the query.
        
        Parameters:
          - query: The input query string.
          - documents: A list of document strings to be reranked.
          - top_n: Optional; the maximum number of top documents to return. 
                   If not provided, returns scores for all documents.
        
        Returns:
          - A list of dictionaries, each containing:
              * index: The index of the document in the original list.
              * relevance_score: The relevance score (as a float between 0 and 1).
          
          Example output:
            [
              {"index": 3, "relevance_score": 0.9990564},
              {"index": 4, "relevance_score": 0.7516481},
              ...
            ]
        """
        # If top_n is not specified, return scores for all documents.
        if top_n is None:
            top_n = len(documents)
        
        if self.vendor == "cohere":
            result = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n
            )
            # The result is expected to contain a key "results" with the list of rankings.
            ranked_results = []
            for idx, result in enumerate(result.results):
                ranked_results.append({
                    "index": result.index,
                    "relevance_score": result.relevance_score
                })
            return ranked_results
        else:
            raise ValueError(f"Vendor {self.vendor} not supported for reranking.")
