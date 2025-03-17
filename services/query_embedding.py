from typing import List, Optional
import numpy as np

class EmbeddingClient:
    """
    A client for generating text embeddings using either OpenAI or Cohere.
    
    Parameters:
      - vendor: A string identifier for the vendor ("openai" or "cohere").
      - api_key: The API key for the chosen vendor.
      - model: (Optional) The model name; defaults are provided if not specified.
    """
    
    def __init__(self, vendor: str, api_key: str, model: Optional[str] = None):
        self.vendor = vendor.lower()
        self.api_key = api_key
        self.model = model
        
        if self.vendor == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            if self.model is None:
                self.model = "text-embedding-3-small"
        elif self.vendor == "cohere":
            import cohere
            self.client = cohere.Client(api_key=self.api_key)
            if self.model is None:
                self.model = "embed-english-v3.0"
        elif self.vendor == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self.model = "sentence-transformers/all-MiniLM-L6-v2"
            self.model_instance = SentenceTransformer(self.model)
        else:
            raise ValueError(f"Unsupported vendor: {vendor}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text.
        
        Parameters:
          - text: The input text for which to generate the embedding.
        
        Returns:
          - A list of floats representing the embedding vector.
        """
        if self.vendor == "openai":
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            # Returns the embedding vector from the first result
            return response.data[0].embedding
        elif self.vendor == "cohere":
            # For Cohere, we wrap the text in a list.
            res = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query",
                embedding_types=["float"]
            )
            # Returns the embedding vector for the text (the first in the list)
            return res.embeddings[0]
        elif self.vendor == "sentence_transformers":
            embedding = self.model_instance.encode([text])[0]
            if isinstance(embedding, np.ndarray):
               embedding = embedding.tolist()
            return embedding

