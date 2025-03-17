import json
import uuid
from typing import List, Dict, Tuple
import numpy as np
import asyncio
from utils.redis_client import RedisClient

# Set TTL for cache entries to 12 hours (in seconds)
TTL_SECONDS = 12 * 3600

class CacheService:
    """
    A service for managing a semantic cache in Redis.
    
    Each cache entry includes:
      - user_id: str
      - user_group: str
      - query: str
      - response: str
      - sources: dict or list (chunks with metadata)
      - reformulated_query_embeddings: List[float]
      - document_id: str
    """

    def __init__(self):
        self.client = RedisClient.get_client()

    def _generate_key(self, user_group: str) -> str:
        """
        Generates a unique cache key using the user group and a UUID.
        """
        cache_id = str(uuid.uuid4())
        return f"cache:{user_group}:{cache_id}"

    def insert_cache(self, user_id: str, user_group: str, query: str, response: str, 
                     sources: Dict, 
                     reformulated_query_embeddings: List[float],
                     document_id: str) -> None:
        """
        Inserts a new cache entry with a TTL of 12 hours.
        """
        key = self._generate_key(user_group)
        cache_entry = {
            "user_id": user_id,
            "user_group": user_group,
            "query": query,
            "response": response,
            "sources": sources,
            "reformulated_query_embeddings": reformulated_query_embeddings,
            document_id: document_id
        }
        
        # Save the cache entry as a JSON string with expiration TTL
        self.client.set(key, json.dumps(cache_entry), ex=TTL_SECONDS)

    def get_cache_by_user_group(self, user_group: str) -> List[Dict]:
        """
        Retrieves all cache entries for a given user group by using a key pattern.
        """
        pattern = f"cache:{user_group}:*"
        keys = self.client.keys(pattern)
        entries = []
        for key in keys:
            data = self.client.get(key)
            if data:
                entries.append(json.loads(data))
        return entries

    def compute_cosine_similarity(self, embedding_a: List[float], embedding_b: List[float]) -> float:
        """
        Computes cosine similarity between two embedding vectors and returns the score as a percentage.
        """
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = np.dot(a, b) / (norm_a * norm_b)
        return similarity * 100  # Convert to percentage

    async def get_similar_cache_entries(self, user_group: str, new_query_embedding: List[float], threshold: float = 90.0) -> List[Tuple[Dict, float]]:
        """
        Retrieves all cache entries for a given user group and computes the cosine similarity between
        the new query embedding and the stored original query embeddings.
        The similarity calculations are executed concurrently for minimal latency.
        
        Returns a list of tuples (cache_entry, similarity_score) for entries with a similarity >= threshold.
        """
        entries = self.get_cache_by_user_group(user_group)
        
        async def similarity_for_entry(entry: Dict) -> Tuple[Dict, float]:
            stored_embedding = entry.get("reformulated_query_embeddings")
            if stored_embedding:
                # Wrap the synchronous computation in an async call
                similarity = await asyncio.to_thread(self.compute_cosine_similarity, new_query_embedding, stored_embedding)
                return entry, similarity
            return entry, 0.0
        
        # Run all similarity calculations concurrently
        tasks = [similarity_for_entry(entry) for entry in entries]
        results = await asyncio.gather(*tasks)
        
        # Filter entries that meet the similarity threshold
        similar_entries = [(entry, score) for entry, score in results if score >= threshold]
        return similar_entries

    def delete_cache_by_document_id(self, document_id: str) -> None:
        """
        Deletes all cache entries that match the given document_id, regardless of user group.
        Iterates over all keys in the cache with the pattern 'cache:*'.
        """
        pattern = "cache:*"
        keys = self.client.keys(pattern)
        for key in keys:
            data = self.client.get(key)
            if data:
                entry = json.loads(data)
                if entry.get("document_id") == document_id:
                    self.client.delete(key)
