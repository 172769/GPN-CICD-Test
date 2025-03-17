import os
import time
from typing import List, Dict, Tuple
from utils.mongodb_client import MongoDBClient
from dotenv import load_dotenv
load_dotenv()

class VectorSearchService:
    """
    Service for performing vector search on documents stored in MongoDB.
    
    The documents are expected to have the following fields:
      - chunk: Text content of the document chunk.
      - document_id: The document's id or name.
      - page_number: An integer indicating the page number.
      - user_groups: A list of user groups authorized for the document.
      - embedding: The vector representation used for search.
      - file_link: A link to the source file for grounding.
    """

    def __init__(self):
        # Load configuration from environment variables
        self.collection_name = os.getenv("VECTOR_SEARCH_COLLECTION")
        self.filter_field = os.getenv("VECTOR_SEARCH_FILTER_FIELD")
        self.embedding_path = os.getenv("VECTOR_SEARCH_PATH")
        self.vector_index_name = os.getenv("VECTOR_INDEX_NAME")
        

    async def search(self, query_embedding: List[float], authorization_filter: str, limit: int = 3) -> Tuple[List[Dict], float]:
        """
        Perform a vector search using the $vectorSearch stage with an authorization filter.
        Returns a tuple containing the search results and the execution time in milliseconds.

        The returned document includes:
          - chunk (text)
          - document_id (document id or name)
          - page_number (int)
          - user_groups (list)
          - file_link (text for grounding)
          - score (vector search score)
        """
        # Retrieve the collection asynchronously
        collection = await MongoDBClient.get_collection(self.collection_name)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": self.embedding_path,
                    "filter": {
                        self.filter_field: {"$in": [authorization_filter]}
                    },
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "content": 1,
                    "document_name": 1,
                    "document_url": 1,
                    "document_type": 1,
                    "page_number": 1,
                    "group_id": 1,
                    "file_link": 1,
                    "_id": 0
                }
            }
        ]
        
        start_time = time.perf_counter()
        results = []
        cursor = collection.aggregate(pipeline)
        async for doc in cursor:
            results.append(doc)
        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000 
        return results, duration_ms
