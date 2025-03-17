import os
import redis
from dotenv import load_dotenv

load_dotenv()

class RedisClient:
    """
    Redis client using connection pooling.
    The connection details are loaded from the environment variables.
    """
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = redis.Redis(
                host=os.getenv("REDIS_HOST"),
                port=int(os.getenv("REDIS_PORT")),
                username=os.getenv("REDIS_USERNAME"),
                password=os.getenv("REDIS_PASSWORD"),
                decode_responses=True
            )
        return cls._client
