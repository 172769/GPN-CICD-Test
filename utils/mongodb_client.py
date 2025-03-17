import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve MongoDB configuration from environment variables
MONGO_URI = os.getenv("MONGO_URI")
# commented that because it was bringing another db name from somewhere
DATABASE_NAME = os.getenv("MONGO_DB")
# DATABASE_NAME = "gpn_db"

# Optional: Production settings for connection pooling and timeouts
MAX_POOL_SIZE = int(os.getenv("MONGO_MAX_POOL_SIZE", 100))
MIN_POOL_SIZE = int(os.getenv("MONGO_MIN_POOL_SIZE", 10))
SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", 5000))

logger = logging.getLogger(__name__)

class MongoDBClient:
    """Async singleton class for MongoDB connection using Motor."""
    _client: AsyncIOMotorClient = None
    _db = None

    @classmethod
    async def connect(cls):
        if cls._client is None:
            try:
                cls._client = AsyncIOMotorClient(
                    MONGO_URI,
                    maxPoolSize=MAX_POOL_SIZE,
                    minPoolSize=MIN_POOL_SIZE,
                    serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS
                )
                cls._db = cls._client[DATABASE_NAME]
                # Validate the connection by issuing a ping command
                await cls._db.command("ping")
                logger.info("Async MongoDB connection established.")
            except Exception as e:
                logger.error("Failed to connect to MongoDB: %s", e)
                raise e
        return cls._db

    @classmethod
    async def get_client(cls):
        if cls._client is None:
            await cls.connect()
        return cls._client

    @classmethod
    async def get_database(cls):
        if cls._db is None:
            await cls.connect()
        return cls._db

    @classmethod
    async def get_collection(cls, collection_name: str):
        """
        Returns a collection object from the connected database.
        """
        db = await cls.get_database()
        return db[collection_name]

    @classmethod
    async def close_connection(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            logger.info("Async MongoDB connection closed.")
