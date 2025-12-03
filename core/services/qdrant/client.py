from loguru import logger

from typing import List, Any
from httpx import ConnectError, RequestError, TimeoutException, AsyncClient
from qdrant_client import QdrantClient as QdrantClientRemote
from qdrant_client.http.models import VectorParams, Distance

from core.config import Settings
from core.schemas import ServiceStatus
from core.services.deepinfra.factory import make_deepinfra_client

class QdrantClient:
    def __init__(self, settings: Settings) -> None:
        logger.info("Initializing QdrantClient...")
        if not isinstance(settings, Settings):
            logger.error("Invalid 'settings' argument provided to QdrantClient.")
            raise TypeError("Argument 'settings' must be an instance of the Settings class")

        self.base_url = settings.QDRANT_HOST
        self.qdrant_collection = settings.QDRANT_COLLECTION
        self.deepinfra_client = make_deepinfra_client("Qwen/Qwen3-Embedding-8B")
        
        logger.debug(f"Qdrant host set to: {self.base_url}")
        logger.debug(f"Qdrant collection set to: {self.qdrant_collection}")

        self.client = QdrantClientRemote(url=self.base_url)
        logger.info("Remote Qdrant client initialized.")

        try:
            collections_response = self.client.get_collections()
            existing_collections = [collection.name for collection in collections_response.collections]
            logger.debug(f"Found existing collections: {existing_collections}")

            if self.qdrant_collection not in existing_collections:
                logger.warning(f"Collection '{self.qdrant_collection}' not found. Creating it...")
                self.client.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=VectorParams(size=settings.qdrant_size, distance=Distance.COSINE)
                )
                logger.success(f"Successfully created collection '{self.qdrant_collection}'.")
            else:
                logger.info(f"Collection '{self.qdrant_collection}' already exists. No action needed.")
        except Exception as e:
            logger.critical(f"Failed to initialize or create Qdrant collection: {e}")
            raise

    async def health_check(self) -> ServiceStatus:
        logger.info("Performing health check on Qdrant service...")
        try:
            async with AsyncClient(timeout=5.0) as client:
                url = f"{self.base_url}"
                response = await client.get(url)

                if response.status_code == 200:
                    logger.success("Qdrant service is healthy.")
                    return ServiceStatus(
                        status="healthy",
                        message="Qdrant service is running"
                    )
                else:
                    logger.warning(f"Qdrant service is unhealthy with status code: {response.status_code}")
                    return ServiceStatus(
                        status="unhealthy",
                        message=f"HTTP {response.status_code}"
                    )
        except (ConnectError, TimeoutException) as e:
            logger.error(f"Health check failed due to connection/timeout error: {e}")
            return ServiceStatus(
                status="unhealthy",
                message=f"Connection to Qdrant failed: {e}"
            )
        except RequestError as e:
            logger.error(f"Health check failed due to request error: {e}")
            return ServiceStatus(
                status="unhealthy",
                message=f"Request to Qdrant failed: {e}"
            )