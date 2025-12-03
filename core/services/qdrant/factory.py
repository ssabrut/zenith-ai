from functools import lru_cache

from core.config import get_settings
from core.services.qdrant import QdrantClient


@lru_cache(maxsize=1)
def make_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(settings)
