from functools import lru_cache
from typing import Literal

from core.services.deepinfra import DeepInfraClient
from core.config import get_settings, Settings

@lru_cache(maxsize=1)
def make_deepinfra_client(model: Literal["openai/gpt-oss-20b", "Qwen/Qwen3-Embedding-8B"]) -> DeepInfraClient:
    settings: Settings = get_settings()
    return DeepInfraClient(settings=settings, model=model)