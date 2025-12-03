from typing import Literal

from langchain_community.chat_models import ChatDeepInfra
from langchain_community.embeddings import DeepInfraEmbeddings

from core.config import Settings


class DeepInfraClient:
    model: ChatDeepInfra

    def __init__(
        self,
        settings: Settings,
        model: Literal["openai/gpt-oss-20b", "Qwen/Qwen3-Embedding-8B"],
        temperature: float = 0,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError(
                "Argument 'settings' must be an instance of the Settings class"
            )

        self.deepinfra_api_token = settings.DEEPINFRA_API_TOKEN

        if model == "openai/gpt-oss-20b":
            self.model = ChatDeepInfra(
                model=model,
                temperature=temperature,
                deepinfra_api_token=self.deepinfra_api_token,
            )
        elif model == "Qwen/Qwen3-Embedding-8B":
            self.model = DeepInfraEmbeddings(
                model_id=model,
                query_instruction="",
                embed_instruction="",
                deepinfra_api_token=self.deepinfra_api_token
            )