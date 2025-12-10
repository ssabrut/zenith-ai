from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_community.chat_models import ChatDeepInfra
from langchain_community.embeddings import DeepInfraEmbeddings

from core.config import Settings
from core.schemas import ServiceStatus

class DeepInfraClient:
    model: ChatDeepInfra

    def __init__(
        self,
        settings: Settings,
        model: Literal["openai/gpt-oss-20b", "Qwen/Qwen3-Embedding-8B", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
        temperature: float = 0,
        max_tokens: int = 2048
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError(
                "Argument 'settings' must be an instance of the Settings class"
            )

        self.deepinfra_api_token = settings.DEEPINFRA_API_TOKEN

        if model == "Qwen/Qwen3-Embedding-8B":
            self.model = DeepInfraEmbeddings(
                model_id=model,
                query_instruction="",
                embed_instruction="",
                deepinfra_api_token=self.deepinfra_api_token
            )
        else:
            self.model = ChatDeepInfra(
                model=model,
                temperature=temperature,
                deepinfra_api_token=self.deepinfra_api_token,
                max_tokens=max_tokens,
            )
            
    async def health_check(self) -> ServiceStatus:
        messages = [
            HumanMessage(
                content="Translate this sentence from English to French. I love programming."
            )
        ]

        try:
            result: LLMResult = await self.model.agenerate([messages])

            if (
                result.generations
                and result.generations[0]
                and result.generations[0][0].message.content
            ):

                return ServiceStatus(
                    status="healthy", message="DeepInfra service is running."
                )
            else:
                return ServiceStatus(
                    status="unhealthy",
                    message="DeepInfra service returned an empty or invalid response",
                )
        except Exception as e:
            return ServiceStatus(
                status="unhealthy", message=f"Connection to DeepInfra failed: {str(e)}"
            )