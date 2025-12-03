from fastapi import APIRouter
from typing import Dict
from loguru import logger

from core.schemas import HealthResponse, ServiceStatus
from core.dependencies import SettingDependencies
from core.services.deepinfra import factory as deepinfra_factory, DeepInfraClient

router = APIRouter()

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health and status of the API service and its dependencies",
    response_description="Service health information",
    tags=["health"]
)
async def health_check(settings: SettingDependencies) -> HealthResponse:
    services: Dict[str, ServiceStatus] = {}
    overall_status: str = "ok"

    try:
        # DeepInfra client
        deepinfra_client: DeepInfraClient = deepinfra_factory.make_deepinfra_client(model="openai/gpt-oss-20b")
        deepinfra_health: ServiceStatus = await deepinfra_client.health_check()
        services["deepinfra"] = ServiceStatus(
            status=deepinfra_health.status, message=deepinfra_health.message
        )

        if deepinfra_health.status != "healthy":
            overall_status = "degraded"
    except Exception as e:
        logger.error(str(e))
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services
    )
