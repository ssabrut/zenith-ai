from typing import Any, Dict, Tuple

import mlflow
import mlflow.tracking
from httpx import AsyncClient, ConnectError, RequestError, TimeoutException
from mlflow.exceptions import MlflowException

from core.config import Settings


class MLflowClient:
    """
    Manages client interactions with an MLflow tracking server and registry.

    This class handles the initialization of the MLflow client,
    provides methods for health checks, and loads models from the
    registry with an in-memory cache.

    Attributes:
        base_url (str): The URI for the MLflow tracking server.
        s3_endpoint_url (str): The URI for the S3-compatible artifact
                               registry.
        model_cache (Dict[str, Tuple[Any, str]]): An in-memory cache
            for loaded models.
    """

    base_url: str
    s3_endpoint_url: str
    model_cache: Dict[str, Tuple[Any, str]]
    _client: mlflow.tracking.MlflowClient

    def __init__(self, settings: Settings) -> None:
        """
        Initializes the MLflowClient and configures the tracking URI.

        Args:
            settings (Settings): The application configuration object.

        Raises:
            TypeError: If 'settings' is not an instance of the Settings class.
            RuntimeError: If configuration of the MLflow client fails (e.g.,
                          invalid URI, connection issues).
        """
        if not isinstance(settings, Settings):
            raise TypeError(
                "Argument 'settings' must be an instance of the Settings class"
            )

        self.base_url = settings.MLFLOW_TRACKING_URI
        self.s3_endpoint_url = settings.MLFLOW_S3_ENDPOINT_URL
        self.model_cache = {}

        try:
            self._configure_client()
            self._client = mlflow.tracking.MlflowClient(
                tracking_uri=self.base_url, registry_uri=self.base_url
            )
        except (MlflowException, Exception) as e:
            raise RuntimeError(f"Failed to initialize MLflowClient: {e}") from e

    async def health_check(self) -> Dict[str, str]:
        """
        Performs a health check against the MLflow tracking server.

        Returns:
            Dict[str, str]: A dictionary containing 'status' ('healthy' or
                            'unhealthy') and a descriptive 'message'.
        """
        try:
            async with AsyncClient(timeout=5.0) as client:
                url: str = self.base_url
                response = await client.get(url)

                if response.status_code == 200:
                    return {"status": "healthy", "message": "MLflow service is running"}
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"HTTP {response.status_code}",
                    }
        except (ConnectError, TimeoutException) as e:
            return {
                "status": "unhealthy",
                "message": f"Connection to MLflow failed: {e}",
            }
        except RequestError as e:
            return {"status": "unhealthy", "message": f"Request to MLflow failed: {e}"}

    def _configure_client(self) -> None:
        """
        Sets the global MLflow tracking URI.

        Raises:
            RuntimeError: If setting the tracking URI fails.
        """
        try:
            # Set the tracking URI for this session
            mlflow.set_tracking_uri(self.base_url)
        except MlflowException as e:
            raise RuntimeError(f"Failed to set MLflow tracking URI: {e}") from e

    @property
    def client(self) -> mlflow.tracking.MlflowClient:
        """
        Returns the underlying MLflow tracking client instance.

        Returns:
            mlflow.tracking.MlflowClient: The MLflow client.
        """
        return self._client