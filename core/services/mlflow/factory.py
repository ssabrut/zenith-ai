from functools import lru_cache

from core.config import Settings, get_settings
from core.services.mlflow import MLflowClient


@lru_cache(maxsize=1)
def make_mlflow_service() -> MLflowClient:
    try:
        settings: Settings = get_settings()
        return MLflowClient(settings)
    except (FileNotFoundError, ValueError) as e:
        # Catching common errors associated with config loading (file not found,
        # parsing errors) or client instantiation (invalid settings).
        raise RuntimeError(f"Failed to initialize MLflow service: {e}") from e
    except Exception as e:
        # A general catch-all for any other unexpected initialization error.
        raise RuntimeError(
            f"An unexpected error occurred during MLflow service initialization: {e}"
        ) from e
