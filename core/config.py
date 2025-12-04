from typing import ClassVar
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class DefaultSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", extra="ignore", frozen=True, env_nested_delimiter="__"
    )

class Settings(DefaultSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "front-desk-agent"

    # Environment config
    IS_DOCKER: bool = Field(..., env="IS_DOCKER")

    # Deepinfra Config
    DEEPINFRA_API_TOKEN: str = Field(..., env="DEEPINFRA_API_TOKEN")
    DEEPINFRA_EMBEDDING_MODEL: str = Field(..., env="DEEPINFRA_EMBEDDING_MODEL")
    DEEPINFRA_CHAT_MODEL: str = Field(..., env="DEEPINFRA_CHAT_MODEL")

    # Qdrant config
    QDRANT_HOST: str = Field(..., env="QDRANT_HOST")
    QDRANT_PORT: int = Field(..., env="QDRANT_PORT")
    QDRANT_COLLECTION: str = Field(..., env="QDRANT_COLLECTION")

    # Postgres config
    POSTGRES_USER: str = Field(..., env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(..., env="POSTGRES_DB")

    # MLflow S3 Config
    AWS_ACCESS_KEY_ID: str = Field(..., env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    MLFLOW_S3_ENDPOINT_URL: str = Field(..., env="MLFLOW_S3_ENDPOINT_URL")
    MLFLOW_S3_IGNORE_TLS: str = Field(..., env="MLFLOW_S3_IGNORE_TLS")

    # MLflow Config
    MLFLOW_DB_USER: str = Field(..., env="MLFLOW_DB_USER")
    MLFLOW_DB_PASSWORD: str = Field(..., env="MLFLOW_DB_PASSWORD")
    MLFLOW_DB_NAME: str = Field(..., env="MLFLOW_DB_NAME")
    MLFLOW_TRACKING_URI: str = Field(..., env="MLFLOW_TRACKING_URI")

def get_settings() -> Settings:
    settings: Settings = Settings()
    new_mlflow_uri: str
    new_s3_uri: str
    new_qdrant_host: str

    print(f"IS_DOCKER: {settings.IS_DOCKER}")
    if settings.IS_DOCKER:
        new_mlflow_uri = settings.MLFLOW_TRACKING_URI
        new_s3_uri = settings.MLFLOW_S3_ENDPOINT_URL
        new_qdrant_host = settings.QDRANT_HOST
    else:
        new_mlflow_uri = "http://127.0.0.1:5050"
        new_s3_uri = "http://127.0.0.1:9002"
        new_qdrant_host = "localhost"
    
    updated_settings = settings.model_copy(
        update={
            "MLFLOW_TRACKING_URI": new_mlflow_uri,
            "MLFLOW_S3_ENDPOINT_URL": new_s3_uri,
            "QDRANT_HOST": new_qdrant_host,
        }
    )

    return updated_settings