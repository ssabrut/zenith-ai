from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    status: str = Field(..., description="Service status", examples=["healthy"])
    message: Optional[str] = Field(
        None, description="Status message", examples=["connected successfuly"]
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status", examples=["ok"])
    version: str = Field(..., description="Application version", examples=["0.1.0"])
    environment: str = Field(
        ..., description="Deployment environment", examples=["development"]
    )
    service_name: str = Field(
        ..., description="Service identifier", examples=["chatbot-api"]
    )
    services: Optional[Dict[str, ServiceStatus]] = Field(
        None, description="Individual service status"
    )

    class Config:
        json_schema_extra: Dict[str, Any] = {
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "environment": "development",
                "service_name": "chatbot-api",
                "services": {
                    "database": {
                        "status": "healthy",
                        "message": "Connected successfully",
                    }
                },
            }
        }
