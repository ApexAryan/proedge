from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://proedge:proedge@localhost:5432/proedge"
    database_url_sync: str = "postgresql://proedge:proedge@localhost:5432/proedge"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    secret_key: str = "dev-secret-change-in-production"

    # Sports Data
    sportradar_api_key: str = ""
    espn_api_base_url: str = "https://site.api.espn.com/apis/site/v2/sports"

    # Azure ML
    azure_subscription_id: str = ""
    azure_resource_group: str = "proedge-rg"
    azure_ml_workspace: str = "proedge-ws"
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""

    # Model Registry
    model_registry_path: str = "./models"
    active_model_version: str = "latest"

    # Drift
    drift_psi_threshold: float = 0.25
    drift_check_interval_hours: int = 24

    # Sports supported
    supported_sports: list[str] = ["nfl", "nba", "mlb"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
