import os
from typing import List
from pydantic import BaseModel, Field


class AtlasSettings(BaseModel):
    """
    Typed configuration for the ATLAS Central Orchestration Platform.
    Loads from environment variables with safe production-ready defaults.
    """
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development").lower())
    app_host: str = Field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    app_port: int = Field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"))

    ollama_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b"))

    database_dir: str = Field(default_factory=lambda: os.getenv("DATABASE_DIR", "storage"))
    trace_dir: str = Field(default_factory=lambda: os.getenv("TRACE_DIR", "traces"))

    api_auth_token: str = Field(default_factory=lambda: os.getenv("API_AUTH_TOKEN", "atlas_dev_secret_token"))
    simulation_mode: bool = Field(default_factory=lambda: os.getenv("SIMULATION_MODE", "true").lower() in ("true", "1", "yes"))
    replay_mode: bool = Field(default_factory=lambda: os.getenv("REPLAY_MODE", "false").lower() in ("true", "1", "yes"))

    # Configured trusted CORS origins (no unrestricted wildcard origin with credentials)
    cors_origins_raw: str = Field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGINS",
            "http://localhost:4200,http://127.0.0.1:4200,http://localhost:8000,http://127.0.0.1:8000",
        )
    )

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]

    def is_production(self) -> bool:
        return self.app_env in ("production", "prod")

    def is_simulation(self) -> bool:
        return self.simulation_mode and not self.replay_mode


# Global settings singleton loader
_settings_instance: AtlasSettings = None


def get_settings() -> AtlasSettings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AtlasSettings()
    return _settings_instance


def reset_settings(new_settings: AtlasSettings = None) -> AtlasSettings:
    global _settings_instance
    _settings_instance = new_settings
    return _settings_instance