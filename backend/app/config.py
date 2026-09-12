from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    # Deprecated compatibility settings.  They are deliberately empty by
    # default and are never consulted by the local intelligence runtime.
    # Keeping the names for one release avoids making a user's .env invalid;
    # they do not enable a cloud fallback.
    ollama_api_key: str = ""
    ollama_host: str = ""
    ollama_model: str = ""
    # Empty brain_* values are resolved by the local model orchestrator.
    brain_model: str = ""
    brain_model_base_url: str = ""
    brain_executive_timeout: float = 30.0
    # Empty planner values reuse the configured brain model and host.
    brain_planner_model: str = ""
    brain_planner_model_base_url: str = ""
    brain_planner_timeout: float = Field(default=30.0, gt=0)
    brain_planner_max_steps: int = Field(default=12, ge=1, le=12)
    # Internal action execution deadline. Model and user input never override it.
    brain_action_timeout: float = Field(default=60.0, gt=0, le=120)
    # Durable learning store. This is intentionally separate from Voicebox data.
    brain_learning_db: str = "data/brain_learning.sqlite3"
    # Local Intelligence Platform. All locations are relative to ULTRON_HOME
    # unless an absolute path is explicitly supplied.
    ultron_home: str = "data/ultron"
    local_model_runtime: str = "mlx"
    local_model_keep_warm_seconds: int = Field(default=300, ge=0, le=86_400)
    local_model_max_tokens: int = Field(default=1024, ge=1, le=16_384)
    local_embedding_model_path: str = ""
    # Tools remain explicitly gated; this preserves the pre-existing local
    # weather/research capabilities for development. Set LOCAL_ONLY or OFFLINE
    # to block every outward-facing tool request.
    local_network_mode: str = "local_plus_tools"
    local_training_enabled: bool = False
    # Public-web research is deliberately bounded in runtime code, never by a
    # model-provided value. Set a token before exposing control endpoints.
    brain_research_api_token: str = ""
    brain_research_timeout: float = Field(default=30.0, gt=0, le=90)
    brain_research_max_response_bytes: int = Field(default=2_000_000, ge=10_000, le=8_000_000)
    brain_research_max_redirects: int = Field(default=3, ge=0, le=5)
    api_max_body_bytes: int = Field(default=20_000_000, ge=1_000_000, le=100_000_000)
    voicebox_api_url: str = "http://127.0.0.1:17493"
    voicebox_host: str = "127.0.0.1"
    voicebox_port: int = 17493
    voicebox_data_dir: str = ""
    # embedded: Ultron starts vendored Voicebox backend (no desktop app)
    # direct: in-process Voicebox Python modules (requires engine venv deps in same interpreter)
    # external: you manage the engine yourself (legacy Voicebox.app or remote URL)
    voice_engine_mode: str = "embedded"
    data_dir: str = "data"


settings = Settings()
