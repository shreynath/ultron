from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    ollama_api_key: str = ""
    ollama_host: str = "https://ollama.com"
    ollama_model: str = "gpt-oss:120b-cloud"
    # Empty brain_* values reuse OLLAMA_MODEL / OLLAMA_HOST.
    brain_model: str = ""
    brain_model_base_url: str = ""
    brain_executive_timeout: float = 30.0
    # Empty planner values reuse the configured brain model and host.
    brain_planner_model: str = ""
    brain_planner_model_base_url: str = ""
    brain_planner_timeout: float = Field(default=30.0, gt=0)
    brain_planner_max_steps: int = Field(default=12, ge=1, le=12)
    # Internal action execution deadline. Model and user input never override it.
    brain_action_timeout: float = Field(default=30.0, gt=0, le=60)
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
