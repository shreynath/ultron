from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")

    ollama_api_key: str = ""
    ollama_host: str = "https://ollama.com"
    ollama_model: str = "gpt-oss:120b-cloud"
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
