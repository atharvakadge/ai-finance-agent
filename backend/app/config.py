"""
Application configuration.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # API Keys
    groq_api_key: str

    # LLM Settings
    llm_model: str = "qwen/qwen3.6-27b"

    # Database
    database_url: str = "sqlite:///./finlens.db"

    # App Settings
    app_env: str = "development"
    app_debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()