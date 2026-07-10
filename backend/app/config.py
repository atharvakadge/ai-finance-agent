"""
Application configuration.

Loads settings from environment variables (defined in .env file).
Uses pydantic-settings for validation - if a required variable
is missing, the app crashes at startup with a clear error
instead of failing mysteriously later.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # API Keys
    groq_api_key: str

    # LLM Settings
    llm_model: str = "qwen/qwen3.6-27b"

    # App Settings
    app_env: str = "development"
    app_debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()