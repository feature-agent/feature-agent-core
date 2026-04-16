"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Feature Agent Core configuration."""

    ANTHROPIC_API_KEY: str = ""
    GITHUB_TOKEN: str = ""
    GITHUB_REPO: str = ""
    NATS_URL: str = "nats://nats:4222"
    DATA_PATH: str = "/data"
    TARGET_REPO_PATH: str = "/target-repo"
    LOG_LEVEL: str = "INFO"
    ENV: str = "development"

    NATS_TASK_SUBJECT: str = "agent.tasks.incoming"
    NATS_STREAM_NAME: str = "AGENT_TASKS"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
